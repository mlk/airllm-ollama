"""Thread-safe wrapper around a single AirLLM model.

The wrapper owns exactly one AirLLM model for the lifetime of the process: the
weights are loaded lazily on the first request and then reused, so a large
checkpoint is never materialised twice.

AirLLM itself is an optional dependency (it pulls in ``torch`` and
``transformers``), therefore it is imported lazily inside
:func:`default_model_factory` instead of at module import time.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from airllm_ollama.config import Settings, get_settings
from airllm_ollama.exceptions import ModelInitializationError, ModelMismatchError


@dataclass(frozen=True)
class GenerationResult:
    """Result of a single generation, including Ollama style statistics."""

    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    load_duration_ns: int = 0
    prompt_duration_ns: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0


def is_airllm_installed() -> bool:
    """Return whether the optional AirLLM runtime is available.

    The package is *not* imported: importing ``airllm`` pulls in torch and
    takes seconds, which is far too slow for a status endpoint.
    """
    try:
        return importlib.util.find_spec("airllm") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


class TokenizerLike(Protocol):
    """Minimal surface of the Hugging Face tokenizer used by the engine."""

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        """Tokenize ``text`` and return a mapping of tensor inputs."""
        ...

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Decode a batch of token ids into strings."""
        ...


class AirLLMModelLike(Protocol):
    """Minimal surface of an AirLLM model used by the engine."""

    @property
    def tokenizer(self) -> TokenizerLike:
        """Return the tokenizer bound to the model."""
        ...

    def generate(self, *args: object, **kwargs: object) -> object:
        """Run generation for the given (already tokenized) inputs."""
        ...


ModelFactory = Callable[[Settings], AirLLMModelLike]


def _split_model_reference(name: str) -> tuple[str, str | None]:
    """Split an Ollama model reference into its name and its optional tag.

    Args:
        name: The model reference, e.g. ``llama3.2:1b``.

    Returns:
        A tuple of the base name and the tag (``None`` when the reference has
        no tag).
    """
    head, separator, tail = name.rpartition(":")
    if not separator:
        return name, None
    return head, tail


def normalize_model_name(name: str) -> str:
    """Return the base name of an Ollama model reference, without its tag.

    Ollama references look like ``name:tag`` (``llama3.2:1b``); the tag is
    separated from the name by the last colon.

    Args:
        name: The model reference to strip.

    Returns:
        The model name without its tag.
    """
    base, _tag = _split_model_reference(name)
    return base


def model_names_match(requested: str, served: str) -> bool:
    """Return whether ``requested`` refers to the model this server serves.

    The server hosts a single checkpoint, so a request for ``name:any-tag``
    resolves to that checkpoint as long as the base names match. When the
    served model declares an explicit tag (``llama3.2:1b``) the request has to
    use that exact tag, so that a request for another variant
    (``llama3.2:70b``) is rejected instead of silently answered with a
    different model.

    Args:
        requested: The model reference asked for by the client.
        served: The model reference this server is configured with.

    Returns:
        ``True`` when the request refers to the served model.
    """
    served_base, served_tag = _split_model_reference(served)
    requested_base, requested_tag = _split_model_reference(requested)
    if served_tag is None:
        # The served model has no explicit tag: any tag resolves to it.
        return requested_base == served_base
    if requested_base != served_base:
        return False
    if requested_tag is None:
        # The request has no tag: it refers to the served variant.
        return True
    return requested_tag == served_tag


def default_model_factory(settings: Settings) -> AirLLMModelLike:
    """Instantiate the real AirLLM model for the configured path.

    Args:
        settings: Runtime settings holding the model path and runtime options.

    Returns:
        The loaded AirLLM model.

    Raises:
        ModelInitializationError: If ``airllm`` is not installed or the model
            cannot be loaded.
    """
    try:
        from airllm import AutoModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ModelInitializationError(
            "The 'airllm' package is not installed. Install it with "
            "`uv sync --extra airllm` to enable local inference."
        ) from exc

    model_path = settings.resolved_model_path
    shard_options: dict[str, object] = {}
    if settings.layer_shards_saving_path:
        shard_options["layer_shards_saving_path"] = settings.layer_shards_saving_path
    try:
        model = AutoModel.from_pretrained(
            model_path,
            device=settings.device,
            max_seq_len=settings.max_seq_len,
            **shard_options,
        )
    except Exception as exc:
        raise ModelInitializationError(
            f"Failed to initialise the AirLLM model '{model_path}': {exc}"
        ) from exc
    return cast("AirLLMModelLike", model)


class AirLLMEngine:
    """Owns a single AirLLM model and serialises access to it.

    The model is loaded lazily on first use.

    Two locks are used on purpose:

    * ``_generation_lock`` serialises the inference calls. It is held for the
      *whole* inference pass, not just for the model access: AirLLM streams the
      layer shards of the checkpoint through a single set of forward hooks that
      move weights between the meta device and the accelerator while they run.
      Two concurrent generations would interleave those swaps and corrupt the
      model state, so the lock has to cover the full call.
    * ``_load_lock`` only guards the creation of the model, so that helpers
      such as :meth:`load` do not have to wait for a running inference to
      finish before they can hand out the already loaded model.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        """Create an engine.

        Args:
            settings: Optional settings override; defaults to the process settings.
            model_factory: Optional factory used to build the model. Injecting a
                factory keeps the test-suite free of the heavy AirLLM runtime.
        """
        self._settings = settings or get_settings()
        self._model_factory = model_factory or default_model_factory
        self._model: AirLLMModelLike | None = None
        self._load_duration_ns = 0
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._unload_lock = threading.Lock()
        self._unload_timer: threading.Timer | None = None

    @property
    def settings(self) -> Settings:
        """Return the settings this engine was built with."""
        return self._settings

    @property
    def model_name(self) -> str:
        """Return the name of the single model served by this engine."""
        return self._settings.model_name

    @property
    def model_path(self) -> str:
        """Return the local path or Hugging Face repo id of the model."""
        return self._settings.resolved_model_path

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is currently loaded in memory."""
        return self._model is not None

    @property
    def is_available(self) -> bool:
        """Return whether the AirLLM runtime is installed.

        A service without the AirLLM extra still answers requests, but every
        inference attempt fails with a :class:`ModelInitializationError`.
        """
        return is_airllm_installed()

    def load(self) -> AirLLMModelLike:
        """Load the model if needed and return it.

        Loading is idempotent: the model is created exactly once per engine.
        The lock is only held while the model is created, so this call does not
        block on a running inference.
        """
        with self._load_lock:
            if self._model is None:
                started = time.perf_counter_ns()
                self._model = self._model_factory(self._settings)
                self._load_duration_ns = time.perf_counter_ns() - started
            return self._model

    def unload(self) -> None:
        """Drop the loaded model so the next request reloads it.

        Waits for a running inference to finish so that the model is never
        released while it is in use.
        """
        with self._generation_lock, self._load_lock:
            self._model = None

    def schedule_unload(self, delay_seconds: float) -> None:
        """Unload the model once it has been idle for ``delay_seconds``.

        A pending unload is cancelled first, so the timer always measures the
        idle time since the last request.

        Args:
            delay_seconds: Number of seconds to wait. A value of ``0`` unloads
                the model immediately.
        """
        with self._unload_lock:
            self._cancel_unload_timer()
            if delay_seconds <= 0:
                self.unload()
                return
            timer = threading.Timer(delay_seconds, self.unload)
            timer.daemon = True
            self._unload_timer = timer
            timer.start()

    def cancel_scheduled_unload(self) -> None:
        """Cancel a pending unload, e.g. because a new request arrived."""
        with self._unload_lock:
            self._cancel_unload_timer()

    def _cancel_unload_timer(self) -> None:
        """Cancel the pending unload timer (the lock is held by the caller)."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def ensure_model_allowed(self, model: str | None) -> None:
        """Validate that ``model`` is the model served by this engine.

        Args:
            model: The model name requested by the client. ``None`` means the
                client did not specify one, which is accepted.

        Raises:
            ModelMismatchError: If the requested model is not the served model.
        """
        if model is None:
            return
        if not model_names_match(model, self._settings.model_name):
            raise ModelMismatchError(
                f"Model '{model}' is not available. This server serves a single model: "
                f"'{self._settings.model_name}'."
            )

    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate text for ``prompt``.

        Args:
            prompt: The prompt to complete.
            **kwargs: Generation options forwarded to AirLLM. The reserved
                ``model`` keyword is used to validate the requested model.

        Returns:
            The generated text.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        return self.generate_result(prompt, **kwargs).text

    def generate_result(self, prompt: str, **kwargs: object) -> GenerationResult:
        """Generate text for ``prompt`` and report the generation statistics.

        Args:
            prompt: The prompt to complete.
            **kwargs: Generation options forwarded to AirLLM. The reserved
                ``model`` keyword is used to validate the requested model.

        Returns:
            The generated text together with token counts and timings.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        requested_model = cast("str | None", kwargs.pop("model", None))
        self.ensure_model_allowed(requested_model)
        self.cancel_scheduled_unload()
        with self._generation_lock:
            started = time.perf_counter_ns()
            model = self.load()
            load_duration = self._load_duration_ns
            prompt_started = time.perf_counter_ns()
            raw_output, prompt_tokens = self._run(model, prompt, **kwargs)
            prompt_duration = time.perf_counter_ns() - prompt_started
            text = self._decode(model, raw_output)
            generated_tokens = self._count_generated_tokens(model, raw_output, prompt_tokens)
            eval_duration = time.perf_counter_ns() - prompt_started
            return GenerationResult(
                text=text,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                load_duration_ns=load_duration,
                prompt_duration_ns=prompt_duration,
                eval_duration_ns=eval_duration,
                total_duration_ns=time.perf_counter_ns() - started,
            )

    @staticmethod
    def _run(model: AirLLMModelLike, prompt: str, **kwargs: object) -> tuple[object, int]:
        """Tokenize, generate and return the raw output plus the prompt length."""
        encoded = model.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        # AirLLM backends accept the token ids positionally; the transformers
        # based one forwards them to `transformers.GenerationMixin.generate`.
        raw_output = model.generate(input_ids, attention_mask=attention_mask, **kwargs)
        return raw_output, _sequence_length(input_ids)

    @staticmethod
    def _count_generated_tokens(
        model: AirLLMModelLike, raw_output: object, prompt_tokens: int
    ) -> int:
        """Return the number of tokens the model generated."""
        if isinstance(raw_output, str):
            # Backends that decode themselves only return the new tokens.
            return _sequence_length(model.tokenizer(raw_output)["input_ids"])
        return max(_sequence_length(raw_output) - prompt_tokens, 0)

    @staticmethod
    def _decode(model: AirLLMModelLike, raw_output: object) -> str:
        """Decode raw generation output into text."""
        if isinstance(raw_output, str):
            return raw_output
        decoded = model.tokenizer.batch_decode(raw_output, skip_special_tokens=True)
        if not decoded:
            return ""
        return decoded[0]


def _sequence_length(token_ids: object) -> int:
    """Return the number of tokens in a tensor or a (possibly nested) sequence."""
    shape = getattr(token_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if isinstance(token_ids, (list, tuple)):
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            return len(token_ids[0])
        return len(token_ids)
    return 0


_engine: AirLLMEngine | None = None
_engine_lock = threading.Lock()


def get_engine(settings: Settings | None = None) -> AirLLMEngine:
    """Return the process-wide engine, creating it on first use.

    Args:
        settings: Optional settings used **only** when the engine is created
            for the first time; later calls return the existing instance.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = AirLLMEngine(settings)
        return _engine


def reset_engine() -> None:
    """Discard the cached engine so the next call creates a fresh one."""
    global _engine
    with _engine_lock:
        _engine = None
