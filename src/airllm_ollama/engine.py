"""Thread-safe wrapper around a single AirLLM model.

The wrapper owns exactly one AirLLM model for the lifetime of the process: the
weights are loaded lazily on the first request and then reused, so a large
checkpoint is never materialised twice.

AirLLM itself is an optional dependency (it pulls in ``torch`` and
``transformers``), therefore it is imported lazily inside
:func:`default_model_factory` instead of at module import time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from airllm_ollama.config import Settings, get_settings
from airllm_ollama.exceptions import ModelInitializationError, ModelMismatchError

LATEST_TAG = ":latest"


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


def normalize_model_name(name: str) -> str:
    """Strip Ollama's ``:latest`` tag so ``model`` and ``model:latest`` match."""
    return name.removesuffix(LATEST_TAG)


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

    The model is loaded lazily on first use. All model interactions are guarded
    by a re-entrant lock: AirLLM streams layer shards through a single set of
    hooks, so concurrent generations would corrupt the shared state.
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
        self._lock = threading.RLock()

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

    def load(self) -> AirLLMModelLike:
        """Load the model if needed and return it.

        Loading is idempotent: the model is created exactly once per engine.
        """
        with self._lock:
            if self._model is None:
                self._model = self._model_factory(self._settings)
            return self._model

    def unload(self) -> None:
        """Drop the loaded model so the next request reloads it."""
        with self._lock:
            self._model = None

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
        if normalize_model_name(model) != normalize_model_name(self._settings.model_name):
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
        requested_model = cast("str | None", kwargs.pop("model", None))
        self.ensure_model_allowed(requested_model)
        with self._lock:
            model = self.load()
            return self._generate(model, prompt, **kwargs)

    @staticmethod
    def _generate(model: AirLLMModelLike, prompt: str, **kwargs: object) -> str:
        """Tokenize, generate and decode a single completion."""
        encoded = model.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        # AirLLM backends accept the token ids positionally; the transformers
        # based one forwards them to `transformers.GenerationMixin.generate`.
        raw_output = model.generate(input_ids, attention_mask=attention_mask, **kwargs)
        return AirLLMEngine._decode(model, raw_output)

    @staticmethod
    def _decode(model: AirLLMModelLike, raw_output: object) -> str:
        """Decode raw generation output into text."""
        if isinstance(raw_output, str):
            return raw_output
        decoded = model.tokenizer.batch_decode(raw_output, skip_special_tokens=True)
        if not decoded:
            return ""
        return decoded[0]


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
