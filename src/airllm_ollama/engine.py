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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from airllm_ollama.chat import ChatTurn, RenderedPrompt, render_chat, render_completion
from airllm_ollama.config import Settings, get_settings
from airllm_ollama.exceptions import ModelInitializationError, ModelMismatchError
from airllm_ollama.streaming import TextStreamer


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


@dataclass(frozen=True)
class StreamChunk:
    """A piece of a streaming generation.

    The last chunk of a stream carries the statistics of the finished
    generation in :attr:`result`.
    """

    text: str
    result: GenerationResult | None = None


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


def _import_error(exc: ImportError) -> str:
    """Return a helpful message for a failed import of the AirLLM runtime."""
    if exc.name == "airllm":
        return (
            "The 'airllm' package is not installed. Install it with "
            "`uv sync --extra airllm` to enable local inference."
        )
    missing = exc.name or "a dependency"
    return (
        f"The 'airllm' package is installed, but importing it failed: {missing} is missing. "
        "On macOS AirLLM also needs Apple's MLX framework, which the 'airllm' extra installs."
    )


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
        raise ModelInitializationError(_import_error(exc)) from exc

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

    def prepare(self, model: str | None = None) -> None:
        """Validate the model and load it, so that failures happen early.

        A streaming response sends its headers before the first chunk is
        produced; loading the model here turns a broken checkpoint into a
        proper error response instead of a truncated stream.

        Args:
            model: The model the client asked for.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        self.ensure_model_allowed(model)
        self.load()

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        **kwargs: object,
    ) -> str:
        """Generate text for ``prompt``.

        Args:
            prompt: The prompt to complete.
            system: An optional system message. It is rendered with the chat
                template of the model, the same way ``/api/chat`` renders it.
            model: The model the client asked for. It is validated against the
                served model and never forwarded to AirLLM.
            **kwargs: Generation options forwarded to AirLLM.

        Returns:
            The generated text.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        return self._generate(prompt, system, model=model, **kwargs).text

    def generate_result(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        **kwargs: object,
    ) -> GenerationResult:
        """Generate text for ``prompt`` and report the generation statistics.

        Args:
            prompt: The prompt to complete.
            system: An optional system message. It is rendered with the chat
                template of the model, the same way ``/api/chat`` renders it.
            model: The model the client asked for. It is validated against the
                served model and never forwarded to AirLLM.
            **kwargs: Generation options forwarded to AirLLM.

        Returns:
            The generated text together with token counts and timings.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        return self._generate(prompt, system, model=model, **kwargs)

    def _generate(
        self,
        prompt: str,
        system: str | None,
        /,
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> GenerationResult:
        """Validate the model, load it, render the prompt and generate."""
        self.ensure_model_allowed(model)
        self.cancel_scheduled_unload()
        with self._generation_lock:
            started = time.perf_counter_ns()
            loaded_model = self.load()
            rendered = render_completion(loaded_model.tokenizer, prompt, system)
            return self._run_prompt(loaded_model, rendered, started, **kwargs)

    def generate_chat(
        self,
        messages: Sequence[ChatTurn],
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> GenerationResult:
        """Render a chat history and generate the assistant's answer.

        The history is rendered with the chat template of the model; models
        without a template get a plain transcript.

        Args:
            messages: The chat history as ``(role, content)`` pairs.
            model: The model the client asked for. It is validated against the
                served model and never forwarded to AirLLM.
            **kwargs: Generation options forwarded to AirLLM.

        Returns:
            The generated text together with token counts and timings.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        self.ensure_model_allowed(model)
        self.cancel_scheduled_unload()
        with self._generation_lock:
            started = time.perf_counter_ns()
            loaded_model = self.load()
            rendered = render_chat(loaded_model.tokenizer, messages)
            return self._run_prompt(loaded_model, rendered, started, **kwargs)

    def _run_prompt(
        self,
        model: AirLLMModelLike,
        prompt: RenderedPrompt,
        started: int,
        **kwargs: object,
    ) -> GenerationResult:
        """Run a single generation and collect its statistics."""
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
            load_duration_ns=self._load_duration_ns,
            prompt_duration_ns=prompt_duration,
            eval_duration_ns=eval_duration,
            total_duration_ns=time.perf_counter_ns() - started,
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        **kwargs: object,
    ) -> Iterator[StreamChunk]:
        """Generate text, yielding the chunks as they are produced.

        The generation runs in a worker thread, because the model only reports
        the produced tokens from the thread it runs in. The last chunk carries
        the statistics of the finished generation.

        Args:
            prompt: The prompt to complete.
            system: An optional system message. It is rendered with the chat
                template of the model, the same way ``/api/chat`` renders it.
            model: The model the client asked for. It is validated against the
                served model and never forwarded to AirLLM.
            **kwargs: Generation options forwarded to AirLLM.

        Yields:
            The chunks of the generated text.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        self.ensure_model_allowed(model)
        self.cancel_scheduled_unload()
        with self._generation_lock:
            started = time.perf_counter_ns()
            loaded_model = self.load()
            rendered = render_completion(loaded_model.tokenizer, prompt, system)
            yield from self._stream(loaded_model, rendered, started, **kwargs)

    def generate_chat_stream(
        self,
        messages: Sequence[ChatTurn],
        *,
        model: str | None = None,
        **kwargs: object,
    ) -> Iterator[StreamChunk]:
        """Render a chat history and stream the assistant's answer.

        Args:
            messages: The chat history as ``(role, content)`` pairs.
            model: The model the client asked for. It is validated against the
                served model and never forwarded to AirLLM.
            **kwargs: Generation options forwarded to AirLLM.

        Yields:
            The chunks of the generated text.

        Raises:
            ModelMismatchError: If a mismatching model was requested.
            ModelInitializationError: If the model could not be loaded.
        """
        self.ensure_model_allowed(model)
        self.cancel_scheduled_unload()
        with self._generation_lock:
            started = time.perf_counter_ns()
            loaded_model = self.load()
            rendered = render_chat(loaded_model.tokenizer, messages)
            yield from self._stream(loaded_model, rendered, started, **kwargs)

    def _stream(
        self,
        model: AirLLMModelLike,
        prompt: RenderedPrompt,
        started: int,
        **kwargs: object,
    ) -> Iterator[StreamChunk]:
        """Run a generation and yield its chunks."""
        prompt_started = time.perf_counter_ns()
        encoded = _encode(model.tokenizer, prompt)
        prompt_duration = time.perf_counter_ns() - prompt_started
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        prompt_tokens = _sequence_length(input_ids)
        prepared_ids = _prepare_input_ids(model, input_ids)
        options = self._backend_options(model, kwargs)
        streamer = TextStreamer(model.tokenizer)
        outcome: list[object] = []

        def run() -> None:
            """Run the generation in the worker thread."""
            try:
                outcome.append(
                    model.generate(
                        prepared_ids,
                        attention_mask=attention_mask,
                        streamer=streamer,
                        **options,
                    )
                )
            finally:
                # Backends that do not stream never end the streamer.
                streamer.end()

        worker = threading.Thread(target=run, name="airllm-generation", daemon=True)
        worker.start()
        streamed = False
        generation_started = time.perf_counter_ns()
        try:
            for text in streamer:
                # The streamer never yields empty chunks.
                streamed = True
                yield StreamChunk(text=text)
        finally:
            # The model may not be used while it is still generating, so the
            # lock is only released after the worker finished.
            worker.join()

        raw_output = outcome[0] if outcome else None
        text = self._decode(model, raw_output) if raw_output is not None else ""
        if not streamed and text:
            # The backend does not support streaming: deliver the whole answer.
            yield StreamChunk(text=text)
        eval_duration = time.perf_counter_ns() - generation_started
        yield StreamChunk(
            text="",
            result=GenerationResult(
                text=text,
                prompt_tokens=prompt_tokens,
                generated_tokens=self._count_generated_tokens(model, raw_output, prompt_tokens),
                load_duration_ns=self._load_duration_ns,
                prompt_duration_ns=prompt_duration,
                eval_duration_ns=eval_duration,
                total_duration_ns=time.perf_counter_ns() - started,
            ),
        )

    def _run(
        self, model: AirLLMModelLike, prompt: RenderedPrompt, **kwargs: object
    ) -> tuple[object, int]:
        """Tokenize, generate and return the raw output plus the prompt length."""
        encoded = _encode(model.tokenizer, prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        prompt_tokens = _sequence_length(input_ids)
        options = self._backend_options(model, kwargs)
        # AirLLM backends accept the token ids positionally; the transformers
        # based one forwards them to `transformers.GenerationMixin.generate`.
        raw_output = model.generate(
            _prepare_input_ids(model, input_ids), attention_mask=attention_mask, **options
        )
        return raw_output, prompt_tokens

    def _backend_options(
        self, model: AirLLMModelLike, kwargs: dict[str, object]
    ) -> dict[str, object]:
        """Return the generation options the backend understands."""
        if not is_mlx_backend(model) or "max_new_tokens" in kwargs:
            return kwargs
        # The MLX backend stops after `max_new_tokens` tokens and has no
        # default for it.
        return {**kwargs, "max_new_tokens": self._settings.max_seq_len}

    @staticmethod
    def _count_generated_tokens(
        model: AirLLMModelLike, raw_output: object, prompt_tokens: int
    ) -> int:
        """Return the number of tokens the model generated."""
        if isinstance(raw_output, str):
            # Backends that decode themselves only return the new tokens. The
            # text is tokenized without the special tokens, otherwise the
            # tokenizer would add a BOS token the model never generated and
            # the count would be one too high.
            return _sequence_length(
                model.tokenizer(raw_output, add_special_tokens=False)["input_ids"]
            )
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


def _encode(tokenizer: TokenizerLike, prompt: RenderedPrompt) -> Mapping[str, object]:
    """Tokenize a rendered prompt.

    A prompt that was rendered with the chat template of the model carries the
    special tokens of the model already, so the tokenizer must not add them a
    second time — the model would see two beginning-of-sequence tokens.

    Args:
        tokenizer: The tokenizer of the model.
        prompt: The rendered prompt.

    Returns:
        The tokenized prompt.
    """
    return tokenizer(
        prompt.text,
        return_tensors="pt",
        add_special_tokens=prompt.add_special_tokens,
    )


def is_mlx_backend(model: object) -> bool:
    """Return whether the model is AirLLM's MLX backend (used on macOS).

    The MLX backend works on Apple's MLX arrays instead of the torch tensors
    the transformers based backend expects. It is recognised by its name, so
    that another backend is not mistaken for it.
    """
    kind = type(model)
    return "mlx" in kind.__name__.lower() or "mlx" in kind.__module__.lower()


def _prepare_input_ids(model: object, input_ids: object) -> object:
    """Return the token ids in the format the backend expects."""
    if not is_mlx_backend(model):
        return input_ids
    import mlx.core as mx

    return mx.array(_as_nested_list(input_ids))


def _as_nested_list(value: object) -> list[list[int]]:
    """Return the token ids of a tensor or sequence as a nested list.

    Values that carry no token ids (e.g. ``None``) yield an empty list.
    """
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        converted = to_list()
    elif isinstance(value, (list, tuple)):
        converted = list(value)
    else:
        converted = []
    if converted and not isinstance(converted[0], list):
        converted = [converted]
    return [[int(item) for item in row] for row in converted]


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
