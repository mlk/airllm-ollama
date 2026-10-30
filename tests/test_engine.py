"""Tests for the AirLLM engine wrapper.

The AirLLM runtime is never imported: a fake model factory stands in for it so
the suite stays fast and offline.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from pytest_mock import MockerFixture

from airllm_ollama.chat import render_transcript
from airllm_ollama.config import Settings
from airllm_ollama.engine import (
    AirLLMEngine,
    _as_nested_list,
    _import_error,
    _sequence_length,
    default_model_factory,
    get_engine,
    is_airllm_installed,
    is_mlx_backend,
    model_names_match,
    normalize_model_name,
    reset_engine,
)
from airllm_ollama.exceptions import (
    GenerationError,
    ModelInitializationError,
    ModelMismatchError,
)
from airllm_ollama.stopping import StopGeneration
from fakes import (
    TOKEN_IDS,
    AbortableModel,
    BosTokenizer,
    BrokenModel,
    FakeModel,
    FakeOutput,
    FakeTokenizer,
    RecordingFactory,
    SystemRoleTokenizer,
)

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
BOS = BosTokenizer.BOS
BOS_TEXT = BosTokenizer.BOS_TEXT


class FakeMlxModel(FakeModel):
    """Model double of AirLLM's MLX backend.

    AirLLM recognises the backend by the name of the class, so the double has
    to carry it.
    """


class _FakeAirLLMModule(ModuleType):
    """Stub module exposing the ``AutoModel`` entry point."""

    AutoModel: type[object]


def _install_fake_airllm(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, dict[str, object]]],
    model: FakeModel | None,
    load: Callable[[str], FakeModel] | None = None,
) -> None:
    """Install a stub ``airllm`` module so the default factory can be exercised."""

    def from_pretrained(path: str, **kwargs: object) -> FakeModel:
        """Record the load request and return the canned model."""
        calls.append((path, dict(kwargs)))
        if load is not None:
            return load(path)
        assert model is not None
        return model

    class AutoModel:
        """Stub of the AirLLM entry point."""

        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            """Delegate to the stub loader."""
            return from_pretrained(path, **kwargs)

    module = _FakeAirLLMModule("airllm")
    module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "airllm", module)


def _wait_until_unloaded(engine: AirLLMEngine, timeout: float = 2.0) -> bool:
    """Wait until the engine released the model, without a fixed sleep."""
    deadline = time.monotonic() + timeout
    while engine.is_loaded and time.monotonic() < deadline:
        time.sleep(0.01)
    return not engine.is_loaded


@pytest.fixture(autouse=True)
def _reset_engine_singleton() -> None:
    """Keep the process-wide engine from leaking between tests."""
    reset_engine()


def test_engine_exposes_configured_model() -> None:
    engine = AirLLMEngine(Settings(model_name="my-model", model_path="/models/my-model"))

    assert engine.model_name == "my-model"
    assert engine.model_path == "/models/my-model"
    assert engine.is_loaded is False


def test_model_path_falls_back_to_model_name() -> None:
    engine = AirLLMEngine(Settings(model_name=TINY_MODEL))

    assert engine.model_path == TINY_MODEL


def test_generate_returns_generated_text() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    result = engine.generate("Hello there")

    assert result == f"decoded:{TOKEN_IDS}"
    assert factory.model.tokenizer.prompts == ["Hello there"]


def test_generate_forwards_generation_options() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    engine.generate("Hello", max_new_tokens=32, temperature=0.7)

    _, kwargs = factory.model.calls[0]
    assert kwargs["max_new_tokens"] == 32
    assert kwargs["temperature"] == 0.7
    assert kwargs["attention_mask"] == [1, 1, 1]


def test_the_model_keyword_is_never_forwarded_to_the_model() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=factory)

    engine.generate("Hello", model="my-model", max_new_tokens=8)

    _, kwargs = factory.model.calls[0]
    assert "model" not in kwargs


def test_model_is_loaded_lazily_and_only_once() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    assert factory.calls == []
    assert engine.is_loaded is False

    engine.generate("first")
    engine.generate("second")

    assert len(factory.calls) == 1
    assert engine.is_loaded is True


def test_unload_forces_a_reload() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    engine.generate("first")
    engine.unload()
    assert engine.is_loaded is False
    engine.generate("second")

    assert len(factory.calls) == 2


def test_generate_accepts_the_configured_model_name() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=factory)

    result = engine.generate("Hello", model="my-model")

    assert result == f"decoded:{TOKEN_IDS}"


def test_generate_accepts_the_latest_tag_variant() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=factory)

    result = engine.generate("Hello", model="my-model:latest")

    assert result == f"decoded:{TOKEN_IDS}"


def test_generate_raises_on_model_mismatch() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=factory)

    with pytest.raises(ModelMismatchError) as excinfo:
        engine.generate("Hello", model="other-model")

    message = str(excinfo.value)
    assert "other-model" in message
    assert "my-model" in message
    assert factory.calls == []


def test_ensure_model_allowed_accepts_none() -> None:
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=RecordingFactory())

    engine.ensure_model_allowed(None)


def test_normalize_model_name_strips_the_tag() -> None:
    assert normalize_model_name("my-model:latest") == "my-model"
    assert normalize_model_name("my-model") == "my-model"
    assert normalize_model_name("my-model:q4_0") == "my-model"
    assert normalize_model_name("llama3.2:1b") == "llama3.2"
    assert normalize_model_name("ns/my-model:v1.0") == "ns/my-model"


@pytest.mark.parametrize(
    "requested",
    ["my-model", "my-model:latest", "my-model:q4_0", "my-model:v1.0"],
)
def test_any_tag_resolves_to_the_served_model(requested: str) -> None:
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=RecordingFactory())

    engine.ensure_model_allowed(requested)


@pytest.mark.parametrize("requested", ["other-model", "my-model-2", "ns/my-model"])
def test_unknown_models_are_rejected(requested: str) -> None:
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=RecordingFactory())

    with pytest.raises(ModelMismatchError):
        engine.ensure_model_allowed(requested)


def test_served_tag_must_match_exactly() -> None:
    engine = AirLLMEngine(Settings(model_name="llama3.2:1b"), model_factory=RecordingFactory())

    engine.ensure_model_allowed("llama3.2:1b")
    engine.ensure_model_allowed("llama3.2")

    with pytest.raises(ModelMismatchError):
        engine.ensure_model_allowed("llama3.2:70b")

    with pytest.raises(ModelMismatchError):
        engine.ensure_model_allowed("llama3.1:1b")


def test_model_names_match() -> None:
    assert model_names_match("my-model", "my-model")
    assert model_names_match("my-model:q4_0", "my-model")
    assert not model_names_match("other-model", "my-model")
    assert model_names_match("llama3.2", "llama3.2:1b")
    assert not model_names_match("llama3.2:70b", "llama3.2:1b")


def test_backend_returning_text_is_used_as_is() -> None:
    factory = RecordingFactory(FakeModel(output="raw text from the backend"))
    engine = AirLLMEngine(Settings(), model_factory=factory)

    assert engine.generate("Hello") == "raw text from the backend"


def test_generation_is_serialised_across_threads() -> None:
    active = 0
    max_active = 0
    guard = threading.Lock()

    class SlowModel(FakeModel):
        """Model double that overlaps calls when it is not serialised."""

        def generate(self, *args: object, **kwargs: object) -> object:
            """Track concurrent invocations."""
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return self.output

    factory = RecordingFactory(SlowModel())
    engine = AirLLMEngine(Settings(), model_factory=factory)

    threads = [
        threading.Thread(target=engine.generate, args=(f"prompt-{index}",)) for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1
    assert len(factory.calls) == 1


def test_load_does_not_block_while_a_generation_runs() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(FakeModel):
        """Model double that blocks until it is released."""

        def generate(self, *args: object, **kwargs: object) -> object:
            """Wait for the test to release the call."""
            started.set()
            release.wait(timeout=5)
            return self.output

    factory = RecordingFactory(BlockingModel())
    engine = AirLLMEngine(Settings(), model_factory=factory)

    worker = threading.Thread(target=engine.generate, args=("prompt",))
    worker.start()
    assert started.wait(timeout=5)

    loaded = engine.load()
    release.set()
    worker.join(timeout=5)

    assert loaded is factory.model
    assert worker.is_alive() is False


def test_unload_waits_for_a_running_generation() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingModel(FakeModel):
        """Model double that blocks until it is released."""

        def generate(self, *args: object, **kwargs: object) -> object:
            """Wait for the test to release the call."""
            started.set()
            release.wait(timeout=5)
            return self.output

    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(BlockingModel()))

    worker = threading.Thread(target=engine.generate, args=("prompt",))
    worker.start()
    assert started.wait(timeout=5)

    unloaded = threading.Event()

    def unload() -> None:
        """Unload while a generation is in flight."""
        engine.unload()
        unloaded.set()

    unloader = threading.Thread(target=unload)
    unloader.start()
    unloader.join(timeout=0.2)
    assert unloaded.is_set() is False

    release.set()
    worker.join(timeout=5)
    unloader.join(timeout=5)

    assert unloaded.is_set() is True
    assert engine.is_loaded is False


def test_schedule_unload_releases_the_model_after_the_delay() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")
    assert engine.is_loaded is True

    engine.schedule_unload(0.05)

    assert _wait_until_unloaded(engine) is True
    assert engine.is_loaded is False


def test_schedule_unload_with_zero_releases_the_model_at_once() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")

    engine.schedule_unload(0)

    assert engine.is_loaded is False


def test_a_new_request_cancels_a_pending_unload() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")
    engine.schedule_unload(0.2)

    engine.generate("hi again")
    time.sleep(0.4)

    assert engine.is_loaded is True


def test_cancel_scheduled_unload_keeps_the_model_loaded() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")
    engine.schedule_unload(0.1)

    engine.cancel_scheduled_unload()
    time.sleep(0.3)

    assert engine.is_loaded is True


def test_a_model_without_a_scheduled_unload_has_no_unload_moment() -> None:
    """Without a scheduled unload there is no moment to report."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")

    assert engine.unload_at is None


def test_the_unload_moment_is_reported() -> None:
    """The engine knows when it will unload the model."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")
    engine.schedule_unload(60)

    unload_at = engine.unload_at
    assert unload_at is not None
    assert unload_at > datetime.now(UTC)


def test_the_unload_moment_is_forgotten_after_the_unload() -> None:
    """A model that was unloaded has no unload moment any more."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)
    engine.generate("hi")
    engine.schedule_unload(0.05)

    assert _wait_until_unloaded(engine) is True
    assert engine.unload_at is None


def test_generate_chat_renders_the_history_and_generates() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    result = engine.generate_chat([("user", "Hello")])

    assert factory.model.tokenizer.prompts == ["User: Hello\n\nAssistant:"]
    assert result.text == f"decoded:{TOKEN_IDS}"
    assert result.total_duration_ns > 0


def test_generate_chat_uses_the_chat_template() -> None:
    class TemplateTokenizer(FakeTokenizer):
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            """Render the messages the way a chat template would."""
            return f"templated:{messages}"

    model = FakeModel()
    model.tokenizer = TemplateTokenizer()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate_chat([("user", "Hello")])

    assert model.tokenizer.prompts[0].startswith("templated:")


def test_generate_chat_rejects_a_mismatching_model() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=factory)

    with pytest.raises(ModelMismatchError):
        engine.generate_chat([("user", "Hello")], model="other-model")

    assert factory.calls == []


class MlxModel(FakeModel):
    """Model double that mimics the MLX backend of AirLLM (macOS)."""

    def __init__(self) -> None:
        """Create a model that looks like the MLX backend."""
        super().__init__()
        self.model_args = object()


def test_the_mlx_backend_is_detected() -> None:
    assert is_mlx_backend(MlxModel()) is True
    assert is_mlx_backend(FakeModel()) is False


def test_a_model_with_a_model_args_attribute_is_not_mlx() -> None:
    """The detection must not rely on an attribute of a single backend."""

    class OtherModel(FakeModel):
        """Model double of a backend that has its own arguments."""

        def __init__(self) -> None:
            """Create a model with a `model_args` attribute."""
            super().__init__()
            self.model_args = object()

    assert is_mlx_backend(OtherModel()) is False


def test_token_ids_are_converted_to_nested_lists() -> None:
    assert _as_nested_list([1, 2, 3]) == [[1, 2, 3]]
    assert _as_nested_list([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    assert _as_nested_list(None) == []

    class FakeTensor:
        """Stand-in for a torch tensor."""

        def tolist(self) -> list[list[int]]:
            """Return the values as a nested list."""
            return [[1, 2, 3]]

    assert _as_nested_list(FakeTensor()) == [[1, 2, 3]]


def test_mlx_models_receive_an_mlx_array() -> None:
    pytest.importorskip("mlx.core")
    model = MlxModel()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("Hello")

    input_ids = model.calls[0][0][0]
    assert type(input_ids).__name__ == "array"
    assert list(getattr(input_ids, "shape", ())) == [1, len(TOKEN_IDS)]


def test_the_mlx_backend_gets_a_max_new_tokens() -> None:
    pytest.importorskip("mlx.core")
    model = MlxModel()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("Hello")

    assert model.calls[0][1]["max_new_tokens"] == 512


def test_the_transformers_backend_keeps_its_options() -> None:
    model = FakeModel()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("Hello")

    assert "max_new_tokens" not in model.calls[0][1]


def test_the_system_message_is_rendered_with_the_chat_template() -> None:
    model = FakeModel()
    model.tokenizer.template = "<|im_start|>assistant\n"
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello", system="Be terse.")

    assert model.tokenizer.prompts == ["<|im_start|>assistant\n"]


def test_the_prompt_is_rendered_with_the_chat_template_without_a_system_message() -> None:
    """Ollama applies the template to every prompt, not only to the ones with a system."""
    model = FakeModel()
    model.tokenizer.template = "<|im_start|>assistant\n"
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello")

    assert model.tokenizer.prompts == ["<|im_start|>assistant\n"]
    assert model.tokenizer.renderings == [[{"role": "user", "content": "hello"}]]


def test_a_raw_prompt_is_sent_unchanged() -> None:
    """`raw` asks for the prompt without the template."""
    model = FakeModel()
    model.tokenizer.template = "<|im_start|>assistant\n"
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello", system="Be terse.", raw=True)

    assert model.tokenizer.prompts == ["hello"]
    assert model.tokenizer.renderings == []


def test_a_template_that_refuses_the_system_role_keeps_the_model_format() -> None:
    """The system message is merged into the first user turn."""
    model = FakeModel()
    model.tokenizer = SystemRoleTokenizer()
    model.tokenizer.template = "<|im_start|>assistant\n"
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate_chat([("system", "Be terse."), ("user", "Hello")])

    assert model.tokenizer.renderings == [[{"role": "user", "content": "Be terse.\n\nHello"}]]
    assert model.tokenizer.prompts == ["<|im_start|>assistant\n"]


def test_without_a_system_message_and_a_template_the_prompt_is_unchanged() -> None:
    model = FakeModel()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello")

    assert model.tokenizer.prompts == ["hello"]


def test_without_a_template_the_system_message_is_joined_with_the_prompt() -> None:
    model = FakeModel()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello", system="Be terse.")

    assert model.tokenizer.prompts == ["Be terse.\n\nhello"]


def test_a_templated_prompt_is_tokenized_without_the_special_tokens() -> None:
    """The template renders the special tokens, the tokenizer must not add them."""
    model = FakeModel()
    model.tokenizer = BosTokenizer()
    model.tokenizer.template = f"{BOS_TEXT}<|im_start|>assistant\n"
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello", system="Be terse.")

    assert model.tokenizer.tokenizations[0]["add_special_tokens"] is False
    assert model.calls[0][0][0] == [BOS, *TOKEN_IDS]


def test_a_prompt_without_a_system_message_is_tokenized_with_the_special_tokens() -> None:
    """A prompt that no template rendered still needs the special tokens."""
    model = FakeModel()
    model.tokenizer = BosTokenizer()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate("hello")

    assert model.tokenizer.tokenizations[0]["add_special_tokens"] is True
    assert model.calls[0][0][0] == [BOS, *TOKEN_IDS]


def test_a_templated_chat_is_tokenized_without_the_special_tokens() -> None:
    """The chat of a model with a template is tokenized without the special tokens."""
    model = FakeModel()
    model.tokenizer = BosTokenizer()
    model.tokenizer.template = f"{BOS_TEXT}<|im_start|>assistant\n"
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate_chat([("user", "Hello")])

    assert model.tokenizer.tokenizations[0]["add_special_tokens"] is False
    assert model.calls[0][0][0] == [BOS, *TOKEN_IDS]


def test_the_transcript_fallback_is_tokenized_with_the_special_tokens() -> None:
    """The transcript is not rendered by a template, so it keeps its BOS token."""
    model = FakeModel()
    model.tokenizer = BosTokenizer()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    engine.generate_chat([("user", "Hello")])

    assert model.tokenizer.prompts == [render_transcript([("user", "Hello")])]
    assert model.tokenizer.tokenizations[0]["add_special_tokens"] is True
    assert model.calls[0][0][0] == [BOS, *TOKEN_IDS]


def test_a_templated_stream_is_tokenized_without_the_special_tokens() -> None:
    """The streaming path tokenizes the prompt the same way."""
    model = FakeModel()
    model.tokenizer = BosTokenizer()
    model.tokenizer.template = f"{BOS_TEXT}<|im_start|>assistant\n"
    model.chunks = ["blue ", "sky"]
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    list(engine.generate_chat_stream([("user", "Hello")]))

    assert model.tokenizer.tokenizations[0]["add_special_tokens"] is False
    assert model.calls[0][0][0] == [BOS, *TOKEN_IDS]


def test_a_raw_stream_sends_the_prompt_unchanged() -> None:
    """`raw` asks for the prompt without the template, in the streaming path too."""
    model = FakeModel()
    model.tokenizer.template = "<|im_start|>assistant\n"
    model.chunks = ["blue ", "sky"]
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    list(engine.generate_stream("hello", system="Be terse.", raw=True))

    assert model.tokenizer.prompts == ["hello"]
    assert model.tokenizer.renderings == []


def test_a_mismatching_model_is_rejected_before_the_model_is_loaded() -> None:
    """A wrong model must not load the checkpoint."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=factory)

    with pytest.raises(ModelMismatchError):
        engine.generate("hello", system="Be terse.", model="other-model")

    assert factory.calls == []


def test_generation_result_reports_the_statistics() -> None:
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    result = engine.generate_result("hi")

    assert result.text == f"decoded:{TOKEN_IDS}"
    assert result.prompt_tokens == len(TOKEN_IDS)
    assert result.generated_tokens == 0
    assert result.total_duration_ns > 0
    assert result.load_duration_ns > 0


def test_generated_tokens_are_counted_for_token_backends() -> None:
    factory = RecordingFactory(FakeModel(output=[[11, 22, 33, 44, 55]]))
    engine = AirLLMEngine(Settings(), model_factory=factory)

    result = engine.generate_result("hi")

    assert result.generated_tokens == 2


def test_generated_tokens_are_counted_for_text_backends() -> None:
    factory = RecordingFactory(FakeModel(output="some text"))
    engine = AirLLMEngine(Settings(), model_factory=factory)

    result = engine.generate_result("hi")

    assert result.generated_tokens == len(TOKEN_IDS)


def test_the_text_of_a_backend_is_counted_without_the_special_tokens() -> None:
    """A BOS token that the model never generated must not be counted."""
    model = FakeModel(output="some text")
    model.tokenizer = BosTokenizer()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    result = engine.generate_result("hi")

    # The double counts four tokens when the tokenizer adds a BOS token and
    # three when it does not — the text of the backend has three.
    assert result.generated_tokens == len(TOKEN_IDS)
    assert model.tokenizer.tokenizations[-1]["add_special_tokens"] is False


def test_sequence_length_handles_tensors_and_sequences() -> None:
    class FakeTensor:
        """Stand-in for a torch tensor."""

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Store the shape of the fake tensor."""
            self.shape = shape

    assert _sequence_length(FakeTensor((1, 7))) == 7
    assert _sequence_length([1, 2, 3]) == 3
    assert _sequence_length([[1, 2], [3, 4]]) == 2
    assert _sequence_length(None) == 0


def test_model_initialisation_failure_is_wrapped() -> None:
    def failing_factory(settings: Settings) -> FakeModel:
        """Fail the way a broken checkpoint would."""
        raise ModelInitializationError("boom")

    engine = AirLLMEngine(Settings(), model_factory=failing_factory)

    with pytest.raises(ModelInitializationError, match="boom"):
        engine.generate("Hello")


def test_a_missing_dependency_is_reported() -> None:
    """A broken import is not the same as a missing package."""
    error = ImportError("No module named 'mlx'")
    error.name = "mlx"

    message = _import_error(error)

    assert "mlx" in message
    assert "installed" in message


def test_default_factory_reports_missing_airllm_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "airllm", None)

    with pytest.raises(ModelInitializationError, match="uv sync --extra airllm"):
        default_model_factory(Settings())


def test_default_factory_loads_the_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    model = FakeModel()
    _install_fake_airllm(monkeypatch, calls, model)

    loaded = default_model_factory(Settings(model_name="my-model", device="cpu"))

    assert loaded is model
    assert calls == [("my-model", {"device": "cpu", "max_seq_len": 512})]


def test_default_factory_forwards_the_shard_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    _install_fake_airllm(monkeypatch, calls, FakeModel())

    default_model_factory(Settings(layer_shards_saving_path="/tmp/shards"))

    assert calls[0][1]["layer_shards_saving_path"] == "/tmp/shards"


def test_default_factory_wraps_load_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_load(path: str, **kwargs: object) -> FakeModel:
        """Fail like a missing checkpoint does."""
        raise OSError("no such file")

    _install_fake_airllm(monkeypatch, [], None, load=failing_load)

    with pytest.raises(ModelInitializationError, match="no such file"):
        default_model_factory(Settings(model_name="my-model"))


def test_default_factory_refuses_a_tied_checkpoint_on_the_mlx_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    """The MLX backend cannot load a model that ties its embeddings."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint", tensors=("model.embed_tokens.weight",))
    _install_fake_airllm(monkeypatch, [], FakeMlxModel())

    with pytest.raises(ModelInitializationError, match="ties its embeddings"):
        default_model_factory(Settings(model_path=str(checkpoint)))


def test_default_factory_loads_a_tied_checkpoint_on_the_torch_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    """The torch backend ties the embeddings itself, so it loads the model."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint", tensors=("model.embed_tokens.weight",))
    _install_fake_airllm(monkeypatch, [], FakeModel())

    loaded = default_model_factory(Settings(model_path=str(checkpoint)))

    assert isinstance(loaded, FakeModel)


def test_engine_exposes_its_settings() -> None:
    settings = Settings(model_name="my-model")
    engine = AirLLMEngine(settings, model_factory=RecordingFactory())

    assert engine.settings is settings


def test_empty_decoding_yields_empty_text() -> None:
    class SilentTokenizer(FakeTokenizer):
        """Tokenizer double that decodes nothing."""

        def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
            """Return no decodings at all."""
            return []

    model = FakeModel()
    model.tokenizer = SilentTokenizer()
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory(model))

    assert engine.generate("Hello") == ""


def test_engine_reports_whether_airllm_is_installed() -> None:
    engine = AirLLMEngine(Settings(), model_factory=RecordingFactory())

    assert engine.is_available is is_airllm_installed()


def test_airllm_availability_check_does_not_import_the_package(
    mocker: MockerFixture,
) -> None:
    find_spec = mocker.patch("airllm_ollama.engine.importlib.util.find_spec", return_value=object())

    assert is_airllm_installed() is True
    find_spec.assert_called_once_with("airllm")
    assert "airllm" not in sys.modules


def test_airllm_availability_is_false_when_the_package_is_missing(
    mocker: MockerFixture,
) -> None:
    mocker.patch("airllm_ollama.engine.importlib.util.find_spec", return_value=None)

    assert is_airllm_installed() is False


def test_airllm_availability_survives_a_broken_lookup(mocker: MockerFixture) -> None:
    mocker.patch("airllm_ollama.engine.importlib.util.find_spec", side_effect=ImportError("broken"))

    assert is_airllm_installed() is False


def test_get_engine_returns_a_singleton() -> None:
    first = get_engine(Settings(model_name="my-model"))
    second = get_engine()

    assert first is second
    assert first.model_name == "my-model"


def test_reset_engine_creates_a_new_instance() -> None:
    first = get_engine()
    reset_engine()
    second = get_engine()

    assert first is not second


class FakeTorch(ModuleType):
    """The part of torch the engine uses — it is an optional dependency."""

    def no_grad(self) -> AbstractContextManager[object]:
        """Return a context manager that switches the autograd graph off."""
        return nullcontext()


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend that torch is installed."""
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))


class DecisionTokenizer(FakeTokenizer):
    """Tokenizer double that gives every text its own token."""

    def __init__(self, ids: dict[str, int] | None = None) -> None:
        """Create the tokenizer.

        Args:
            ids: The token id of every text.
        """
        super().__init__()
        self.ids = ids or {}

    def token_ids(self, text: str, **kwargs: object) -> list[int]:
        """Return the token id of the text."""
        return [self.ids.get(text, 1)]


def test_decide_reports_the_probability_of_every_candidate(fake_torch: None) -> None:
    """The probabilities are read from the logits of the last position."""
    factory = RecordingFactory()
    model = factory.model
    model.tokenizer = DecisionTokenizer({"billing": 101, "bug": 102})
    model.output = FakeOutput({101: 0.7, 102: 0.3})
    engine = AirLLMEngine(Settings(), model_factory=factory)

    decision = engine.decide("Context: a ticket", ["billing", "bug"])

    assert decision.probabilities == {"billing": 0.7, "bug": 0.3}
    assert decision.prompt_tokens == 1


def test_decide_asks_the_model_once(fake_torch: None) -> None:
    """The decision is a single forward pass, without the autoregressive loop."""
    factory = RecordingFactory()
    model = factory.model
    model.tokenizer = DecisionTokenizer({"a": 101})
    model.output = FakeOutput({101: 1.0})
    engine = AirLLMEngine(Settings(), model_factory=factory)

    engine.decide("Context: a ticket", ["a"])

    assert len(model.forwards) == 1
    assert model.calls == []


def test_decide_rejects_another_model() -> None:
    """The model of the client is validated like everywhere else."""
    engine = AirLLMEngine(Settings(model_name="my-model"), model_factory=RecordingFactory())

    with pytest.raises(ModelMismatchError):
        engine.decide("Context: a ticket", ["a"], model="other-model")


def test_decide_looks_the_candidate_up_with_a_leading_space(
    fake_torch: None,
) -> None:
    """A tokenizer with a prefix space writes the candidate as " billing"."""
    factory = RecordingFactory()
    model = factory.model
    model.tokenizer = DecisionTokenizer({"billing": 101, " billing": 102})
    model.output = FakeOutput({102: 0.9})
    engine = AirLLMEngine(Settings(), model_factory=factory)

    decision = engine.decide("Context: a ticket", ["billing"])

    assert decision.probabilities == {"billing": 0.9}


def test_a_failed_decision_is_reported() -> None:
    """A model that fails while it decides is reported, not swallowed."""
    factory = RecordingFactory(BrokenModel())
    engine = AirLLMEngine(Settings(model_path="/models/broken"), model_factory=factory)

    with pytest.raises(GenerationError) as failure:
        engine.decide("Context: a ticket", ["a"])

    assert "failed to decide" in str(failure.value)
    assert "/models/broken" in str(failure.value)


def test_the_stop_sequences_end_the_generation() -> None:
    """The criterion that stops at a stop sequence is handed to the model."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    engine.generate("hello", stop=["STOP"])

    _, kwargs = factory.model.calls[0]
    criteria = cast("list[StopGeneration]", kwargs["stopping_criteria"])
    assert len(criteria) == 1
    assert isinstance(criteria[0], StopGeneration)


def test_the_stop_sequences_end_the_stream() -> None:
    """The streaming path hands the criterion to the model as well."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    list(engine.generate_stream("hello", stop=["STOP"]))

    _, kwargs = factory.model.calls[0]
    criteria = cast("list[StopGeneration]", kwargs["stopping_criteria"])
    assert len(criteria) == 1
    assert isinstance(criteria[0], StopGeneration)


def test_without_a_stop_sequence_the_model_gets_no_criterion() -> None:
    """A request without a stop sequence generates until the end."""
    factory = RecordingFactory()
    engine = AirLLMEngine(Settings(), model_factory=factory)

    engine.generate("hello")

    _, kwargs = factory.model.calls[0]
    assert "stopping_criteria" not in kwargs


def test_a_disconnected_client_stops_the_generation() -> None:
    """The generation ends when the stream is closed."""
    factory = RecordingFactory(AbortableModel())
    engine = AirLLMEngine(Settings(), model_factory=factory)

    stream = engine.generate_stream("hello")
    next(stream)
    _, kwargs = factory.model.calls[0]
    criteria = cast("list[StopGeneration]", kwargs["stopping_criteria"])
    criterion = criteria[0]
    assert criterion.abort.is_set() is False

    stream.close()

    assert criterion.abort.is_set() is True


def test_the_caller_stops_the_generation() -> None:
    """The event of the caller stops the generation."""
    factory = RecordingFactory(AbortableModel())
    engine = AirLLMEngine(Settings(), model_factory=factory)
    abort = threading.Event()

    stream = engine.generate_stream("hello", abort=abort)
    next(stream)
    _, kwargs = factory.model.calls[0]
    criteria = cast("list[StopGeneration]", kwargs["stopping_criteria"])

    assert criteria[0].abort is abort


def test_a_failed_generation_is_reported() -> None:
    """A model that fails while it generates is reported, not swallowed."""
    factory = RecordingFactory(BrokenModel())
    engine = AirLLMEngine(Settings(model_path="/models/broken"), model_factory=factory)

    with pytest.raises(GenerationError) as failure:
        engine.generate("hello")

    assert "failed to generate" in str(failure.value)
    # The message names the checkpoint, so that the operator knows the files.
    assert "/models/broken" in str(failure.value)


def test_a_failed_stream_is_reported() -> None:
    """The failure of the worker thread is raised in the reading thread."""
    factory = RecordingFactory(BrokenModel())
    engine = AirLLMEngine(Settings(), model_factory=factory)

    with pytest.raises(GenerationError):
        list(engine.generate_stream("hello"))
