"""Tests for the AirLLM engine wrapper.

The AirLLM runtime is never imported: a fake model factory stands in for it so
the suite stays fast and offline.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Mapping
from types import ModuleType

import pytest

from airllm_ollama.config import Settings
from airllm_ollama.engine import (
    AirLLMEngine,
    default_model_factory,
    get_engine,
    normalize_model_name,
    reset_engine,
)
from airllm_ollama.exceptions import ModelInitializationError, ModelMismatchError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TOKEN_IDS = [11, 22, 33]


class FakeTokenizer:
    """Tokenizer double returning deterministic, easily assertable values."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.prompts: list[str] = []
        self.decoded: list[object] = []

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        """Return fake token ids for ``text``."""
        self.prompts.append(text)
        return {"input_ids": list(TOKEN_IDS), "attention_mask": [1, 1, 1]}

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Return a deterministic decoding of ``sequences``."""
        self.decoded.append(sequences)
        return [f"decoded:{sequences}"]


class FakeModel:
    """AirLLM model double recording how it was called."""

    def __init__(self, output: object = None) -> None:
        """Create a model returning ``output`` (token ids by default)."""
        self.tokenizer = FakeTokenizer()
        self.output = list(TOKEN_IDS) if output is None else output
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate(self, *args: object, **kwargs: object) -> object:
        """Record the call and return the canned output."""
        self.calls.append((args, kwargs))
        return self.output


class RecordingFactory:
    """Model factory double counting how often the model was created."""

    def __init__(self, model: FakeModel | None = None) -> None:
        """Create a factory returning ``model``."""
        self.model = model or FakeModel()
        self.calls: list[Settings] = []

    def __call__(self, settings: Settings) -> FakeModel:
        """Record the settings and return the canned model."""
        self.calls.append(settings)
        return self.model


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


def test_normalize_model_name_strips_latest_tag() -> None:
    assert normalize_model_name("my-model:latest") == "my-model"
    assert normalize_model_name("my-model") == "my-model"


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


def test_model_initialisation_failure_is_wrapped() -> None:
    def failing_factory(settings: Settings) -> FakeModel:
        """Fail the way a broken checkpoint would."""
        raise ModelInitializationError("boom")

    engine = AirLLMEngine(Settings(), model_factory=failing_factory)

    with pytest.raises(ModelInitializationError, match="boom"):
        engine.generate("Hello")


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
