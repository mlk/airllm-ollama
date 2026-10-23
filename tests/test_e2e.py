"""End-to-end integration tests.

These tests drive the *whole* application: routing, request validation, the
engine, the tokenizer and the decoding. Only the AirLLM runtime itself is
replaced by a double, so the mocked inference output travels through the same
code path a real model would take.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from airllm_ollama.config import Settings, get_settings
from airllm_ollama.engine import AirLLMEngine
from airllm_ollama.exceptions import ModelInitializationError
from airllm_ollama.main import create_app
from fakes import TOKEN_IDS, FakeFactory, FakeModel

MODEL = "test-model"
PROMPT = "Why is the sky blue?"


def _build_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    factory: FakeFactory,
    *,
    template: str | None = None,
) -> TestClient:
    """Create a client for an application backed by the given fake model."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}', encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", MODEL)
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_PATH", str(checkpoint))
    get_settings.cache_clear()

    factory.model.tokenizer.template = template
    engine = AirLLMEngine(get_settings(), model_factory=factory)
    monkeypatch.setattr("airllm_ollama.api.routes.get_engine", lambda *args, **kwargs: engine)
    return TestClient(create_app())


@pytest.fixture
def factory() -> FakeFactory:
    """Return the fake AirLLM backend used by the service."""
    return FakeFactory()


@pytest.fixture
def service(
    factory: FakeFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a client for the whole application, with a mocked AirLLM backend."""
    with _build_service(monkeypatch, tmp_path, factory) as client:
        yield client


def test_api_tags_lists_the_served_model(service: TestClient) -> None:
    response = service.get("/api/tags")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["name"] == MODEL
    assert model["model"] == MODEL
    assert model["size"] > 0
    assert model["digest"].startswith("sha256:")
    assert model["details"]["family"] == "llama"
    assert model["details"]["format"] == "safetensors"


def test_api_generate_completes_the_prompt(service: TestClient, factory: FakeFactory) -> None:
    response = service.post("/api/generate", json={"model": MODEL, "prompt": PROMPT})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == MODEL
    assert payload["response"] == f"decoded:{TOKEN_IDS}"
    assert payload["done"] is True
    assert payload["done_reason"] == "stop"
    assert payload["prompt_eval_count"] == len(TOKEN_IDS)
    assert payload["eval_count"] == 0
    assert factory.model.tokenizer.prompts == [PROMPT]


def test_api_generate_forwards_the_options(service: TestClient, factory: FakeFactory) -> None:
    service.post(
        "/api/generate",
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "options": {"temperature": 0.4, "num_predict": 16},
        },
    )

    _, kwargs = factory.model.calls[0]
    assert kwargs["temperature"] == 0.4
    assert kwargs["max_new_tokens"] == 16
    assert kwargs["do_sample"] is True


def test_api_generate_prefixes_the_system_prompt(service: TestClient, factory: FakeFactory) -> None:
    service.post(
        "/api/generate",
        json={"model": MODEL, "prompt": PROMPT, "system": "Be terse."},
    )

    assert factory.model.tokenizer.prompts == [f"Be terse.\n\n{PROMPT}"]


def test_api_generate_cuts_at_a_stop_sequence(service: TestClient, factory: FakeFactory) -> None:
    factory.model.output = "before STOP after"

    response = service.post(
        "/api/generate",
        json={"model": MODEL, "prompt": PROMPT, "options": {"stop": ["STOP"]}},
    )

    assert response.json()["response"] == "before "


def test_api_chat_renders_the_history(service: TestClient, factory: FakeFactory) -> None:
    factory.model.tokenizer.template = "<|im_start|>assistant\n"

    response = service.post(
        "/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": PROMPT},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == {"role": "assistant", "content": f"decoded:{TOKEN_IDS}"}
    assert payload["done"] is True
    assert factory.model.tokenizer.prompts == ["<|im_start|>assistant\n"]


def test_api_chat_falls_back_to_the_transcript(service: TestClient, factory: FakeFactory) -> None:
    response = service.post(
        "/api/chat",
        json={"model": MODEL, "messages": [{"role": "user", "content": PROMPT}]},
    )

    assert response.status_code == 200
    assert factory.model.tokenizer.prompts == [f"User: {PROMPT}\n\nAssistant:"]


def test_health_reflects_the_loaded_model(service: TestClient) -> None:
    before = service.get("/health")
    assert before.status_code == 200
    assert before.json()["model_loaded"] is False

    service.post("/api/generate", json={"model": MODEL, "prompt": PROMPT})

    after = service.get("/health")
    assert after.json()["model_loaded"] is True
    assert after.json()["model"] == MODEL


def test_the_root_endpoint_announces_the_service(service: TestClient) -> None:
    response = service.get("/")

    assert response.status_code == 200
    assert response.text.startswith("airllm-ollama is ")


def test_the_version_endpoint_reports_the_version(service: TestClient) -> None:
    response = service.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": get_settings().version}


def test_an_unknown_model_is_answered_with_404(service: TestClient) -> None:
    response = service.post("/api/generate", json={"model": "other-model", "prompt": PROMPT})

    assert response.status_code == 404
    assert "error" in response.json()


def test_an_invalid_request_is_answered_with_400(service: TestClient) -> None:
    response = service.post("/api/generate", json={"prompt": PROMPT})

    assert response.status_code == 400
    assert "error" in response.json()


def test_a_model_that_cannot_be_loaded_is_answered_with_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing_factory(settings: Settings) -> FakeModel:
        """Fail like a broken checkpoint does."""
        raise ModelInitializationError("cannot load the model")

    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", MODEL)
    get_settings.cache_clear()
    engine = AirLLMEngine(get_settings(), model_factory=failing_factory)
    monkeypatch.setattr("airllm_ollama.api.routes.get_engine", lambda *args, **kwargs: engine)

    with TestClient(create_app()) as client:
        response = client.post("/api/generate", json={"model": MODEL, "prompt": PROMPT})

    assert response.status_code == 503
    assert response.json() == {"error": "cannot load the model"}


def test_the_full_journey_of_an_ollama_client(service: TestClient, factory: FakeFactory) -> None:
    """A client discovers the model, generates with it and chats with it."""
    tags = service.get("/api/tags")
    assert tags.status_code == 200
    model_name = tags.json()["models"][0]["name"]

    generated = service.post("/api/generate", json={"model": model_name, "prompt": PROMPT})
    assert generated.status_code == 200
    assert generated.json()["response"]

    answered = service.post(
        "/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": PROMPT}],
        },
    )
    assert answered.status_code == 200
    assert answered.json()["message"]["content"]

    health = service.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    assert len(factory.model.calls) == 2
