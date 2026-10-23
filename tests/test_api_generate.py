"""Integration tests for the Ollama compatible ``/api/generate`` endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from airllm_ollama.config import get_settings
from airllm_ollama.engine import GenerationResult
from airllm_ollama.exceptions import ModelInitializationError, ModelMismatchError

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def engine(mocker: MockerFixture) -> Any:
    """Patch the engine used by the API layer with a recording double."""
    double = mocker.patch("airllm_ollama.api.routes.get_engine")
    double.return_value.generate_result.return_value = GenerationResult(
        text="the sky is blue",
        prompt_tokens=4,
        generated_tokens=3,
        load_duration_ns=1_000,
        prompt_duration_ns=2_000,
        eval_duration_ns=3_000,
        total_duration_ns=4_000,
    )
    double.return_value.is_loaded = False
    return double


def test_generate_returns_the_ollama_response_shape(client: TestClient, engine: Any) -> None:
    response = client.post("/api/generate", json={"model": MODEL, "prompt": "why is the sky blue?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == MODEL
    assert payload["response"] == "the sky is blue"
    assert payload["done"] is True
    assert payload["done_reason"] == "stop"
    assert payload["eval_count"] == 3
    assert payload["prompt_eval_count"] == 4
    assert payload["total_duration"] == 4_000
    assert payload["load_duration"] == 1_000
    assert payload["eval_duration"] == 3_000
    assert payload["prompt_eval_duration"] == 2_000
    assert payload["created_at"].endswith("Z")


def test_generate_delegates_to_the_engine(client: TestClient, engine: Any) -> None:
    client.post("/api/generate", json={"model": MODEL, "prompt": "hello"})

    engine.return_value.generate_result.assert_called_once_with("hello", model=MODEL)


def test_generate_maps_ollama_options(client: TestClient, engine: Any) -> None:
    client.post(
        "/api/generate",
        json={
            "model": MODEL,
            "prompt": "hello",
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 128,
                "num_ctx": 2048,
                "repeat_penalty": 1.1,
            },
        },
    )

    _, kwargs = engine.return_value.generate_result.call_args
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 40
    assert kwargs["max_new_tokens"] == 128
    assert kwargs["max_length"] == 2048
    assert kwargs["repetition_penalty"] == 1.1
    assert kwargs["do_sample"] is True


def test_generate_without_options_does_not_sample(client: TestClient, engine: Any) -> None:
    client.post("/api/generate", json={"model": MODEL, "prompt": "hello"})

    _, kwargs = engine.return_value.generate_result.call_args
    assert "temperature" not in kwargs
    assert "do_sample" not in kwargs


def test_generate_prefixes_the_system_prompt(client: TestClient, engine: Any) -> None:
    client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "hello", "system": "You are helpful."},
    )

    engine.return_value.generate_result.assert_called_once_with(
        "You are helpful.\n\nhello", model=MODEL
    )


def test_generate_cuts_the_response_at_a_stop_sequence(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_result.return_value = GenerationResult(text="before STOP after")

    response = client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "hello", "options": {"stop": ["STOP"]}},
    )

    assert response.json()["response"] == "before "


def test_generate_rejects_an_unknown_model(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_result.side_effect = ModelMismatchError("model 'x' not found")

    response = client.post("/api/generate", json={"model": "other-model", "prompt": "hello"})

    assert response.status_code == 404
    assert response.json() == {"error": "model 'x' not found"}


def test_generate_reports_a_model_that_cannot_be_loaded(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_result.side_effect = ModelInitializationError("no airllm")

    response = client.post("/api/generate", json={"model": MODEL, "prompt": "hello"})

    assert response.status_code == 503
    assert response.json() == {"error": "no airllm"}


def test_generate_requires_a_model(client: TestClient, engine: Any) -> None:
    response = client.post("/api/generate", json={"prompt": "hello"})

    assert response.status_code == 400
    assert "error" in response.json()


def test_generate_rejects_an_invalid_payload(client: TestClient, engine: Any) -> None:
    response = client.post("/api/generate", json={"model": MODEL, "options": "not-an-object"})

    assert response.status_code == 400
    assert "error" in response.json()


def test_generate_accepts_a_streaming_request(client: TestClient, engine: Any) -> None:
    """Streaming is not implemented, the answer is returned as one object."""
    response = client.post(
        "/api/generate", json={"model": MODEL, "prompt": "hello", "stream": True}
    )

    assert response.status_code == 200
    assert response.json()["done"] is True


def test_generate_schedules_the_unload(client: TestClient, engine: Any) -> None:
    client.post("/api/generate", json={"model": MODEL, "prompt": "hello", "keep_alive": "30s"})

    engine.return_value.schedule_unload.assert_called_once_with(30.0)


def test_generate_keeps_the_model_loaded_without_keep_alive(
    client: TestClient, engine: Any
) -> None:
    client.post("/api/generate", json={"model": MODEL, "prompt": "hello"})

    engine.return_value.schedule_unload.assert_not_called()


def test_generate_unloads_immediately_for_keep_alive_zero(client: TestClient, engine: Any) -> None:
    client.post("/api/generate", json={"model": MODEL, "prompt": "hello", "keep_alive": "0"})

    engine.return_value.schedule_unload.assert_called_once_with(0.0)


def test_generate_accepts_unknown_options(client: TestClient, engine: Any) -> None:
    """Clients send the full Ollama option set; unknown entries are ignored."""
    response = client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "hello", "options": {"mirostat": 2, "unknown": True}},
    )

    assert response.status_code == 200
    _, kwargs = engine.return_value.generate_result.call_args
    assert "mirostat" not in kwargs


def test_openapi_documents_the_generate_endpoint(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert "/api/generate" in response.json()["paths"]


def test_the_served_model_is_reported(client: TestClient, engine: Any) -> None:
    response = client.post("/api/generate", json={"model": get_settings().model_name})

    assert response.status_code == 200
    assert response.json()["model"] == get_settings().model_name
