"""Integration tests for the Ollama compatible ``/api/generate`` endpoint."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from airllm_ollama.config import get_settings
from airllm_ollama.engine import GenerationResult, StreamChunk
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
    double.return_value.generate_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(text="the "),
            StreamChunk(text="sky"),
            StreamChunk(
                text="",
                result=GenerationResult(text="the sky", prompt_tokens=4, generated_tokens=3),
            ),
        ]
    )
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


def test_generate_streams_the_answer(client: TestClient, engine: Any) -> None:
    response = client.post(
        "/api/generate", json={"model": MODEL, "prompt": "hello", "stream": True}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert [event["response"] for event in events[:-1]] == ["the ", "sky"]
    assert events[0]["done"] is False
    assert events[0]["model"] == MODEL
    assert events[-1]["done"] is True
    assert events[-1]["response"] == ""
    assert events[-1]["eval_count"] == 3
    assert events[-1]["prompt_eval_count"] == 4


def test_generate_streams_the_rest_after_the_last_chunk(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_stream.side_effect = lambda *args, **kwargs: iter(
        [StreamChunk(text="the "), StreamChunk(text="sky")]
    )

    response = client.post(
        "/api/generate", json={"model": MODEL, "prompt": "hello", "stream": True}
    )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert "".join(event["response"] for event in events) == "the sky"


def test_a_stream_sends_a_held_back_tail(client: TestClient, engine: Any) -> None:
    """A tail that could still become a stop sequence is sent at the end."""
    engine.return_value.generate_stream.side_effect = lambda *args, **kwargs: iter(
        [StreamChunk(text="the ST")]
    )

    response = client.post(
        "/api/generate",
        json={
            "model": MODEL,
            "prompt": "hello",
            "stream": True,
            "options": {"stop": ["STOP"]},
        },
    )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert "".join(event["response"] for event in events) == "the ST"


def test_a_stream_schedules_the_unload(client: TestClient, engine: Any) -> None:
    client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "hello", "stream": True, "keep_alive": "90s"},
    )

    engine.return_value.schedule_unload.assert_called_once_with(90.0)


def test_a_stream_reports_length_when_the_token_limit_is_reached(
    client: TestClient, engine: Any
) -> None:
    engine.return_value.generate_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(
                text="",
                result=GenerationResult(text="truncated", prompt_tokens=4, generated_tokens=64),
            )
        ]
    )

    response = client.post(
        "/api/generate",
        json={
            "model": MODEL,
            "prompt": "hello",
            "stream": True,
            "options": {"num_predict": 64},
        },
    )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1]["done_reason"] == "length"


def test_a_stream_reports_stop_when_the_model_finished(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(
                text="",
                result=GenerationResult(text="done", prompt_tokens=4, generated_tokens=3),
            )
        ]
    )

    response = client.post(
        "/api/generate",
        json={
            "model": MODEL,
            "prompt": "hello",
            "stream": True,
            "options": {"num_predict": 64},
        },
    )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1]["done_reason"] == "stop"


def test_a_stream_rejects_an_unknown_model(client: TestClient, engine: Any) -> None:
    engine.return_value.prepare.side_effect = ModelMismatchError("model 'x' not found")

    response = client.post(
        "/api/generate", json={"model": "other-model", "prompt": "hello", "stream": True}
    )

    assert response.status_code == 404
    assert response.json() == {"error": "model 'x' not found"}


def test_a_stream_reports_a_model_that_cannot_be_loaded(client: TestClient, engine: Any) -> None:
    engine.return_value.prepare.side_effect = ModelInitializationError("no airllm")

    response = client.post(
        "/api/generate", json={"model": MODEL, "prompt": "hello", "stream": True}
    )

    assert response.status_code == 503
    assert response.json() == {"error": "no airllm"}


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


def test_generate_reports_stop_when_the_model_finished(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_result.return_value = GenerationResult(
        text="complete", generated_tokens=3
    )

    response = client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "hello", "options": {"num_predict": 128}},
    )

    assert response.json()["done_reason"] == "stop"


def test_generate_reports_length_when_the_token_limit_is_reached(
    client: TestClient, engine: Any
) -> None:
    engine.return_value.generate_result.return_value = GenerationResult(
        text="truncated", generated_tokens=128
    )

    response = client.post(
        "/api/generate",
        json={"model": MODEL, "prompt": "hello", "options": {"num_predict": 128}},
    )

    assert response.json()["done_reason"] == "length"


def test_generate_uses_the_configured_default_keep_alive(
    client: TestClient, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_KEEP_ALIVE", "10m")
    get_settings.cache_clear()

    client.post("/api/generate", json={"model": MODEL, "prompt": "hello"})

    engine.return_value.schedule_unload.assert_called_once_with(600.0)


def test_the_request_keep_alive_wins_over_the_configured_default(
    client: TestClient, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_KEEP_ALIVE", "10m")
    get_settings.cache_clear()

    client.post("/api/generate", json={"model": MODEL, "prompt": "hello", "keep_alive": "5s"})

    engine.return_value.schedule_unload.assert_called_once_with(5.0)


def test_openapi_documents_the_generate_endpoint(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert "/api/generate" in response.json()["paths"]


def test_the_served_model_is_reported(client: TestClient, engine: Any) -> None:
    response = client.post("/api/generate", json={"model": get_settings().model_name})

    assert response.status_code == 200
    assert response.json()["model"] == get_settings().model_name
