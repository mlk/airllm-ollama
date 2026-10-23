"""Integration tests for the Ollama compatible ``/api/chat`` endpoint."""

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
MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Why is the sky blue?"},
]


@pytest.fixture
def engine(mocker: MockerFixture) -> Any:
    """Patch the engine used by the API layer with a recording double."""
    double = mocker.patch("airllm_ollama.api.routes.get_engine")
    double.return_value.generate_chat.return_value = GenerationResult(
        text="the sky is blue",
        prompt_tokens=7,
        generated_tokens=3,
        load_duration_ns=1_000,
        prompt_duration_ns=2_000,
        eval_duration_ns=3_000,
        total_duration_ns=4_000,
    )
    double.return_value.is_loaded = False
    double.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(text="blue "),
            StreamChunk(text="sky"),
            StreamChunk(
                text="",
                result=GenerationResult(text="blue sky", prompt_tokens=7, generated_tokens=3),
            ),
        ]
    )
    return double


def test_chat_returns_the_ollama_response_shape(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == MODEL
    assert payload["message"] == {"role": "assistant", "content": "the sky is blue"}
    assert payload["done"] is True
    assert payload["done_reason"] == "stop"
    assert payload["eval_count"] == 3
    assert payload["prompt_eval_count"] == 7
    assert payload["total_duration"] == 4_000
    assert payload["load_duration"] == 1_000
    assert payload["created_at"].endswith("Z")


def test_chat_delegates_the_history_to_the_engine(client: TestClient, engine: Any) -> None:
    client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    engine.return_value.generate_chat.assert_called_once_with(
        [("system", "Be brief."), ("user", "Why is the sky blue?")], model=MODEL
    )


def test_chat_maps_ollama_options(client: TestClient, engine: Any) -> None:
    client.post(
        "/api/chat",
        json={
            "model": MODEL,
            "messages": MESSAGES,
            "options": {"temperature": 0.3, "num_predict": 32},
        },
    )

    _, kwargs = engine.return_value.generate_chat.call_args
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_new_tokens"] == 32
    assert kwargs["do_sample"] is True


def test_chat_cuts_the_answer_at_a_stop_sequence(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(text="before STOP after")

    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": MESSAGES, "options": {"stop": ["STOP"]}},
    )

    assert response.json()["message"]["content"] == "before "


def test_chat_reports_length_when_the_token_limit_is_reached(
    client: TestClient, engine: Any
) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(
        text="truncated", generated_tokens=64
    )

    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": MESSAGES, "options": {"num_predict": 64}},
    )

    assert response.json()["done_reason"] == "length"


def test_chat_rejects_an_unknown_model(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat.side_effect = ModelMismatchError("model 'x' not found")

    response = client.post("/api/chat", json={"model": "other-model", "messages": MESSAGES})

    assert response.status_code == 404
    assert response.json() == {"error": "model 'x' not found"}


def test_chat_reports_a_model_that_cannot_be_loaded(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat.side_effect = ModelInitializationError("no airllm")

    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    assert response.status_code == 503
    assert response.json() == {"error": "no airllm"}


def test_chat_requires_messages(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"model": MODEL, "messages": []})

    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_requires_a_model(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"messages": MESSAGES})

    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_rejects_an_invalid_message(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"model": MODEL, "messages": [{"role": "user"}]})

    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_streams_the_answer(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert [event["message"]["content"] for event in events[:-1]] == ["blue ", "sky"]
    assert events[0]["message"]["role"] == "assistant"
    assert events[0]["done"] is False
    assert events[-1]["done"] is True
    assert events[-1]["message"] == {"role": "assistant", "content": ""}
    assert events[-1]["eval_count"] == 3


def test_chat_cuts_the_stream_at_a_stop_sequence(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(text="before "),
            StreamChunk(text="STOP after"),
        ]
    )

    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": MESSAGES, "stream": True, "options": {"stop": ["STOP"]}},
    )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert "".join(event["message"]["content"] for event in events) == "before "


def test_chat_streams_the_rest_after_the_last_chunk(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [StreamChunk(text="blue "), StreamChunk(text="sky")]
    )

    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": True})

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert "".join(event["message"]["content"] for event in events) == "blue sky"


def test_chat_streams_a_held_back_tail(client: TestClient, engine: Any) -> None:
    """A tail that could still become a stop sequence is sent at the end."""
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [StreamChunk(text="blue ST")]
    )

    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": MESSAGES, "stream": True, "options": {"stop": ["STOP"]}},
    )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert "".join(event["message"]["content"] for event in events) == "blue ST"


def test_chat_schedules_the_unload(client: TestClient, engine: Any) -> None:
    client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "keep_alive": "2m"})

    engine.return_value.schedule_unload.assert_called_once_with(120.0)


def test_chat_uses_the_configured_default_keep_alive(
    client: TestClient, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_KEEP_ALIVE", "10m")
    get_settings.cache_clear()

    client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    engine.return_value.schedule_unload.assert_called_once_with(600.0)


def test_chat_keeps_the_model_loaded_without_keep_alive(client: TestClient, engine: Any) -> None:
    client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    engine.return_value.schedule_unload.assert_not_called()


def test_openapi_documents_the_chat_endpoint(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert "/api/chat" in response.json()["paths"]
