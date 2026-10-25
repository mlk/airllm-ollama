"""Tests for the OpenAI compatible endpoints."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from airllm_ollama.config import get_settings
from airllm_ollama.engine import GenerationResult, StreamChunk
from airllm_ollama.exceptions import ModelInitializationError

MODEL = "test-model"
MESSAGES = [{"role": "user", "content": "why is the sky blue?"}]


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Serve the test model."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", MODEL)
    get_settings.cache_clear()
    return MODEL


@pytest.fixture
def engine(mocker: MockerFixture) -> Any:
    """Patch the engine used by the OpenAI routes with a recording double."""
    double = mocker.patch("airllm_ollama.api.openai.get_engine")
    double.return_value.generate_chat.return_value = GenerationResult(
        text="the sky is blue",
        prompt_tokens=4,
        generated_tokens=3,
    )
    double.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
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


def test_list_models_returns_the_openai_shape(client: TestClient, model: str) -> None:
    """The list of the models mirrors the OpenAI API."""
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    card = payload["data"][0]
    assert card["id"] == model
    assert card["object"] == "model"
    assert card["owned_by"] == "airllm"
    assert isinstance(card["created"], int)
    assert card["created"] > 0


def test_list_models_reports_the_configured_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id of the model is the configured one."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "other/model")
    get_settings.cache_clear()

    payload = client.get("/v1/models").json()

    assert payload["data"][0]["id"] == "other/model"


def test_retrieve_model_returns_the_model(client: TestClient, model: str) -> None:
    """A single model is returned in the OpenAI shape."""
    response = client.get(f"/v1/models/{model}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == model
    assert payload["object"] == "model"
    assert payload["owned_by"] == "airllm"


def test_retrieve_model_accepts_a_tag(client: TestClient, model: str) -> None:
    """A tagged reference resolves to the served model, like Ollama's does."""
    response = client.get(f"/v1/models/{model}:latest")

    assert response.status_code == 200
    assert response.json()["id"] == model


def test_retrieve_model_accepts_a_name_with_a_slash(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hugging Face repository id contains a slash."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "org/model")
    get_settings.cache_clear()

    response = client.get("/v1/models/org/model")

    assert response.status_code == 200
    assert response.json()["id"] == "org/model"


def test_retrieve_model_rejects_another_model(client: TestClient, model: str) -> None:
    """An unknown model is answered with the OpenAI error format."""
    response = client.get("/v1/models/other-model")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["message"] == "The model 'other-model' does not exist"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "model"
    assert payload["error"]["code"] == "model_not_found"


def test_the_ollama_endpoints_keep_working(client: TestClient, model: str) -> None:
    """The OpenAI routes do not disturb the Ollama ones."""
    response = client.get("/api/version")

    assert response.status_code == 200
    assert "version" in response.json()


def test_chat_completions_returns_the_openai_shape(
    client: TestClient, model: str, engine: Any
) -> None:
    """The completion mirrors the OpenAI API."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == model
    assert payload["id"].startswith("chatcmpl-airllm-")
    assert isinstance(payload["created"], int)
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "the sky is blue"},
            "finish_reason": "stop",
        }
    ]
    assert payload["usage"] == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}


def test_chat_completions_forwards_the_messages(
    client: TestClient, model: str, engine: Any
) -> None:
    """The messages are rendered with the chat template of the model."""
    client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    args, kwargs = engine.return_value.generate_chat.call_args
    assert args[0] == [("system", "Be terse."), ("user", "Hello")]
    assert kwargs["model"] == model


def test_chat_completions_translates_the_options(
    client: TestClient, model: str, engine: Any
) -> None:
    """The OpenAI parameters are mapped to the ones of the engine."""
    client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": MESSAGES,
            "temperature": 0.5,
            "max_tokens": 16,
        },
    )

    _, kwargs = engine.return_value.generate_chat.call_args
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_new_tokens"] == 16
    assert kwargs["do_sample"] is True


def test_chat_completions_rejects_another_model(
    client: TestClient, model: str, engine: Any
) -> None:
    """An unknown model is answered with the OpenAI error format."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "other-model", "messages": MESSAGES},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["message"] == (
        f"Model 'other-model' not found. Server only supports '{model}'."
    )
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "model"
    assert payload["error"]["code"] == "model_not_found"
    engine.return_value.generate_chat.assert_not_called()


def test_chat_completions_answers_a_malformed_body_in_the_openai_format(
    client: TestClient, model: str, engine: Any
) -> None:
    """The OpenAI routes answer with the OpenAI error format."""
    response = client.post("/v1/chat/completions", json={"model": model})

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "messages" in payload["error"]["message"]


def test_chat_completions_cuts_at_a_stop_sequence(
    client: TestClient, model: str, engine: Any
) -> None:
    """The answer is cut at the first stop sequence."""
    engine.return_value.generate_chat.return_value = GenerationResult(text="before STOP after")

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stop": "STOP"},
    )

    assert response.json()["choices"][0]["message"]["content"] == "before "


def test_chat_completions_streams_server_sent_events(
    client: TestClient, model: str, engine: Any
) -> None:
    """A streamed answer is a stream of Server-Sent Events."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks) == "the sky"
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"]["content"] == ""


def test_chat_completions_streams_a_cut_answer(client: TestClient, model: str, engine: Any) -> None:
    """A streamed answer is cut at the first stop sequence."""
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(text="before "),
            StreamChunk(text="STOP after"),
            StreamChunk(
                text="",
                result=GenerationResult(
                    text="before STOP after", prompt_tokens=4, generated_tokens=3
                ),
            ),
        ]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stop": "STOP", "stream": True},
    )

    events = [line for line in response.text.splitlines() if line]
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks) == "before "


def test_a_stream_ends_with_the_last_chunk(client: TestClient, model: str, engine: Any) -> None:
    """The client always gets the last chunk, even when a stop sequence was found."""
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(text="before "),
            StreamChunk(text="STOP after"),
        ]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stop": "STOP", "stream": True},
    )

    events = [line for line in response.text.splitlines() if line]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_stream_releases_the_tail_of_a_stop_sequence(
    client: TestClient, model: str, engine: Any
) -> None:
    """The text that could still become a stop sequence is released at the end."""
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [
            StreamChunk(text="the sky is "),
            StreamChunk(text="STO"),
            StreamChunk(
                text="",
                result=GenerationResult(text="the sky is STO", prompt_tokens=4, generated_tokens=3),
            ),
        ]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stop": "STOP", "stream": True},
    )

    events = [line for line in response.text.splitlines() if line]
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks) == "the sky is STO"


def test_chat_completions_accepts_a_list_of_stop_sequences(
    client: TestClient, model: str, engine: Any
) -> None:
    """The stop sequences may be sent as a list."""
    engine.return_value.generate_chat.return_value = GenerationResult(text="before STOP after")

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stop": ["STOP", "END"]},
    )

    assert response.json()["choices"][0]["message"]["content"] == "before "


def test_a_stream_without_statistics_still_ends(
    client: TestClient, model: str, engine: Any
) -> None:
    """A stream that ends without the statistics still ends with ``[DONE]``."""
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: iter(
        [StreamChunk(text="the sky")]
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stream": True},
    )

    events = [line for line in response.text.splitlines() if line]
    assert events[-1] == "data: [DONE]"


def test_a_stream_loads_the_model_before_the_headers_are_sent(
    client: TestClient, model: str, engine: Any
) -> None:
    """A broken checkpoint becomes a 503, not a truncated stream."""
    engine.return_value.prepare.side_effect = ModelInitializationError("cannot load the model")

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": MESSAGES, "stream": True},
    )

    assert response.status_code == 503
    assert "error" in response.json()


def test_the_model_is_unloaded_after_the_configured_idle_time(
    client: TestClient, model: str, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keep alive setting is honoured, like in the Ollama routes."""
    monkeypatch.setenv("AIRLLM_OLLAMA_KEEP_ALIVE", "5m")
    get_settings.cache_clear()

    client.post("/v1/chat/completions", json={"model": model, "messages": MESSAGES})

    engine.return_value.schedule_unload.assert_called_once_with(300.0)


def test_the_model_stays_loaded_without_a_keep_alive(
    client: TestClient, model: str, engine: Any
) -> None:
    """Without a lifetime the model stays in memory."""
    client.post("/v1/chat/completions", json={"model": model, "messages": MESSAGES})

    engine.return_value.schedule_unload.assert_not_called()
