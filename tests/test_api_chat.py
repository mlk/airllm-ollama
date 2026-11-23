"""Integration tests for the Ollama compatible ``/api/chat`` endpoint."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from airllm_ollama.chat import ChatTurn
from airllm_ollama.config import get_settings
from airllm_ollama.engine import GenerationResult, StreamChunk
from airllm_ollama.exceptions import (
    GenerationError,
    ModelInitializationError,
    ModelMismatchError,
)
from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction, ToolDefinition, ToolFunction
from fakes import FailingStream

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
        [ChatTurn("system", "Be brief."), ChatTurn("user", "Why is the sky blue?")],
        model=MODEL,
        stop=(),
        tools=None,
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


def test_chat_reports_a_failed_generation(client: TestClient, engine: Any) -> None:
    """A generation that fails is answered with the error shape of Ollama."""
    engine.return_value.generate_chat.side_effect = GenerationError("the checkpoint is broken")

    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    assert response.status_code == 503
    assert response.json() == {"error": "the checkpoint is broken"}


def test_a_stream_reports_a_failed_generation(client: TestClient, engine: Any) -> None:
    """The client gets the error as the last event of the stream."""
    engine.return_value.generate_chat_stream.side_effect = lambda *args, **kwargs: FailingStream(
        GenerationError("the checkpoint is broken")
    )

    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "stream": True})

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1] == {"error": "the checkpoint is broken"}


def test_chat_requires_messages(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"model": MODEL, "messages": []})

    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_requires_a_model(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"messages": MESSAGES})

    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_accepts_a_message_without_content(client: TestClient, engine: Any) -> None:
    """Assistant messages carrying tool calls may omit the content."""
    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": [{"role": "assistant", "tool_calls": []}]},
    )

    assert response.status_code == 200


def test_chat_rejects_an_invalid_message(client: TestClient, engine: Any) -> None:
    response = client.post("/api/chat", json={"model": MODEL, "messages": [{"content": "hi"}]})

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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

TOOL_CALL_TEXT = (
    'It is sunny. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
)


def test_chat_returns_the_tool_calls_of_the_model(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(
        text=TOOL_CALL_TEXT,
        prompt_tokens=7,
        generated_tokens=3,
        total_duration_ns=4_000,
    )

    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": MESSAGES, "tools": TOOLS},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "It is sunny."
    assert payload["message"]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    assert payload["done_reason"] == "stop"


def test_chat_reports_length_when_a_tool_call_is_returned_at_the_token_limit(
    client: TestClient, engine: Any
) -> None:
    """A complete call followed by truncation reports length, not stop."""
    engine.return_value.generate_chat.return_value = GenerationResult(
        text=(
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call> '
            '<tool_call>{"name": "get_time", "argume'
        ),
        prompt_tokens=7,
        generated_tokens=16,
        total_duration_ns=4_000,
    )

    response = client.post(
        "/api/chat",
        json={
            "model": MODEL,
            "messages": MESSAGES,
            "tools": TOOLS,
            "options": {"num_predict": 16},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [call["function"]["name"] for call in payload["message"]["tool_calls"]] == [
        "get_weather"
    ]
    # The truncated block is kept in the content and the reason is length.
    assert '<tool_call>{"name": "get_time"' in payload["message"]["content"]
    assert payload["done_reason"] == "length"


def test_chat_reports_stop_for_a_tool_call_within_the_limit(
    client: TestClient, engine: Any
) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(
        text='<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
        prompt_tokens=7,
        generated_tokens=16,
        total_duration_ns=4_000,
    )

    response = client.post(
        "/api/chat",
        json={
            "model": MODEL,
            "messages": MESSAGES,
            "tools": TOOLS,
            "options": {"num_predict": 64},
        },
    )

    payload = response.json()
    assert payload["done_reason"] == "stop"
    assert payload["message"]["content"] == ""


def test_chat_omits_the_tool_calls_when_the_model_does_not_call_a_tool(
    client: TestClient, engine: Any
) -> None:
    response = client.post(
        "/api/chat",
        json={"model": MODEL, "messages": MESSAGES, "tools": TOOLS},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "tool_calls" not in payload["message"]
    assert payload["message"]["content"] == "the sky is blue"


def test_chat_passes_the_tools_to_the_engine(client: TestClient, engine: Any) -> None:
    client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "tools": TOOLS})

    _, kwargs = engine.return_value.generate_chat.call_args
    assert kwargs["tools"] == [
        ToolDefinition(
            function=ToolFunction(
                name="get_weather",
                description="Get the weather of a city.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        )
    ]


def test_chat_does_not_parse_tool_calls_without_tools_in_the_request(
    client: TestClient, engine: Any
) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(
        text=TOOL_CALL_TEXT,
        prompt_tokens=7,
        generated_tokens=3,
    )

    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES})

    assert response.status_code == 200
    payload = response.json()
    assert "tool_calls" not in payload["message"]
    assert TOOL_CALL_TEXT in payload["message"]["content"]


def test_chat_forwards_the_tool_history_to_the_engine(client: TestClient, engine: Any) -> None:
    tool_call = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "22C", "tool_name": "get_weather"},
    ]

    client.post("/api/chat", json={"model": MODEL, "messages": messages, "tools": TOOLS})

    args, _ = engine.return_value.generate_chat.call_args
    assert args[0] == [
        ChatTurn("user", "Weather?"),
        ChatTurn(
            "assistant",
            "",
            tool_calls=(
                ToolCall(
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})
                ),
            ),
        ),
        ChatTurn("tool", "22C", tool_name="get_weather"),
    ]


def test_chat_keeps_a_malformed_call_alongside_a_valid_one(client: TestClient, engine: Any) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(
        text=(
            '<tool_call>{"broken": true</tool_call>'
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        ),
        prompt_tokens=7,
        generated_tokens=3,
    )

    response = client.post("/api/chat", json={"model": MODEL, "messages": MESSAGES, "tools": TOOLS})

    assert response.status_code == 200
    payload = response.json()
    assert [call["function"]["name"] for call in payload["message"]["tool_calls"]] == [
        "get_weather"
    ]
    assert '<tool_call>{"broken": true</tool_call>' in payload["message"]["content"]


def test_chat_parses_tool_calls_from_a_tool_history_without_tools(
    client: TestClient, engine: Any
) -> None:
    engine.return_value.generate_chat.return_value = GenerationResult(
        text='<tool_call>{"name": "get_weather", "arguments": {"city": "Lyon"}}</tool_call>',
        prompt_tokens=7,
        generated_tokens=3,
    )
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": "22C", "tool_name": "get_weather"},
    ]

    response = client.post("/api/chat", json={"model": MODEL, "messages": messages})

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"]["tool_calls"][0]["function"]["arguments"] == {"city": "Lyon"}
