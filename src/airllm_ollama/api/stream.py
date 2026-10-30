"""Streaming responses for the Ollama compatible endpoints.

Ollama streams newline delimited JSON: one JSON object per line, each one a
partial answer, the last one carrying the timings and ``done: true``. The
official clients parse the response line by line, so that framing is the
default here. Server-Sent Events can be selected with
``AIRLLM_OLLAMA_STREAM_FORMAT=sse``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

from airllm_ollama.config import get_settings
from airllm_ollama.engine import AirLLMEngine, GenerationResult, StreamChunk
from airllm_ollama.options import StopSequenceFilter, done_reason

NDJSON_MEDIA_TYPE = "application/x-ndjson"
SSE_MEDIA_TYPE = "text/event-stream"


def configured_format() -> str:
    """Return the configured framing of the streaming responses."""
    return get_settings().stream_format


def media_type() -> str:
    """Return the media type of the streaming responses."""
    return SSE_MEDIA_TYPE if configured_format() == "sse" else NDJSON_MEDIA_TYPE


def format_event(payload: dict[str, Any], stream_format: str | None = None) -> str:
    """Render a single streaming event.

    Args:
        payload: The JSON serialisable payload of the event.
        stream_format: The framing to use; defaults to the configured one.

    Returns:
        The event, framed for the stream format.
    """
    data = json.dumps(payload)
    if (stream_format or configured_format()) == "sse":
        return f"data: {data}\n\n"
    return f"{data}\n"


def generate_events(
    engine: AirLLMEngine,
    prompt: str,
    *,
    model: str,
    options: dict[str, object],
    system: str | None = None,
    raw: bool = False,
    stop: list[str] | None = None,
    abort: threading.Event | None = None,
) -> Iterator[str]:
    """Yield the streaming events of a completion.

    Args:
        engine: The engine that runs the generation.
        prompt: The prompt to complete.
        model: The model the client asked for.
        options: The translated generation options.
        system: An optional system message, rendered with the chat template.
        raw: Whether the prompt is sent unchanged, without the chat template.
        stop: The stop sequences requested by the client.
        abort: The event that stops the generation when the client is gone.

    Yields:
        The rendered events.
    """
    created_at = _now()
    stream_format = configured_format()
    yield from _stream_events(
        engine.generate_stream(
            prompt,
            system=system,
            raw=raw,
            model=model,
            stop=stop or (),
            abort=abort,
            **options,
        ),
        model=model,
        created_at=created_at,
        stream_format=stream_format,
        options=options,
        stop=stop,
        partial=lambda text: _partial_payload(model, created_at, text),
        final=lambda result: _final_payload(model, created_at, result, options),
    )


def chat_events(
    engine: AirLLMEngine,
    messages: list[tuple[str, str]],
    *,
    model: str,
    options: dict[str, object],
    stop: list[str] | None = None,
    abort: threading.Event | None = None,
) -> Iterator[str]:
    """Yield the streaming events of a chat answer.

    Args:
        engine: The engine that runs the generation.
        messages: The chat history as ``(role, content)`` pairs.
        model: The model the client asked for.
        options: The translated generation options.
        stop: The stop sequences requested by the client.
        abort: The event that stops the generation when the client is gone.

    Yields:
        The rendered events.
    """
    created_at = _now()
    stream_format = configured_format()
    yield from _stream_events(
        engine.generate_chat_stream(
            messages,
            model=model,
            stop=stop or (),
            abort=abort,
            **options,
        ),
        model=model,
        created_at=created_at,
        stream_format=stream_format,
        options=options,
        stop=stop,
        partial=lambda text: _partial_chat_payload(model, created_at, text),
        final=lambda result: _final_chat_payload(model, created_at, result, options),
    )


def _stream_events(
    chunks: Iterator[StreamChunk],
    *,
    model: str,
    created_at: str,
    stream_format: str,
    options: dict[str, object],
    stop: list[str] | None,
    partial: Callable[[str], dict[str, Any]],
    final: Callable[[GenerationResult | None], dict[str, Any]],
) -> Iterator[str]:
    """Yield the events of a stream.

    The stream is read until the generation is over: the last event carries the
    statistics of the generation. A stop sequence cuts the text, not the
    stream — the generation continues, only the text after the stop sequence is
    dropped.

    Args:
        chunks: The chunks of the generation.
        model: The model the client asked for.
        created_at: The timestamp of the answer.
        stream_format: The framing of the events.
        options: The translated generation options.
        stop: The stop sequences requested by the client.
        partial: The builder of the payload of a piece of text.
        final: The builder of the payload of the last event.

    Yields:
        The rendered events.
    """
    stop_filter = StopSequenceFilter(stop)
    result: GenerationResult | None = None
    for chunk in chunks:
        if chunk.result is not None:
            result = chunk.result
            break
        text = stop_filter.feed(chunk.text)
        if text:
            yield format_event(partial(text), stream_format)
    # The generation is over: the filter releases the tail it held back, it
    # cannot become a stop sequence any more.
    remaining = stop_filter.flush()
    if remaining:
        yield format_event(partial(remaining), stream_format)
    yield format_event(final(result), stream_format)


def _partial_payload(model: str, created_at: str, text: str) -> dict[str, Any]:
    """Return the payload of a partial completion."""
    return {"model": model, "created_at": created_at, "response": text, "done": False}


def _final_payload(
    model: str, created_at: str, result: GenerationResult | None, options: dict[str, object]
) -> dict[str, Any]:
    """Return the payload of the last completion event."""
    return {
        "model": model,
        "created_at": created_at,
        "response": "",
        "done": True,
        "done_reason": done_reason(result, options) if result else "stop",
        "total_duration": result.total_duration_ns if result else 0,
        "load_duration": result.load_duration_ns if result else 0,
        "prompt_eval_count": result.prompt_tokens if result else 0,
        "prompt_eval_duration": result.prompt_duration_ns if result else 0,
        "eval_count": result.generated_tokens if result else 0,
        "eval_duration": result.eval_duration_ns if result else 0,
    }


def _partial_chat_payload(model: str, created_at: str, text: str) -> dict[str, Any]:
    """Return the payload of a partial chat message."""
    return {
        "model": model,
        "created_at": created_at,
        "message": {"role": "assistant", "content": text},
        "done": False,
    }


def _final_chat_payload(
    model: str, created_at: str, result: GenerationResult | None, options: dict[str, object]
) -> dict[str, Any]:
    """Return the payload of the last chat event."""
    return {
        "model": model,
        "created_at": created_at,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": done_reason(result, options) if result else "stop",
        "total_duration": result.total_duration_ns if result else 0,
        "load_duration": result.load_duration_ns if result else 0,
        "prompt_eval_count": result.prompt_tokens if result else 0,
        "prompt_eval_duration": result.prompt_duration_ns if result else 0,
        "eval_count": result.generated_tokens if result else 0,
        "eval_duration": result.eval_duration_ns if result else 0,
    }


def _now() -> str:
    """Return the current time in the format Ollama uses."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
