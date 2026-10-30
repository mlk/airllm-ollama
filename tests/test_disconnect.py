"""Tests for the streaming of the events and the abort of a generation."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from airllm_ollama.api.disconnect import stream_events


def test_the_events_reach_the_client() -> None:
    """The events of the stream are yielded one after the other."""

    async def scenario() -> None:
        abort = threading.Event()
        events = [event async for event in stream_events(iter(["first", "second"]), abort)]

        assert events == ["first", "second"]
        # The answer is complete: the generation is over anyway.
        assert abort.is_set() is True

    asyncio.run(scenario())


def test_the_generation_stops_when_the_stream_is_closed() -> None:
    """The client is gone: the generation stops."""

    async def scenario() -> None:
        abort = threading.Event()
        stream = stream_events(iter(["first", "second"]), abort)

        assert await anext(stream) == "first"
        await stream.aclose()

        assert abort.is_set() is True

    asyncio.run(scenario())


def test_the_generation_stops_when_the_request_is_cancelled() -> None:
    """A cancelled request stops the generation at once."""

    async def scenario() -> None:
        abort = threading.Event()
        waiting = threading.Event()

        def events() -> Iterator[str]:
            yield "first"
            waiting.set()
            # The model is still generating: it waits for the abort.
            abort.wait(timeout=5)

        stream = stream_events(events(), abort)
        assert await anext(stream) == "first"

        async def next_event() -> str:
            return await anext(stream)

        task = asyncio.create_task(next_event())
        # Wait until the stream waits for the next event.
        await asyncio.to_thread(waiting.wait, 5)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert abort.is_set() is True

    asyncio.run(scenario())
