"""Streaming the events of a generation to the client.

The events are produced by a generator that blocks on the model, so they are
read in a thread: the event loop of the service stays free while the model
generates. The thread is abandoned when the request is cancelled, so a client
that disconnects is noticed at once — the generation stops at the next token
instead of producing an answer nobody waits for.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncGenerator, Iterator

from anyio import to_thread


async def stream_events(events: Iterator[str], abort: threading.Event) -> AsyncGenerator[str, None]:
    """Yield the events of a stream and stop the generation when the client is gone.

    Args:
        events: The events of the stream.
        abort: The event that stops the generation. It is set as soon as the
            stream is closed or the request is cancelled.

    Yields:
        The events of the stream.
    """
    iterator = iter(events)
    try:
        while True:
            event = await to_thread.run_sync(_next_event, iterator, abandon_on_cancel=True)
            if event is None:
                break
            yield event
    finally:
        # The client is gone (or the answer is complete): the model stops at
        # the next token, so that it is free for the next request.
        abort.set()


def _next_event(iterator: Iterator[str]) -> str | None:
    """Return the next event of the stream, or None when it is over.

    Args:
        iterator: The events of the stream.

    Returns:
        The next event, or None at the end of the stream.
    """
    try:
        return next(iterator)
    except StopIteration:
        return None
