"""Streaming helpers for the AirLLM engine.

The engine talks to AirLLM through ``transformers``' generation loop, which
can hand the produced tokens to a *streamer*. The streamer in this module turns
those tokens into text chunks; backends that do not support streaming (they
return the finished text instead) are handled by the engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from queue import Queue
from typing import Protocol, cast


class DecoderLike(Protocol):
    """The part of a tokenizer the streamer needs."""

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Decode a batch of token ids into strings."""
        ...


class TextStreamer:
    """Collects the tokens of a generation and turns them into text chunks.

    ``transformers`` calls :meth:`put` with the new token ids; depending on
    the version it either passes the new tokens or the whole sequence so far.
    Both variants are handled, so that the chunk boundaries are the token
    boundaries.
    """

    def __init__(self, decoder: DecoderLike) -> None:
        """Create a streamer decoding the tokens with ``decoder``."""
        self._decoder = decoder
        self._queue: Queue[str] = Queue()
        self._decoded = 0

    def put(self, value: object) -> None:
        """Handle a value produced by the generation loop."""
        text = self._to_text(value)
        if text:
            self._queue.put(text)

    def end(self) -> None:
        """Signal the end of the generation."""
        self._queue.put("")

    def __iter__(self) -> Iterator[str]:
        """Yield the produced chunks until the generation ends."""
        while True:
            chunk = self._queue.get()
            if not chunk:
                return
            yield chunk

    def _to_text(self, value: object) -> str:
        """Turn the value of the generation loop into text."""
        if isinstance(value, str):
            return value
        token_ids = _as_token_ids(value)
        if token_ids is None:
            return ""
        if len(token_ids) > self._decoded:
            # The backend sends the whole sequence so far.
            new_ids = token_ids[self._decoded :]
            self._decoded = len(token_ids)
        else:
            # The backend sends the new tokens only.
            new_ids = token_ids
            self._decoded += len(token_ids)
        if not new_ids:
            return ""
        decoded = self._decoder.batch_decode([new_ids], skip_special_tokens=True)
        return decoded[0] if decoded else ""


def _as_token_ids(value: object) -> list[int] | None:
    """Return the token ids of a value produced by the generation loop."""
    if isinstance(value, (list, tuple)):
        converted: list[object] = list(value)
    else:
        to_list = getattr(value, "tolist", None)
        if to_list is None:
            return None
        listed = to_list()
        if not isinstance(listed, list):
            return None
        converted = listed
    if converted and isinstance(converted[0], list):
        converted = converted[0]
    try:
        return [int(cast("int", item)) for item in converted]
    except (TypeError, ValueError):
        return None
