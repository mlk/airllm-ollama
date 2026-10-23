"""Unit tests for the streaming helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from airllm_ollama.api.stream import format_event, media_type
from airllm_ollama.config import get_settings
from airllm_ollama.options import StopSequenceFilter
from airllm_ollama.streaming import TextStreamer


class FakeDecoder:
    """Decoder double turning token ids into readable text."""

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Return the ids of every sequence as a string."""
        batch = sequences if isinstance(sequences, list) else [sequences]
        return [f"<{ids}>" for ids in batch]


class FakeTensor:
    """Stand-in for a torch tensor."""

    def __init__(self, values: list[int]) -> None:
        """Store the values of the fake tensor."""
        self._values = values

    def tolist(self) -> list[int]:
        """Return the values as a list."""
        return list(self._values)


def test_the_streamer_yields_the_new_tokens() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put([1, 2])
    streamer.put([3])
    streamer.end()

    assert list(streamer) == ["<[1, 2]>", "<[3]>"]


def test_the_streamer_handles_a_growing_sequence() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put([1, 2])
    streamer.put([1, 2, 3])
    streamer.put([1, 2, 3, 4])
    streamer.end()

    assert list(streamer) == ["<[1, 2]>", "<[3]>", "<[4]>"]


def test_the_streamer_handles_tensors() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put(FakeTensor([5, 6]))
    streamer.put(FakeTensor([5, 6, 7]))
    streamer.end()

    assert list(streamer) == ["<[5, 6]>", "<[7]>"]


def test_the_streamer_handles_nested_token_ids() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put([[1, 2]])
    streamer.end()

    assert list(streamer) == ["<[1, 2]>"]


def test_the_streamer_handles_nested_tensors() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put(FakeTensor([5, 6]))
    streamer.end()

    assert list(streamer) == ["<[5, 6]>"]


def test_the_streamer_accepts_plain_text() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put("hello")
    streamer.end()

    assert list(streamer) == ["hello"]


def test_the_streamer_ignores_a_tensor_that_is_not_a_list() -> None:
    class WeirdTensor:
        """Stand-in for a tensor with an unexpected content."""

        def tolist(self) -> str:
            """Return something that is not a list."""
            return "not a list"

    streamer = TextStreamer(FakeDecoder())

    streamer.put(WeirdTensor())
    streamer.end()

    assert list(streamer) == []


def test_the_streamer_ignores_values_that_are_no_integers() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put(["a", "b"])
    streamer.end()

    assert list(streamer) == []


def test_the_streamer_ignores_unknown_values() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put(object())
    streamer.end()

    assert list(streamer) == []


def test_the_streamer_ignores_empty_chunks() -> None:
    streamer = TextStreamer(FakeDecoder())

    streamer.put([])
    streamer.end()

    assert list(streamer) == []


def test_the_stop_filter_cuts_at_the_stop_sequence() -> None:
    stop_filter = StopSequenceFilter(["STOP"])

    assert stop_filter.feed("before ") == "before "
    # The stop sequence itself is not part of the answer.
    assert stop_filter.feed("STOP after") == ""
    assert stop_filter.stopped is True
    assert stop_filter.feed("more") is None
    assert stop_filter.flush() == ""


def test_the_stop_filter_keeps_a_possible_stop_sequence() -> None:
    stop_filter = StopSequenceFilter(["STOP"])

    # The tail that could still become a stop sequence is held back.
    assert stop_filter.feed("text ST") == "text "
    assert stop_filter.feed("OP rest") == ""
    assert stop_filter.stopped is True


def test_the_stop_filter_flushes_the_rest() -> None:
    stop_filter = StopSequenceFilter(["STOP"])

    assert stop_filter.feed("hello") == "hello"
    assert stop_filter.flush() == ""


def test_the_stop_filter_without_stop_sequences() -> None:
    stop_filter = StopSequenceFilter(None)

    assert stop_filter.feed("hello") == "hello"
    assert stop_filter.flush() == ""
    assert stop_filter.stopped is False


def test_the_stop_filter_ignores_empty_stop_sequences() -> None:
    stop_filter = StopSequenceFilter([""])

    assert stop_filter.feed("hello") == "hello"
    assert stop_filter.stopped is False


def test_the_events_use_newline_delimited_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_STREAM_FORMAT", "ndjson")
    get_settings.cache_clear()

    event = format_event({"response": "hi", "done": False})

    assert event == '{"response": "hi", "done": false}\n'
    assert media_type() == "application/x-ndjson"


def test_the_events_can_use_server_sent_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_STREAM_FORMAT", "sse")
    get_settings.cache_clear()

    event = format_event({"response": "hi", "done": False})

    assert event == 'data: {"response": "hi", "done": false}\n\n'
    assert media_type() == "text/event-stream"


def test_an_unknown_stream_format_falls_back_to_ndjson(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_STREAM_FORMAT", "unknown")
    get_settings.cache_clear()

    assert media_type() == "application/x-ndjson"


def _drain(events: Iterator[str]) -> list[str]:
    """Collect the events of a stream."""
    return list(events)
