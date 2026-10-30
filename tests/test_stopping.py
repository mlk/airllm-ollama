"""Unit tests for the stopping criterion of a generation."""

from __future__ import annotations

import threading
from typing import cast

import pytest

from airllm_ollama.stopping import StopGeneration


class FakeIds:
    """The token ids of a fake generation."""

    def __init__(self, ids: list[int]) -> None:
        """Create the token ids.

        Args:
            ids: The ids of the generated tokens.
        """
        self.ids = ids

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the tensor."""
        return (1, len(self.ids))

    @property
    def device(self) -> object:
        """Return the device of the tensor."""
        return "cpu"

    def __getitem__(self, index: object) -> FakeIds:
        """Return a part of the tensor.

        The double holds a single sequence, so the first row is the sequence
        itself.
        """
        if isinstance(index, slice):
            return FakeIds(self.ids[index])
        return FakeIds(self.ids)

    def tolist(self) -> list[object]:
        """Return the tensor as a list."""
        return list(self.ids)


class FakeTokenizer:
    """Tokenizer double that decodes the ids to their letters."""

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        """Return the token ids of the text."""
        return {"input_ids": [ord(letter) for letter in text]}

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Decode the token ids to their letters."""
        decoded = cast("list[list[int]]", sequences)
        return ["".join(chr(letter) for letter in ids) for ids in decoded]


def test_the_tail_holds_the_longest_stop_sequence() -> None:
    """The tail has to hold the stop sequence and a margin."""
    criterion = StopGeneration(FakeTokenizer(), ["STOP", "a much longer sequence"])

    assert criterion._tail == len("a much longer sequence") + 4


def test_without_a_stop_sequence_the_tail_is_empty() -> None:
    """Without a stop sequence there is nothing to look for."""
    criterion = StopGeneration(FakeTokenizer(), [])

    assert criterion._tail == 0


def test_the_criterion_stops_at_a_stop_sequence() -> None:
    """The generation ends when the answer contains a stop sequence."""
    pytest.importorskip("torch")
    criterion = StopGeneration(FakeTokenizer(), ["STOP"])
    ids = FakeIds([ord(letter) for letter in "the answer STOP"])

    assert bool(criterion(ids)) is True


def test_the_criterion_keeps_going_without_a_stop_sequence() -> None:
    """The generation continues while the answer does not contain one."""
    pytest.importorskip("torch")
    criterion = StopGeneration(FakeTokenizer(), ["STOP"])
    ids = FakeIds([ord(letter) for letter in "the answer"])

    assert bool(criterion(ids)) is False


def test_a_stop_sequence_in_the_prompt_does_not_stop_the_generation() -> None:
    """Only the generated tokens are looked at."""
    pytest.importorskip("torch")
    criterion = StopGeneration(FakeTokenizer(), ["STOP"], prompt_length=len("say STOP "))
    ids = FakeIds([ord(letter) for letter in "say STOP now"])

    assert bool(criterion(ids)) is False


def test_the_criterion_stops_at_a_stop_sequence_of_the_answer() -> None:
    """The tokens of the prompt are left out of the tail."""
    pytest.importorskip("torch")
    criterion = StopGeneration(FakeTokenizer(), ["STOP"], prompt_length=len("say "))
    ids = FakeIds([ord(letter) for letter in "say STOP now"])

    assert bool(criterion(ids)) is True


def test_the_criterion_stops_when_the_client_is_gone() -> None:
    """The generation ends when the client disconnects."""
    pytest.importorskip("torch")
    abort = threading.Event()
    criterion = StopGeneration(FakeTokenizer(), [], abort)
    ids = FakeIds([ord(letter) for letter in "the answer"])

    assert bool(criterion(ids)) is False

    abort.set()

    assert bool(criterion(ids)) is True
