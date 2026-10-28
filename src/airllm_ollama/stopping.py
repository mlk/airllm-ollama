"""Stopping the generation of a model.

Two things end a generation early: a stop sequence of the request and a client
that is gone. Both are checked between two generated tokens, so the model stops
at the next token instead of producing an answer nobody waits for.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast


class TokenIds(Protocol):
    """The token ids of a generation."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the tensor."""
        ...

    @property
    def device(self) -> object:
        """Return the device of the tensor."""
        ...

    def __getitem__(self, index: object) -> TokenIds:
        """Return a part of the tensor."""
        ...

    def tolist(self) -> list[object]:
        """Return the tensor as a list."""
        ...


class TokenizerLike(Protocol):
    """The part of the tokenizer the criterion needs."""

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        """Tokenize the text."""
        ...

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Decode the token ids into strings."""
        ...


class StopGeneration:
    """Stop the generation when a stop sequence appears or the client is gone.

    The criterion is called between two generated tokens, so the generation ends
    at the next token. Only the generated tokens are looked at: a stop sequence
    in the prompt does not end the generation.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        sequences: Sequence[str] = (),
        abort: threading.Event | None = None,
        prompt_length: int = 0,
    ) -> None:
        """Create the criterion.

        Args:
            tokenizer: The tokenizer of the model, used to decode the tail of
                the answer.
            sequences: The stop sequences of the request.
            abort: The event that is set when the client is gone. A new event is
                used when the caller has none.
            prompt_length: The number of tokens of the prompt. The tokens before
                it are not part of the answer and are ignored.
        """
        self._tokenizer = tokenizer
        self._sequences = [sequence for sequence in sequences if sequence]
        self.abort = abort if abort is not None else threading.Event()
        self._prompt_length = prompt_length
        self._tail = self._tail_size()

    def __call__(self, input_ids: TokenIds, scores: object = None, **kwargs: object) -> object:
        """Return whether the generation has to stop.

        Args:
            input_ids: The prompt and the tokens generated so far.
            scores: The scores of the generation — not used.
            **kwargs: The other arguments of the criterion.

        Returns:
            A tensor that tells for every sequence whether it has to stop.
        """
        import torch

        stopped = self.abort.is_set()
        if not stopped and self._sequences:
            stopped = self._found(self._tail_text(input_ids))
        return torch.full(
            (input_ids.shape[0],),
            stopped,
            dtype=torch.bool,
            device=cast("Any", input_ids.device),
        )

    def _found(self, text: str) -> bool:
        """Return whether the text contains one of the stop sequences."""
        return any(sequence in text for sequence in self._sequences)

    def _tail_text(self, input_ids: TokenIds) -> str:
        """Return the text of the last generated tokens.

        Only the tail is decoded: the stop sequence has to be at the end of
        the answer, and the tail is long enough to hold it. The tokens of the
        prompt are left out — a stop sequence in the prompt does not end the
        generation.

        Args:
            input_ids: The prompt and the tokens generated so far.

        Returns:
            The text of the tail.
        """
        if not self._tail:
            return ""
        generated = input_ids[0][self._prompt_length :]
        tail = generated[-self._tail :].tolist()
        decoded = self._tokenizer.batch_decode([cast("list[int]", tail)], skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def _tail_size(self) -> int:
        """Return the number of tokens the tail of the answer needs.

        The tail has to hold the longest stop sequence and a few tokens of
        margin, so that a stop sequence that is split over two tokens is found
        as well.

        Returns:
            The number of tokens.
        """
        if not self._sequences:
            return 0
        longest = max(
            len(cast("list[int]", self._tokenizer(sequence, add_special_tokens=False)["input_ids"]))
            for sequence in self._sequences
        )
        return longest + 4
