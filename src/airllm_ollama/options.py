"""Translation of Ollama's request options into engine keyword arguments.

Ollama's options are llama.cpp concepts; the engine drives a Hugging Face
model through ``transformers``. This module maps the options that have a
transformers counterpart and ignores the ones that have none.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from airllm_ollama.schemas.generate import GenerateOptions

if TYPE_CHECKING:
    from airllm_ollama.engine import GenerationResult

DURATION_PATTERN = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smh]?)\s*$")
UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600}


def map_options(options: GenerateOptions | None) -> dict[str, object]:
    """Translate Ollama's options into keyword arguments for the engine.

    Args:
        options: The options sent by the client, if any.

    Returns:
        Keyword arguments understood by the underlying model.
    """
    if options is None:
        return {}

    mapped: dict[str, object] = {}
    sampling = False
    if options.temperature is not None:
        mapped["temperature"] = options.temperature
        sampling = True
    if options.top_p is not None:
        mapped["top_p"] = options.top_p
        sampling = True
    if options.top_k is not None:
        mapped["top_k"] = options.top_k
        sampling = True
    if options.num_predict is not None:
        mapped["max_new_tokens"] = options.num_predict
    if options.num_ctx is not None:
        mapped["max_length"] = options.num_ctx
    if options.repeat_penalty is not None:
        mapped["repetition_penalty"] = options.repeat_penalty
    if sampling:
        # transformers only applies the sampling parameters when sampling is
        # enabled; Ollama always samples.
        mapped["do_sample"] = True
    return mapped


def done_reason(result: GenerationResult, options: Mapping[str, object]) -> str:
    """Return Ollama's reason for the end of a generation.

    The generation stops either because the model finished its answer or
    because it reached the requested number of tokens.

    Args:
        result: The result of the generation.
        options: The translated generation options.

    Returns:
        ``"stop"`` or ``"length"``.
    """
    max_new_tokens = options.get("max_new_tokens")
    if isinstance(max_new_tokens, int) and result.generated_tokens >= max_new_tokens:
        return "length"
    return "stop"


def apply_stop_sequences(text: str, stop: list[str] | None) -> str:
    """Cut ``text`` at the first stop sequence, the way Ollama does.

    Args:
        text: The generated text.
        stop: The stop sequences requested by the client.

    Returns:
        The text without the stop sequence and everything behind it.
    """
    if not stop:
        return text
    cut = len(text)
    for sequence in stop:
        if not sequence:
            continue
        index = text.find(sequence)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


class StopSequenceFilter:
    """Cuts a stream of text at the first stop sequence.

    The filter keeps back the tail of the text that could still become the
    beginning of a stop sequence, so a stop sequence spanning two chunks is
    detected as well.
    """

    def __init__(self, stop: list[str] | None) -> None:
        """Create a filter for the given stop sequences."""
        self._stop = [sequence for sequence in (stop or []) if sequence]
        self._text = ""
        self._emitted = 0
        self._stopped = False

    @property
    def stopped(self) -> bool:
        """Return whether a stop sequence was seen."""
        return self._stopped

    def feed(self, chunk: str) -> str | None:
        """Return the part of ``chunk`` that may be sent to the client.

        Args:
            chunk: The next piece of generated text.

        Returns:
            The text that may be emitted, or ``None`` when the generation has
            to stop.
        """
        if self._stopped:
            return None
        self._text += chunk
        cut = self._stop_index()
        if cut != -1:
            self._stopped = True
            return self._pending(self._text[:cut])
        return self._pending(self._text[: self._safe_length()])

    def flush(self) -> str:
        """Return the text that is still held back."""
        if self._stopped:
            return ""
        return self._pending(self._text)

    def _pending(self, text: str) -> str:
        """Return the part of ``text`` that was not emitted yet."""
        if len(text) <= self._emitted:
            return ""
        pending = text[self._emitted :]
        self._emitted = len(text)
        return pending

    def _stop_index(self) -> int:
        """Return the index of the first stop sequence, or ``-1``."""
        cut = -1
        for sequence in self._stop:
            index = self._text.find(sequence)
            if index != -1 and (cut == -1 or index < cut):
                cut = index
        return cut

    def _safe_length(self) -> int:
        """Return the length of the text that can be emitted safely."""
        if not self._stop:
            return len(self._text)
        longest = max(len(sequence) for sequence in self._stop)
        keep = min(longest - 1, len(self._text))
        while keep > 0:
            tail = self._text[-keep:]
            if any(sequence.startswith(tail) for sequence in self._stop):
                break
            keep -= 1
        return len(self._text) - keep


def parse_keep_alive(keep_alive: str | None) -> float | None:
    """Parse Ollama's ``keep_alive`` value into a number of seconds.

    Args:
        keep_alive: The raw value, e.g. ``"5m"``, ``"30s"``, ``"1h"`` or ``"0"``.

    Returns:
        The number of seconds the model should stay loaded, or ``None`` when
        the client did not ask for a specific lifetime.
    """
    if keep_alive is None:
        return None
    match = DURATION_PATTERN.match(keep_alive)
    if match is None:
        return None
    value = float(match.group("value"))
    return value * UNIT_SECONDS[match.group("unit")]
