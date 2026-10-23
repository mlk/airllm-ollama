"""Translation of Ollama's request options into engine keyword arguments.

Ollama's options are llama.cpp concepts; the engine drives a Hugging Face
model through ``transformers``. This module maps the options that have a
transformers counterpart and ignores the ones that have none.
"""

from __future__ import annotations

import re

from airllm_ollama.schemas.generate import GenerateOptions

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
