"""Extraction of the tool calls a model reports in its answer.

Tool-calling models (Hermes, Qwen, ...) answer with a ``<tool_call>`` block
around a JSON object::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

The text is decoded from the tokens the model generated; the block is an
ordinary string in it, so a plain text scan is enough — no constrained
decoding is needed. The blocks that do not hold a valid call are left in the
text untouched: a call the model garbled is not silently dropped, and the
client sees the same text the model produced.
"""

from __future__ import annotations

import json
import math

from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_WHITESPACE = " \t\r\n"


def _reject_constant(value: str) -> None:
    """Reject the non-standard JSON constants (``NaN``, ``Infinity``)."""
    raise ValueError(f"invalid constant: {value}")


def _finite_float(value: str) -> float:
    """Parse a float, rejecting values outside the finite range."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite number: {value}")
    return parsed


def _strict_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(parse_constant=_reject_constant, parse_float=_finite_float)


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split the text of an answer into its plain text and its tool calls.

    Args:
        text: The decoded answer of the model.

    Returns:
        The text with the valid tool call blocks removed (stripped) and the
        tool calls that were found, in the order they appear. When the text
        holds no valid tool call, it is returned unchanged.
    """
    decoder = _strict_decoder()
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        start = text.find(_OPEN, position)
        if start == -1:
            break
        found = _parse_call(text, start + len(_OPEN), decoder)
        if found is None:
            # Not a valid call: the whole block is dead text. Skipping to the
            # matching closing marker (counting nested blocks) keeps a marker
            # inside the malformed block from being picked up as a call of
            # its own, no matter how deep the nesting.
            position = _skip_block(text, start)
            continue
        call, end = found
        calls.append(call)
        spans.append((start, end))
        position = end
    if not calls:
        return text, []
    parts: list[str] = []
    last = 0
    for start, end in spans:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts).strip(), calls


def _skip_block(text: str, start: int) -> int:
    """Return the offset after the block that opens at ``start``.

    Counts nested ``<tool_call>`` markers, so a malformed block is skipped as
    a unit; an unterminated block extends to the end of the text.
    """
    depth = 0
    position = start
    while position <= len(text):
        opener = text.find(_OPEN, position)
        closer = text.find(_CLOSE, position)
        if closer == -1:
            return len(text)
        if opener != -1 and opener < closer:
            depth += 1
            position = opener + len(_OPEN)
            continue
        depth -= 1
        position = closer + len(_CLOSE)
        if depth == 0:
            return position
    return len(text)


def _parse_call(
    text: str, opening_end: int, decoder: json.JSONDecoder
) -> tuple[ToolCall, int] | None:
    """Parse the tool call that starts at ``opening_end``.

    Args:
        text: The whole answer of the model.
        opening_end: The offset just after the ``<tool_call>`` marker.
        decoder: The JSON decoder that rejects non-finite numbers and the
            non-standard constants.

    Returns:
        The tool call and the offset after the closing ``</tool_call>``, or
        ``None`` when the text does not hold a valid call. A closing marker
        inside a JSON string does not end the call — the JSON value decides
        where the body ends.
    """
    start = _skip_whitespace(text, opening_end)
    try:
        data, end = decoder.raw_decode(text, start)
    except ValueError:
        return None
    call = _to_tool_call(data)
    if call is None:
        return None
    closing = _skip_whitespace(text, end)
    if not text.startswith(_CLOSE, closing):
        return None
    return call, closing + len(_CLOSE)


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position] in _WHITESPACE:
        position += 1
    return position


def _has_unpaired_surrogate(value: object) -> bool:
    """Detect strings that cannot be encoded as UTF-8.

    The JSON decoder combines well-formed surrogate escape pairs, so a code
    unit in the surrogate range means an unpaired surrogate — the value
    cannot be serialized by a UTF-8 response.
    """
    if isinstance(value, str):
        return any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    if isinstance(value, dict):
        return any(
            _has_unpaired_surrogate(key) or _has_unpaired_surrogate(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_unpaired_surrogate(item) for item in value)
    return False


def _to_tool_call(data: object) -> ToolCall | None:
    """Validate the decoded JSON object of a tool call."""
    if not isinstance(data, dict):
        return None
    if _has_unpaired_surrogate(data):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = data.get("arguments", {})
    if isinstance(arguments, str):
        # Some models encode the arguments as a JSON string instead of an
        # object.
        try:
            arguments = _strict_decoder().decode(arguments)
        except ValueError:
            return None
        # The escaped string may decode to lone surrogates; the check on the
        # outer object does not see them.
        if _has_unpaired_surrogate(arguments):
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return ToolCall(function=ToolCallFunction(name=name, arguments=arguments))
