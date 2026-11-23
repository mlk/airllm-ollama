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
client sees the same text the model produced. Markdown is not a call: fenced
code blocks and inline code spans are skipped, so an example the model writes
is never mistaken for a call.
"""

from __future__ import annotations

import json
import math

from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_WHITESPACE = " \t\r\n"
_FENCES = ("```", "~~~")


def _fence_run(text: str, start: int) -> int:
    """Return the length of the run of the fence character at ``start``."""
    char = text[start]
    return len(text[start:]) - len(text[start:].lstrip(char))


def _skip_fence(text: str, start: int) -> int:
    """Return the offset after the fenced code block that opens at ``start``.

    The block ends at the first line that holds only the fence character,
    at least as many as in the opening fence — or at the end of the text
    when the fence is not closed.
    """
    char = text[start]
    run = _fence_run(text, start)
    first_newline = text.find("\n", start)
    if first_newline == -1:
        return len(text)
    position = first_newline + 1
    while True:
        eol = text.find("\n", position)
        line_end = len(text) if eol == -1 else eol
        stripped = text[position:line_end].strip(" \t")
        if stripped and set(stripped) == {char} and len(stripped) >= run:
            return len(text) if eol == -1 else eol + 1
        if eol == -1:
            return len(text)
        position = eol + 1


def _find_backtick_run(text: str, run: int, start: int) -> int:
    """Return the offset of a run of exactly ``run`` backticks, or ``-1``."""
    position = start
    while True:
        found = text.find("`" * run, position)
        if found == -1:
            return -1
        before = text[found - 1] if found > 0 else ""
        after = text[found + run] if found + run < len(text) else ""
        if before != "`" and after != "`":
            return found
        position = found + run


def _skip_inline_code(text: str, start: int) -> int:
    """Return the offset after the inline code span that opens at ``start``.

    A span closes at a run of exactly as many backticks as opened it —
    anywhere, even on a later line (inline code spans may contain line
    breaks). An unmatched run is skipped over and scanning continues.
    """
    run = _fence_run(text, start)
    closing = _find_backtick_run(text, run, start + run)
    if closing == -1:
        return start + run
    return closing + run


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
    while position < len(text):
        if text.startswith(_FENCES, position):
            # A fenced code block: an example in it is not a call.
            position = _skip_fence(text, position)
            continue
        char = text[position]
        if char == "`":
            # An inline code span: an example in it is not a call either.
            position = _skip_inline_code(text, position)
            continue
        if text.startswith(_OPEN, position):
            found = _parse_call(text, position + len(_OPEN), decoder)
            if found is not None:
                call, end = found
                calls.append(call)
                spans.append((position, end))
                position = end
            else:
                # Not a valid call: the whole block is dead text. Skipping to
                # the matching closing marker (counting nested blocks) keeps
                # a marker inside the malformed block from being picked up as
                # a call of its own, no matter how deep the nesting.
                position = _skip_block(text, position)
            continue
        position += 1
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
    a unit. Markers inside JSON strings (with backslash escapes) do not
    count. An unterminated block extends to the end of the text.
    """
    depth = 0
    position = start
    in_string = False
    while position < len(text):
        char = text[position]
        if in_string:
            if char == "\\":
                position += 2
                continue
            if char == '"':
                in_string = False
            position += 1
            continue
        if char == '"':
            in_string = True
            position += 1
            continue
        if text.startswith(_OPEN, position):
            depth += 1
            position += len(_OPEN)
            continue
        if text.startswith(_CLOSE, position):
            depth -= 1
            position += len(_CLOSE)
            if depth == 0:
                return position
            continue
        position += 1
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
    except (ValueError, RecursionError):
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
    cannot be serialized by a UTF-8 response. The walk is iterative, so an
    arbitrarily deep nesting does not overflow the stack.
    """
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            if any(0xD800 <= ord(char) <= 0xDFFF for char in current):
                return True
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
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
        except (ValueError, RecursionError):
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
