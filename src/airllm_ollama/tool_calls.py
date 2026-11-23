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
client sees the same text the model produced. Markdown is not a call: code
regions (fenced and indented code blocks, inline code spans) are skipped, so
an example the model writes is never mistaken for a call.
"""

from __future__ import annotations

import json
import math
import re

from markdown_it import MarkdownIt

from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_WHITESPACE = " \t\r\n"
_MARKDOWN = MarkdownIt("commonmark")


def _code_regions(text: str) -> list[tuple[int, int]]:
    """Return the spans of the code regions of the text.

    Fenced and indented code blocks and inline code spans are examples the
    model wrote for the user — never calls. Markdown decides what counts:
    blockquotes, headings, list items and nested fences all follow the same
    rules as a rendered document.
    """
    try:
        tokens = _MARKDOWN.parse(text)
    except Exception:
        # An unparseable document: nothing is treated as code.
        return []
    line_starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            line_starts.append(index + 1)
    line_starts.append(len(text))
    regions: list[tuple[int, int]] = []
    for token in tokens:
        if token.map is not None and token.type in ("fence", "code_block", "html_block"):
            regions.append((line_starts[token.map[0]], line_starts[token.map[1]]))
            continue
        if token.type != "inline" or token.map is None or not token.children:
            continue
        position = line_starts[token.map[0]]
        for child in token.children:
            if not child.content:
                continue
            # The parser normalizes the whitespace inside the content (a
            # newline inside an inline code span becomes a space), so the
            # source is matched with flexible whitespace.
            pattern = re.compile(re.escape(child.content).replace(r"\ ", r"\s+"))
            match = pattern.search(text, position)
            if match is None:
                # Cannot locate the token in the source; stop tracking this
                # inline block rather than guess positions.
                position = line_starts[min(token.map[1], len(line_starts) - 1)]
                break
            start, end = match.start(), match.end()
            if child.type == "code_inline":
                regions.append((start, end))
            position = end
    return sorted(regions)


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
    regions = _code_regions(text)
    region_index = 0
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        while region_index < len(regions) and regions[region_index][1] <= position:
            region_index += 1
        if region_index < len(regions) and regions[region_index][0] <= position:
            # A code region (fenced or indented block, code span): whatever it
            # contains is an example, not a call.
            position = regions[region_index][1]
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
                # the matching closing marker (counting nested blocks) keeps a
                # marker inside the malformed block from being picked up as a
                # call of its own, no matter how deep the nesting.
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
