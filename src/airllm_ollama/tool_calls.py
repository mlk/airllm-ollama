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
regions (fenced and indented code blocks, inline code spans), HTML blocks
and the model's reasoning (``<think>...</think>``) are skipped, so an
example the model writes is never mistaken for a call.
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

# Inline HTML that is all example text: comments and raw-text elements. A
# call inside any of these is an example, never an action.
_HTML_SPAN = re.compile(
    r"<!--.*?-->|<(script|style|pre|code|kbd|samp|xmp|div|section|article|table|ul|ol|li|p|blockquote|form|details|summary|svg|math)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

_THINK_TAG = re.compile(r"<(/?)think\b[^>]*>", re.IGNORECASE)


def _line_starts(text: str) -> list[int]:
    """Return the offset at which each line starts (all line-ending styles)."""
    starts = [0]
    for match in re.finditer(r"\r\n|[\r\n]", text):
        starts.append(match.end())
    starts.append(len(text))
    return starts


def _markdown_regions(text: str) -> list[tuple[int, int]]:
    """Return the code and HTML block spans of the text, per CommonMark."""
    try:
        tokens = _MARKDOWN.parse(text)
    except Exception:
        # An unparseable document: treat it as opaque, nothing is a call.
        return [(0, len(text))]
    line_starts = _line_starts(text)
    regions: list[tuple[int, int]] = []
    for token in tokens:
        if token.map is not None and token.type in ("fence", "code_block"):
            regions.append((line_starts[token.map[0]], line_starts[token.map[1]]))
    regions.extend(match.span() for match in _HTML_SPAN.finditer(text))
    return regions


def _intersects(regions: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < region_end and end > region_start for region_start, region_end in regions)


def _think_regions(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of the reasoning blocks (``<think>...</think>``).

    The reasoning of the model is not its answer: nothing inside it is a
    call. Tags inside code regions are literal text, not delimiters. An
    unterminated block swallows the rest of the text (fail closed).
    """
    regions: list[tuple[int, int]] = []
    start: int | None = None
    depth = 0
    for match in _THINK_TAG.finditer(text):
        if _intersects(blocked, match.start(), match.end()):
            continue
        closing = match.group(1) == "/"
        if not closing:
            if depth == 0:
                start = match.start()
            depth += 1
            continue
        if depth == 0:
            continue
        depth -= 1
        if depth == 0 and start is not None:
            regions.append((start, match.end()))
    if depth > 0 and start is not None:
        regions.append((start, len(text)))
    return regions


def _merge_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _run_length(text: str, start: int, char: str) -> int:
    end = start
    while end < len(text) and text[end] == char:
        end += 1
    return end - start


def _find_backtick_run(text: str, run: int, start: int, end: int) -> int:
    """Return the offset of a run of exactly ``run`` backticks, or ``-1``."""
    position = start
    needle = "`" * run
    while True:
        found = text.find(needle, position, end)
        if found == -1:
            return -1
        before = text[found - 1] if found > 0 else ""
        after = text[found + run] if found + run < len(text) else ""
        if before != "`" and after != "`":
            return found
        position = found + run


def _segments(text: str) -> list[tuple[int, int]]:
    """Split the text into paragraphs (delimited by blank lines).

    Code spans cannot span a blank line, so each segment is scanned on its
    own — a stray backtick in one paragraph can never swallow the markup of
    the next one, as CommonMark treats it as a literal backtick.
    """
    segments: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n[ \t]*(?:\n|$)", text):
        segments.append((start, match.start() + 1))
        start = match.end()
    if start < len(text):
        segments.append((start, len(text)))
    return segments


def _inline_code_regions(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of the inline code spans.

    CommonMark: a span opens with a run of backticks and closes at a run of
    exactly the same length, anywhere in the same paragraph (a span may span
    line breaks, but never a paragraph boundary). Spans do not open inside
    a block region.
    """
    regions: list[tuple[int, int]] = []
    blocked_index = 0
    for segment_start, segment_end in _segments(text):
        position = segment_start
        while position < segment_end:
            while blocked_index < len(blocked) and blocked[blocked_index][1] <= position:
                blocked_index += 1
            if blocked_index < len(blocked) and blocked[blocked_index][0] <= position:
                position = min(blocked[blocked_index][1], segment_end)
                continue
            char = text[position]
            if char == "\\":
                # An escaped character is never the start of a code span.
                position += 2
                continue
            if char == "`":
                run = _run_length(text, position, "`")
                closing = _find_backtick_run(text, run, position + run, segment_end)
                if closing != -1:
                    regions.append((position, closing + run))
                    position = closing + run
                    continue
                position += run
                continue
            position += 1
    return regions


def _code_regions(text: str) -> list[tuple[int, int]]:
    """Return the spans of the code regions of the text.

    Fenced and indented code blocks, HTML blocks, reasoning blocks and
    inline code spans are not calls — an example the model writes for the
    user is never executed. Markdown decides what counts as a code region:
    blockquotes, headings, list items and nested fences all follow the same
    rules as a rendered document.
    """
    blocks = _merge_regions(_markdown_regions(text))
    inline = _inline_code_regions(text, blocks)
    blocked = _merge_regions(blocks + inline)
    thinks = _think_regions(text, blocked)
    return _merge_regions(blocked + thinks)


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
