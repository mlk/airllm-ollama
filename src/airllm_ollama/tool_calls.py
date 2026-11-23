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
from collections.abc import Iterator

from markdown_it import MarkdownIt

from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction

_THINK_TAG = re.compile(r"<(/?)think\b[^>]*>", re.IGNORECASE)
_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_WHITESPACE = " \t\r\n"
_MARKDOWN = MarkdownIt("commonmark")

# A tag scanner that never backtracks: every position of the text is
# consumed at most once, no matter how broken the markup is.

# Elements whose content is raw text, up to the matching close tag. An
# unterminated one swallows the rest of the text, like in HTML.
_RAW_TEXT_ELEMENTS = frozenset(
    {
        "script",
        "style",
        "pre",
        "xmp",
        "iframe",
        "noembed",
        "noframes",
        "noscript",
        "plaintext",
        "title",
        "textarea",
    }
)

# Void elements: they never open a region.
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def _is_escaped(text: str, position: int) -> bool:
    """Return whether the character at ``position`` is backslash-escaped."""
    slashes = 0
    index = position - 1
    while index >= 0 and text[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def _iter_tags(text: str) -> Iterator[tuple[int, int, bool, str]]:
    r"""Yield ``(start, end, closing, name)`` for every tag in the text.

    A single pass: every position is consumed at most once, so even
    pathological input (thousands of unterminated ``<a``) stays linear.
    The first ``>`` outside a quoted attribute value terminates the tag;
    a tag whose quoted value is never closed runs to the end of the text,
    the way an HTML parser treats it. Text that cannot be a tag is skipped
    one character at a time. Tag names never contain ``_``, so the
    ``<tool_call>`` and ``<think>`` markers are not tags.
    """
    length = len(text)
    position = 0
    while True:
        start = text.find("<", position)
        if start == -1:
            return
        index = start + 1
        closing = index < length and text[index] == "/"
        if closing:
            index += 1
        name_start = index
        while index < length and (text[index].isalnum() or text[index] == "-"):
            index += 1
        if index == name_start:
            position = start + 1
            continue
        name = text[name_start:index].lower()
        if index < length and text[index] not in " \t\r\n/>":
            # A name followed by anything else (e.g. "<tool_call>" parsed as
            # a tag "tool" followed by attribute "_call") is not a tag.
            position = start + 1
            continue
        pos = index
        end = length
        while pos < length:
            char = text[pos]
            if char == ">":
                end = pos + 1
                break
            if char in " \t\r\n":
                pos += 1
                continue
            if char == "/" and pos + 1 < length and text[pos + 1] == ">":
                end = pos + 2
                break
            if char in "\"'":
                closed = text.find(char, pos + 1)
                if closed == -1:
                    # Unterminated attribute value: the tag runs to the
                    # end of the text, the way an HTML parser treats it.
                    pos = length
                    break
                pos = closed + 1
                continue
            # Anything else is part of the tag (an attribute name, an
            # equals sign, or a bare value): consume until the next
            # delimiter. One character at a time, never backtracking.
            while pos < length and text[pos] not in " \t\r\n>/\"'":
                pos += 1
        yield start, end, closing, name
        position = end


_NULL = "\x00"


def _mask_think(text: str) -> str:
    """Replace the reasoning delimiters with NUL bytes of the same length.

    CommonMark classifies a line starting with ``<think>`` as an HTML block,
    which would swallow the reasoning block into a fence region and hide
    the very delimiters this module looks for. Masking keeps the offsets of
    the text unchanged while making the delimiters invisible to the Markdown
    parser.
    """
    return _THINK_TAG.sub(lambda match: _NULL * len(match.group(0)), text)


def _comment_regions(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of the HTML comments, honouring the blocked regions."""
    regions: list[tuple[int, int]] = []
    for match in re.finditer(r"<!--", text):
        if _is_escaped(text, match.start()):
            continue
        if _is_blocked(blocked, match.start(), match.start() + 4):
            continue
        end = text.find("-->", match.end())
        regions.append((match.start(), len(text) if end == -1 else end + 3))
    return regions


def _html_regions(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of the HTML regions of the text.

    Raw-text elements (``<script>`` and the like) and nested elements all
    swallow everything inside them — a call inside any of them is markup,
    never an action. An unterminated element swallows the rest of the text,
    like its HTML counterpart does. A tag name may not contain ``<`` or
    ``>``; comments need not match any tag grammar.
    """
    regions: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    raw_text: tuple[str, int] | None = None
    for start, end, closing, name in _iter_tags(text):
        if _is_blocked(blocked, start, end):
            continue
        if _is_escaped(text, start):
            continue
        if name == "think":
            # The reasoning block is handled separately by _think_regions;
            # its delimiters are not HTML tags and never pair here.
            continue
        if raw_text is not None:
            if closing and name == raw_text[0]:
                regions.append((raw_text[1], end))
                raw_text = None
            continue
        if closing:
            # Pop everything up to and including the matching open tag; an
            # unmatched close is ignored.
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == name:
                    start_of_element = stack[index][1]
                    del stack[index:]
                    regions.append((start_of_element, end))
                    break
            continue
        if name in _RAW_TEXT_ELEMENTS:
            raw_text = (name, start)
            continue
        if name in _VOID_ELEMENTS:
            continue
        stack.append((name, start))
    if raw_text is not None:
        regions.append((raw_text[1], len(text)))
    for _name, start in stack:
        regions.append((start, len(text)))
    return regions


def _tag_regions(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of every tag-like ``<...>`` in the text.

    A call whose whole span lies inside a tag is part of an attribute
    value — markup, never an action. The tool-call markers themselves are
    not tags and are not reported. Tags that start inside a blocked
    region, or are backslash-escaped, are not tags either.
    """
    regions: list[tuple[int, int]] = []
    for start, end, _closing, name in _iter_tags(text):
        if _is_blocked(blocked, start, end):
            continue
        if _is_escaped(text, start):
            continue
        if name == "think":
            continue
        regions.append((start, end))
    return regions


def _raw_text_regions(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of the raw-text elements (script, style, ...).

    Their contents are opaque, like in HTML: a reasoning delimiter or a
    comment inside them is text, not markup. An unterminated raw-text
    element swallows the rest of the text.
    """
    regions: list[tuple[int, int]] = []
    raw_text: tuple[str, int] | None = None
    for start, end, closing, name in _iter_tags(text):
        if _is_blocked(blocked, start, end):
            continue
        if _is_escaped(text, start):
            continue
        if name not in _RAW_TEXT_ELEMENTS:
            continue
        if closing:
            if raw_text is not None and name == raw_text[0]:
                regions.append((raw_text[1], end))
                raw_text = None
            continue
        if raw_text is None:
            raw_text = (name, start)
    if raw_text is not None:
        regions.append((raw_text[1], len(text)))
    return regions


def _link_destination_spans(text: str) -> list[tuple[int, int]]:
    """Return the spans of inline-link destinations and reference labels.

    Backticks inside a link destination are part of the URL, not code
    delimiters, so the inline-code scan must not pair them. Handles nested
    parentheses in destinations; quotes and escapes are honoured. Reference
    definitions (``[label]: destination``) are covered as well.
    """
    regions: list[tuple[int, int]] = []
    for match in re.finditer(r"\]\(", text):
        start = match.start() + 2
        depth = 1
        pos = start
        in_quotes = False
        while pos < len(text):
            char = text[pos]
            if char == "\\":
                pos += 2
                continue
            if in_quotes:
                if char == '"':
                    in_quotes = False
                pos += 1
                continue
            if char == '"':
                in_quotes = True
                pos += 1
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        regions.append((start, pos))
    for match in re.finditer(r"^\[[^\]]*\]:\s*(\S+)", text, re.MULTILINE):
        regions.append((match.start(1), match.end(1)))
    return regions


def _line_starts(text: str) -> list[int]:
    """Return the offset at which each line starts (all line-ending styles)."""
    starts = [0]
    for match in re.finditer(r"\r\n|[\r\n]", text):
        starts.append(match.end())
    starts.append(len(text))
    return starts


def _intersects(regions: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < region_end and end > region_start for region_start, region_end in regions)


def _is_blocked(blocked: list[tuple[int, int]], start: int, end: int) -> bool:
    """Return whether a span overlaps a blocked region."""
    return _intersects(blocked, start, end)


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
        if _is_escaped(text, match.start()):
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
    """Return the spans of the block-level segments of the document."""
    try:
        tokens = _MARKDOWN.parse(text)
    except Exception:
        return [(0, len(text))]
    line_starts = _line_starts(text)
    segments: list[tuple[int, int]] = []
    for token in tokens:
        if token.map is None:
            continue
        if token.type == "inline":
            segments.append((line_starts[token.map[0]], line_starts[token.map[1]]))
    if not segments:
        segments.append((0, len(text)))
    return segments


def _markdown_blocks(text: str) -> list[tuple[int, int]]:
    """Return the fence/code regions the Markdown parser reports."""
    try:
        tokens = _MARKDOWN.parse(text)
    except Exception:
        return [(0, len(text))]
    line_starts = _line_starts(text)
    regions: list[tuple[int, int]] = []
    for token in tokens:
        if token.map is None:
            continue
        if token.type in ("fence", "code_block", "html_block"):
            regions.append((line_starts[token.map[0]], line_starts[token.map[1]]))
    return regions


def _inline_spans(
    text: str,
    blocked: list[tuple[int, int]],
    segments: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the spans of the inline code spans.

    CommonMark: a span opens with a run of backticks and closes at a run of
    exactly the same length, anywhere inside one block (a span may span
    line breaks, but never a block boundary). Spans do not open inside a
    blocked region.
    """
    regions: list[tuple[int, int]] = []
    blocked_index = 0
    for segment_start, segment_end in segments:
        position = segment_start
        while position < segment_end:
            while blocked_index < len(blocked) and blocked[blocked_index][1] <= position:
                blocked_index += 1
            if blocked_index < len(blocked) and blocked[blocked_index][0] <= position:
                position = min(blocked[blocked_index][1], segment_end)
                continue
            char = text[position]
            if char == "\\":
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


def _call_spans(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the spans of the well-formed calls outside the blocked regions."""
    spans: list[tuple[int, int]] = []
    decoder = _strict_decoder()
    blocked_index = 0
    position = 0
    while position < len(text):
        while blocked_index < len(blocked) and blocked[blocked_index][1] <= position:
            blocked_index += 1
        if blocked_index < len(blocked) and blocked[blocked_index][0] <= position:
            position = blocked[blocked_index][1]
            continue
        if text.startswith(_OPEN, position) and not _is_escaped(text, position):
            found = _parse_call(text, position + len(_OPEN), decoder)
            if found is not None:
                _, end = found
                spans.append((position, end))
                position = end
                continue
            position = _skip_block(text, position)
            continue
        position += 1
    return spans


def _code_regions(text: str) -> list[tuple[int, int]]:
    """Return the spans of the regions of the text that are not calls.

    Fenced and indented code blocks, inline code spans, HTML blocks, and
    reasoning blocks (``<think>...</think>``) are not calls — an example the
    model writes for the user is never executed. Calls are scanned first
    and are opaque to every later pass: their contents may contain anything,
    including markup-looking text, so nothing inside a call is ever a
    region of its own. Regions that end up covering a call make the call
    inert.
    """
    masked = _mask_think(text)
    fences = _markdown_blocks(masked)
    # Tag spans are needed early: a call inside a tag's attribute value is
    # markup, never an action.
    tags = _tag_regions(text, fences)
    calls = _call_spans(text, _merge_regions(fences + tags))
    segments = _segments(masked)
    links = _link_destination_spans(text)
    # Inline code spans come before comments and HTML blocks: a comment
    # opener inside a code span is an example, not a comment. Backticks
    # inside link destinations are part of the URL, not code delimiters.
    inline = _inline_spans(text, _merge_regions(fences + tags + calls + links), segments)
    comments = _comment_regions(text, _merge_regions(fences + tags + calls + inline))
    raw = _raw_text_regions(text, _merge_regions(fences + comments + calls + inline))
    think = _think_regions(text, _merge_regions(fences + tags + calls + inline + comments + raw))
    blocks = _merge_regions(fences + comments + think)
    html = _html_regions(text, _merge_regions(blocks + calls + inline))
    # Tag spans stay in the blocking set: a call inside a tag (for instance
    # inside a void element's attribute) never executes.
    return _merge_regions(blocks + inline + html + tags)


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
        if text.startswith(_OPEN, position) and not _is_escaped(text, position):
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
        if text.startswith(_OPEN, position) and not _is_escaped(text, position):
            depth += 1
            position += len(_OPEN)
            continue
        if text.startswith(_CLOSE, position) and not _is_escaped(text, position):
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
