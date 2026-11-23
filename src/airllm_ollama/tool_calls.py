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
import re
from typing import Any

from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction

# The block a tool-calling model wraps the call in. The body may span
# several lines and may nest braces (the arguments are an object), so the
# pattern anchors on the markers instead of the braces.
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split the text of an answer into its plain text and its tool calls.

    Args:
        text: The decoded answer of the model.

    Returns:
        The text with the valid tool call blocks removed (stripped) and the
        tool calls that were found, in the order they appear. When the text
        holds no valid tool call, it is returned unchanged.
    """
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        call = _parse_call(match.group("body"))
        if call is None:
            continue
        calls.append(call)
        spans.append(match.span())
    if not calls:
        return text, []
    parts: list[str] = []
    last = 0
    for start, end in spans:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts).strip(), calls


def _parse_call(body: str) -> ToolCall | None:
    """Parse the body of a tool call block.

    Args:
        body: The text between ``<tool_call>`` and ``</tool_call>``.

    Returns:
        The tool call, or ``None`` when the body is not a JSON object with a
        name and a JSON object of arguments.
    """
    try:
        data: Any = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = data.get("arguments", {})
    if isinstance(arguments, str):
        # Some models encode the arguments as a JSON string instead of an
        # object.
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return ToolCall(function=ToolCallFunction(name=name, arguments=arguments))
