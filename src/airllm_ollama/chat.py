"""Rendering of a chat history into a single prompt.

Ollama renders the messages of ``/api/chat`` with the chat template of the
model. Base models do not have one, so the history is flattened into a plain
transcript instead.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction, ToolDefinition, ToolFunction

# The system message is merged into the first user turn with this separator —
# the same one the plain join of a model without a template uses.
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ChatTurn:
    """A single message of a chat history, with its tool calls.

    The tool calls of an assistant message and the name of the tool a
    ``tool`` role message answers are carried alongside the content, so the
    history can be rendered the way the model expects it.
    """

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None


# What the engine accepts as a message: a structured ``ChatTurn`` or a plain
# ``(role, content)`` pair, the format used before the messages carried tool
# calls. The pair is normalized at the engine boundary.
ChatTurnInput = ChatTurn | tuple[str, str]


def to_turn(message: ChatTurnInput) -> ChatTurn:
    """Normalize a message to a ``ChatTurn``.

    Args:
        message: A structured message or a legacy ``(role, content)`` pair.

    Returns:
        The message as a ``ChatTurn``.
    """
    if isinstance(message, ChatTurn):
        return message
    role, content = message
    return ChatTurn(role, content)


class ChatTemplate(Protocol):
    """The chat template of a tokenizer."""

    def __call__(self, messages: object, **kwargs: object) -> str:
        """Render the messages with the template."""
        ...


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt that is ready to be tokenized.

    A chat template renders the special tokens of the model itself (Llama 3
    starts its chat with ``<|begin_of_text|>``). Such a prompt is tokenized
    *without* the special tokens, so that the template stays the single source
    of truth for the format — the same way
    ``tokenizer.apply_chat_template(..., tokenize=True)`` does it.

    Every other prompt is tokenized the way the tokenizer does it by default,
    so that a base model still gets its beginning-of-sequence token.

    ``add_special_tokens`` has no default: every caller has to decide how the
    prompt is tokenized, so that a new call site cannot bring back the double
    BOS token by accident.
    """

    text: str
    add_special_tokens: bool


def _tools_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, object]]:
    """Render the tools the way the chat templates expect them."""
    payload: list[dict[str, object]] = []
    for tool in tools:
        function: dict[str, object] = {
            "name": tool.function.name,
            # Some templates concatenate the description unconditionally.
            "description": tool.function.description or "",
            "parameters": tool.function.parameters,
        }
        payload.append({"type": tool.type, "function": function})
    return payload


def _tools_line(tools: Sequence[ToolDefinition]) -> str:
    """Render the tools as a single line of the transcript."""
    payload = json.dumps(_tools_payload(tools), ensure_ascii=False)
    return f"Tools: {payload}"


def has_tool_material(messages: Sequence[ChatTurn]) -> bool:
    """Check whether the history carries tool calls or tool results."""
    return any(message.tool_calls or message.role == "tool" for message in messages)


def _template_supports_tool_calls(raw: object, tool_use_selected: bool) -> bool:
    """Tell whether the chat template that gets selected renders tool calls.

    A template that renders tool calls references ``tool_calls`` (Qwen,
    Hermes, ...). A template that does not — e.g. SmolLM3 renders only the
    content of an assistant message — silently drops the calls from the
    history; for those the calls are inlined into the content instead, in
    the same ``<tool_call>`` format the model was trained to emit.

    Args:
        raw: The ``chat_template`` attribute of the tokenizer.
        tool_use_selected: Whether the tool-aware variant of the template is
            selected (a dict of named templates is used by transformers when
            tools are offered — see ``apply_chat_template``).
    """
    if raw is None:
        # Unknown template source: assume it renders the tool calls.
        return True
    if isinstance(raw, dict):
        # A dict of named templates (e.g. Llama 3.1): transformers selects
        # ``tool_use`` when tools are passed, falls back to ``default`` and
        # only then to the first entry.
        if tool_use_selected and "tool_use" in raw:
            selected = raw["tool_use"]
        elif "default" in raw:
            selected = raw["default"]
        else:
            selected = next(iter(raw.values()), "")
        return "tool_calls" in str(selected)
    return "tool_calls" in str(raw)


def render_transcript(
    messages: Sequence[ChatTurn], tools: Sequence[ToolDefinition] | None = None
) -> str:
    """Flatten a chat history into a plain transcript.

    The transcript is used for models without a chat template::

        System: Be brief.

        User: Hello

        Assistant:

    Tool calls are rendered on the same line as the message, so the model
    still sees them::

        Assistant: Checking. [tool_calls: get_weather({"city": "Paris"})]
        Tool[get_weather]: 22°C

    When the request offers tools, they are listed on their own line in
    front of the history, so that a model without a template still learns
    which tools it may call.

    Args:
        messages: The chat history.
        tools: The tools offered to the model, if any.

    Returns:
        The rendered prompt.
    """
    lines = [_transcript_line(message) for message in messages]
    lines.append("Assistant:")
    if tools:
        return "\n\n".join([_tools_line(tools), *lines])
    return "\n\n".join(lines)


def _transcript_line(message: ChatTurn) -> str:
    """Render a single message of the transcript."""
    if message.role == "tool":
        if message.tool_name:
            return f"Tool[{message.tool_name}]: {message.content}"
        return f"Tool: {message.content}"
    line = f"{message.role.capitalize()}:"
    if message.content:
        line = f"{line} {message.content}"
    if message.tool_calls:
        calls = ", ".join(_format_tool_call(call) for call in message.tool_calls)
        line = f"{line} [tool_calls: {calls}]"
    return line


def _format_tool_call(call: ToolCall) -> str:
    """Render a tool call as ``name({arguments})`` for the transcript."""
    arguments = json.dumps(call.function.arguments, ensure_ascii=False, sort_keys=True)
    return f"{call.function.name}({arguments})"


def apply_chat_template(
    tokenizer: object,
    messages: Sequence[ChatTurn],
    tools: Sequence[ToolDefinition] | None = None,
) -> str | None:
    """Render the messages with the chat template of the model.

    A template that refuses the messages is asked a second time, with the
    system message merged into the first user turn. Templates without support
    for the system role (Gemma, older Mistral versions) raise, and the model
    would lose its own format completely — the merged messages keep it.

    The tools of the request are passed to the template as ``tools`` — the
    chat templates of the tool-calling models render them into the prompt.

    Args:
        tokenizer: The tokenizer of the model.
        messages: The chat history.
        tools: The tools offered to the model, if any.

    Returns:
        The rendered prompt, or ``None`` when the model has no chat template
        or when the template refuses the messages.
    """
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        return None
    offered = tools if tools else None
    render = cast("ChatTemplate", template)
    supports_tool_calls = _template_supports_tool_calls(
        getattr(tokenizer, "chat_template", None), offered is not None
    )
    for candidate in (messages, _merge_system_message(messages)):
        if candidate is None:
            continue
        # Null content only matters for assistant turns that carry tool
        # calls: try the empty string first (GLM and most templates render
        # it), fall back to null (DeepSeek-R1 needs it).
        needs_null = any(message.tool_calls and not message.content for message in candidate)
        for null_content in (False, True) if needs_null else (False,):
            if not _verified(render, candidate, offered, supports_tool_calls, null_content):
                continue
            rendered = _render_with_template(
                render, candidate, offered, supports_tool_calls, null_content
            )
            if rendered is not None:
                return rendered
    return None


def _squash(text: str) -> str:
    """Collapse all runs of whitespace, so formatting cannot hide a probe."""
    return " ".join(text.split())


def _mark_tool_messages(
    messages: Sequence[ChatTurn], seed: str
) -> tuple[list[ChatTurn], list[str]]:
    """Insert sentinel markers into the tool messages and call arguments.

    The markers are random per render and are unlikely to collide with any
    content the user supplied, so a template that drops any of them can be
    detected by looking for them in its output. Every tool message is
    marked, including ones with empty content — a template that ignores the
    tool role must not pass verification just because there was nothing to
    interpolate.
    """
    marked: list[ChatTurn] = []
    markers: list[str] = []
    for index, message in enumerate(messages):
        if message.role == "tool":
            marker = f"qx{seed}{index}q"
            marked.append(
                ChatTurn(
                    "tool", f"{marker} {message.content}", message.tool_calls, message.tool_name
                )
            )
            markers.append(marker)
            continue
        if message.tool_calls:
            calls = []
            for ordinal, call in enumerate(message.tool_calls):
                marker = f"qx{seed}{index}x{ordinal}q"
                arguments = {**call.function.arguments, f"qx{seed}": marker}
                calls.append(
                    ToolCall(
                        function=ToolCallFunction(
                            name=call.function.name,
                            arguments=arguments,
                        )
                    )
                )
                markers.append(marker)
            content = message.content
            if content.strip():
                marker = f"qx{seed}{index}t"
                content = f"{content} {marker}"
                markers.append(marker)
            marked.append(ChatTurn(message.role, content, tuple(calls), message.tool_name))
            continue
        marked.append(message)
    return marked, markers


def _verified(
    render: ChatTemplate,
    candidate: Sequence[ChatTurn],
    tools: Sequence[ToolDefinition] | None,
    supports_tool_calls: bool,
    null_content: bool,
) -> bool:
    """Check that the template really renders tools and tool exchanges.

    The candidate is rendered once more with a synthetic probe tool and with
    sentinel markers injected into the tool messages and call arguments.
    The sentinels are unambiguous markers that survive every tool argument
    serialization (JSON, dict literal, ``<parameter>`` blocks), so a
    template that drops the offered tools or a part of the tool history is
    detected here and falls back to the transcript.
    """
    if tools:
        probe = f"qxprobe{secrets.token_hex(4)}"
        probe_tools: list[ToolDefinition] | None = [
            *tools,
            ToolDefinition(
                function=ToolFunction(
                    name=probe,
                    description="Synthetic probe to detect templates that ignore tools.",
                    parameters={"type": "object", "properties": {}},
                )
            ),
        ]
    else:
        probe = None
        probe_tools = None
    seed = secrets.token_hex(6)
    marked, markers = _mark_tool_messages(candidate, seed)
    if probe_tools is None and not markers:
        return True
    rendered = _render_with_template(render, marked, probe_tools, supports_tool_calls, null_content)
    if rendered is None:
        return False
    squashed = _squash(rendered)
    if probe is not None and probe not in squashed:
        return False
    return all(marker in squashed for marker in markers)


def _render_with_string_arguments(
    template: ChatTemplate, payload: list[dict[str, object]], kwargs: dict[str, object]
) -> str | None:
    """Retry a rendering with the tool-call arguments as JSON strings."""
    converted: list[dict[str, object]] = []
    for message in payload:
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            converted_calls: list[dict[str, object]] = []
            for raw_call in cast("list[dict[str, Any]]", calls):
                function = cast("dict[str, Any]", raw_call.get("function", {}))
                converted_calls.append(
                    {
                        **raw_call,
                        "function": {
                            **function,
                            "arguments": json.dumps(function.get("arguments", {})),
                        },
                    }
                )
            message = {**message, "tool_calls": converted_calls}
        converted.append(message)
    try:
        rendered: object = template(converted, **kwargs)
    except Exception:
        return None
    if not isinstance(rendered, str) or not rendered.strip():
        return None
    return rendered


def _render_with_template(
    template: ChatTemplate,
    messages: Sequence[ChatTurn],
    tools: Sequence[ToolDefinition] | None = None,
    supports_tool_calls: bool = True,
    null_content: bool = False,
) -> str | None:
    """Render the messages with the template.

    Args:
        template: The chat template of the model.
        messages: The chat history.
        tools: The tools offered to the model, if any.
        supports_tool_calls: Whether the template renders the tool calls of
            an assistant message. When it does not, the calls are inlined
            into the content, the way the model emits them.
        null_content: Whether a tool-calling assistant message with empty
            content is passed ``null`` (the DeepSeek-R1 convention) instead
            of the empty string (which GLM-style templates interpolate).

    Returns:
        The rendered prompt, or ``None`` when the template refuses the
        messages.
    """
    payload = [_message_payload(message, supports_tool_calls, null_content) for message in messages]
    _assign_call_ids(payload)
    kwargs: dict[str, object] = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = _tools_payload(tools)
    tool_calls = [
        call
        for message in payload
        for call in cast("list[dict[str, Any]]", message.get("tool_calls") or [])
    ]
    has_dict_arguments = any(
        isinstance(cast("dict[str, Any]", call.get("function", {})).get("arguments"), dict)
        for call in tool_calls
    )
    rendered: object
    try:
        rendered = template(payload, **kwargs)
    except Exception:
        if not has_dict_arguments:
            # The template may reject any message (unknown role, missing tool
            # support, ...); the caller decides what to do instead.
            return None
        # Some templates (DeepSeek-R1) concatenate the arguments as a string
        # and fail on a dict — render those with the arguments encoded as a
        # JSON string, the way the model was trained to see them.
        rendered = _render_with_string_arguments(template, payload, kwargs)
        if rendered is None:
            return None
    if not isinstance(rendered, str) or not rendered.strip():
        return None
    return rendered


def _assign_call_ids(payload: list[dict[str, object]]) -> None:
    """Assign synthetic call ids to the tool calls of an assistant turn.

    The Mistral family of templates requires every tool call to carry a
    nine-character alphanumeric ``id`` and every tool result a matching
    ``tool_call_id``. Other templates ignore the keys. The ids are
    deterministic for a given history.
    """
    counter = 0
    pending: list[str] = []
    for message in payload:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for call in cast("list[dict[str, Any]]", tool_calls):
                counter += 1
                call_id = f"call{counter:05d}"[-9:]
                call["id"] = call_id
                pending.append(call_id)
        if message.get("role") == "tool":
            message["tool_call_id"] = pending.pop(0) if pending else "call00000"


def _message_payload(
    message: ChatTurn, supports_tool_calls: bool = True, null_content: bool = False
) -> dict[str, object]:
    """Render a message the way the chat templates of the models expect it.

    The tool calls of an assistant message are passed in the Hugging Face
    shape — ``{"type": "function", "function": {"name": ..., "arguments":
    {...}}}`` with the arguments as a JSON object. A tool message names the
    tool it answers with both ``tool_name`` (Ollama's name for it) and
    ``name`` (what most templates read); templates ignore the key they do
    not know.

    A template that does not render the tool calls gets them inlined into
    the content instead — the ``<tool_call>`` block the model was trained
    to emit — so the history is not silently emptied.
    """
    payload: dict[str, object] = {"role": message.role, "content": message.content}
    if not message.tool_calls:
        if message.tool_name is not None:
            payload["tool_name"] = message.tool_name
            payload["name"] = message.tool_name
        return payload
    if not supports_tool_calls:
        payload["content"] = _inline_tool_calls(message.content, message.tool_calls)
        return payload
    if not message.content and message.tool_calls and null_content:
        # The DeepSeek-R1 family renders the tool calls of an assistant
        # message only when its content is null — an empty string would
        # render an empty turn and drop the call. Templates that render
        # empty strings (GLM) get the empty string instead.
        payload["content"] = None
    payload["tool_calls"] = [
        {
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in message.tool_calls
    ]
    if message.tool_name is not None:
        payload["tool_name"] = message.tool_name
        payload["name"] = message.tool_name
    return payload


def _inline_tool_calls(content: str, calls: Sequence[ToolCall]) -> str:
    """Inline the tool calls of a message as ``<tool_call>`` blocks."""
    blocks = []
    for call in calls:
        body = json.dumps(
            {"name": call.function.name, "arguments": call.function.arguments},
            ensure_ascii=False,
        )
        blocks.append(f"<tool_call>\n{body}\n</tool_call>")
    inlined = "\n".join(blocks)
    return f"{content}\n{inlined}" if content else inlined


def _merge_system_message(messages: Sequence[ChatTurn]) -> list[ChatTurn] | None:
    """Merge the system messages into the first user turn.

    A template without support for the system role refuses the messages. The
    system messages become part of the first user turn instead, so that the
    model still gets its own format.

    Args:
        messages: The chat history.

    Returns:
        The messages with the system messages merged, or ``None`` when there is
        no system message or no user turn to merge them into.
    """
    systems = [message.content for message in messages if message.role == "system"]
    if not systems:
        return None
    system = SEPARATOR.join(systems)
    merged: list[ChatTurn] = []
    merged_system = False
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user" and not merged_system:
            # An empty system message is dropped instead of adding newlines.
            prefix = f"{system}{SEPARATOR}" if system.strip() else ""
            merged.append(
                ChatTurn(
                    "user", f"{prefix}{message.content}", message.tool_calls, message.tool_name
                )
            )
            merged_system = True
            continue
        merged.append(message)
    if not merged_system:
        return None
    return merged


def render_chat(
    tokenizer: object,
    messages: Sequence[ChatTurn],
    tools: Sequence[ToolDefinition] | None = None,
) -> RenderedPrompt:
    """Render a chat history with the chat template of the model.

    Falls back to :func:`render_transcript` when the tokenizer has no chat
    template or when the template refuses the messages.

    Args:
        tokenizer: The tokenizer of the model.
        messages: The chat history.
        tools: The tools offered to the model, if any.

    Returns:
        The rendered prompt and how it has to be tokenized.
    """
    rendered = apply_chat_template(tokenizer, messages, tools)
    if rendered is None:
        return RenderedPrompt(render_transcript(messages, tools), add_special_tokens=True)
    # The template rendered the special tokens of the model, so the tokenizer
    # must not add them a second time.
    return RenderedPrompt(rendered, add_special_tokens=False)


def render_completion(
    tokenizer: object, prompt: str, system: str | None, raw: bool = False
) -> RenderedPrompt:
    """Render a completion prompt together with its system message.

    Ollama applies the chat template of the model to **every** prompt, so the
    prompt is rendered as a chat here as well — the same way ``/api/chat``
    renders the history. A request with ``raw`` set is sent unchanged, without
    the template.

    A model without a chat template gets the prompt with the system message
    joined in front of it, the way Ollama does it.

    Args:
        tokenizer: The tokenizer of the model.
        prompt: The prompt of the user.
        system: The system message, if any.
        raw: Whether the prompt is sent unchanged, without the template.

    Returns:
        The rendered prompt and how it has to be tokenized.
    """
    if raw:
        # `raw` asks for the prompt without the template, like Ollama does.
        # A client that rendered the template itself sends the BOS token with
        # the prompt (#44).
        return RenderedPrompt(prompt, add_special_tokens=True)
    messages: list[ChatTurn] = [ChatTurn("user", prompt)]
    if system:
        messages.insert(0, ChatTurn("system", system))
    rendered = apply_chat_template(tokenizer, messages)
    if rendered is None:
        # No template: the system message is joined with the prompt, the way
        # Ollama does it for a model without a template.
        if system:
            return RenderedPrompt(f"{system}{SEPARATOR}{prompt}", add_special_tokens=True)
        return RenderedPrompt(prompt, add_special_tokens=True)
    return RenderedPrompt(rendered, add_special_tokens=False)
