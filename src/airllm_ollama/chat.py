"""Rendering of a chat history into a single prompt.

Ollama renders the messages of ``/api/chat`` with the chat template of the
model. Base models do not have one, so the history is flattened into a plain
transcript instead.
"""

from __future__ import annotations

import json
import math
import re
import secrets
from collections.abc import Mapping, Sequence
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
    system_fragment = SEPARATOR.join(
        message.content for message in messages if message.role == "system"
    )
    for index, candidate in enumerate((messages, _merge_system_message(messages))):
        if candidate is None:
            continue
        # The merged candidate has no system message left, so the system
        # probe cannot guard it; require the merged content to survive
        # instead, so a template that keeps only part of the first user
        # turn does not pass verification.
        required = system_fragment.strip() if index == 1 and system_fragment.strip() else None
        # Null content only matters for assistant turns that carry tool
        # calls: try the empty string first (GLM and most templates render
        # it), fall back to null (DeepSeek-R1 needs it).
        needs_null = any(message.tool_calls and not message.content for message in candidate)
        for null_content in (False, True) if needs_null else (False,):
            if not _verified(
                render, candidate, offered, supports_tool_calls, null_content, required
            ):
                continue
            rendered = _render_with_template(
                render, candidate, offered, supports_tool_calls, null_content
            )
            if rendered is not None:
                return rendered
    return None


_JSON_SURROGATE_PAIR = re.compile(r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})")
_JSON_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}|\\[\"\\/bfnrt]|\\\\")


def _unescape_json(text: str) -> str:
    r"""Decode JSON string escapes in rendered text.

    Templates serialize non-ASCII text (and some punctuation) with JSON
    escapes, so the rendered output may spell ``é`` as ``\\u00e9``. The
    probes compare against the decoded form.
    """
    text = _JSON_SURROGATE_PAIR.sub(
        lambda m: chr(
            0x10000 + ((int(m.group(1), 16) - 0xD800) << 10) + (int(m.group(2), 16) - 0xDC00)
        ),
        text,
    )

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("\\u"):
            return chr(int(token[2:], 16))
        return {
            '\\"': '"',
            "\\/": "/",
            "\\b": "\b",
            "\\f": "\f",
            "\\n": "\n",
            "\\r": "\r",
            "\\t": "\t",
            "\\\\": "\\",
        }[token]

    return _JSON_ESCAPE.sub(replace, text)


def _squash(text: str) -> str:
    """Collapse all runs of whitespace, so formatting cannot hide a probe."""
    return " ".join(text.split())


def _mark_tool_messages(
    messages: Sequence[ChatTurn],
    seed: str,
    rename: (
        Mapping[str, tuple[str, dict[str, str], dict[str, Any] | None, str, str]] | None
    ) = None,
) -> tuple[list[ChatTurn], list[str]]:
    """Insert sentinel markers into the tool messages and call arguments.

    The markers are random per render and are unlikely to collide with any
    content the user supplied, so a template that drops any of them can be
    detected by looking for them in its output. Every tool message is
    marked, including ones with empty content — a template that ignores the
    tool role must not pass verification just because there was nothing to
    interpolate. Each call's function name is replaced with a sentinel, and
    every value of the call arguments is marked — strings get a suffix,
    numbers a replacement, booleans and null contribute their key — without
    adding or removing anything from the arguments, so the marked call
    stays valid for templates that validate the arguments against the
    schema.

    ``rename`` maps an offered tool's name to the probe name, the parameter
    renames of the probe schema, the probe parameters and the probe argument
    value injected into every marked call (see :func:`_verified`). Calls
    and tool messages that reference a renamed tool use the probe names, so
    the marked history stays consistent with the offered probe tools.
    """
    marked: list[ChatTurn] = []
    markers: list[str] = []
    rename = rename or {}
    for index, message in enumerate(messages):
        tool_name = message.tool_name
        if tool_name is not None and tool_name in rename:
            tool_name = rename[tool_name][0]
        if message.role == "tool":
            marker = f"qx{seed}{index}q"
            marked.append(
                ChatTurn("tool", f"{marker} {message.content}", message.tool_calls, tool_name)
            )
            markers.append(marker)
            continue
        if message.tool_calls:
            calls = []
            for ordinal, call in enumerate(message.tool_calls):
                original = call.function.name
                rename_map = rename.get(original)
                if rename_map is not None:
                    name_marker, param_map, parameters_schema, probe_key, _shared_value = rename_map
                    # Each call carries its own probe value: a template that
                    # renders only one of two identical calls cannot pass.
                    probe_value = f"qxval{seed}{index}{ordinal}"
                else:
                    name_marker = f"qxname{seed}{index}{ordinal}"
                    param_map = {}
                    parameters_schema = None
                    probe_key = f"qxkey{seed}{index}{ordinal}"
                    probe_value = f"qxval{seed}{index}{ordinal}"
                    markers.append(name_marker)
                arguments: dict[str, Any] = {
                    param_map.get(key, key): value for key, value in call.function.arguments.items()
                }
                arguments_marked_raw, argument_markers = _mark_argument_values(
                    arguments, seed, index, ordinal, parameters_schema
                )
                arguments = cast("dict[str, Any]", arguments_marked_raw)
                # The call needs a marker of its own: the schema markers
                # alone cannot prove the template rendered the call and not
                # only the offered tool definition. The probe key is
                # declared in the probe schema, so the marked call stays
                # schema-valid for strict templates. Without a probe
                # schema there is no schema to violate.
                arguments = {**arguments, probe_key: probe_value}
                argument_markers.append(probe_value)
                markers.extend(argument_markers)
                calls.append(
                    ToolCall(
                        function=ToolCallFunction(
                            name=name_marker,
                            arguments=arguments,
                        )
                    )
                )
            content = message.content
            if content.strip():
                marker = f"qx{seed}{index}t"
                content = f"{content} {marker}"
                markers.append(marker)
            marked.append(ChatTurn(message.role, content, tuple(calls), tool_name))
            continue
        marked.append(message)
    return marked, markers


def _schema_of(node: object) -> dict[str, Any]:
    """Return the JSON schema node, or an empty schema when absent."""
    return node if isinstance(node, dict) else {}


def _pick_number(value: float, low: object, high: object) -> int | None:
    """Pick a marker integer inside the schema bounds, different from value."""
    lo = math.ceil(low) if isinstance(low, (int, float)) else None
    hi = math.floor(high) if isinstance(high, (int, float)) else None
    if lo is not None and hi is not None:
        if lo > hi:
            return None
        for candidate in range(lo, hi + 1):
            if candidate != value:
                return candidate
        return None
    if lo is not None:
        return lo if lo != value else lo + 1
    if hi is not None:
        return hi if hi != value else hi - 1
    return 10**9 + secrets.randbelow(9 * 10**9)


def _mark_argument_values(
    value: object,
    seed: str,
    index: int,
    ordinal: int,
    schema: dict[str, Any] | None = None,
) -> tuple[object, list[str]]:
    """Replace every argument value with a marked counterpart.

    Strings get a random sentinel suffix (a different enum member when the
    schema constrains the value), numbers are replaced with a different
    value inside the schema bounds (a unique ten-digit marker when
    unconstrained), booleans are flipped to the other enum member when the
    enum allows it, and booleans/null/empty containers contribute their
    key as a marker. The structure, keys and constraints of the arguments
    are otherwise untouched, so the marked call stays valid for templates
    that check the arguments against the schema.
    """
    markers: list[str] = []
    counter = 0

    def mark(current: object, schema: object, key: str | None) -> object:
        nonlocal counter
        sch = _schema_of(schema)
        enum = sch.get("enum")
        enum = enum if isinstance(enum, list) else None
        if isinstance(current, bool) or current is None:
            flipped = None
            if isinstance(enum, list):
                candidates = [o for o in enum if isinstance(o, bool) and o is not current]
                if candidates:
                    flipped = candidates[0]
            if flipped is not None:
                markers.append(str(flipped).lower())
            elif key is not None:
                markers.append(key)
            return flipped if flipped is not None else current
        if isinstance(current, (dict, list)) and not current:
            # Empty containers have nothing to mark; require their key.
            if key is not None:
                markers.append(key)
            return current
        if isinstance(current, str):
            if enum is not None:
                choices = [o for o in enum if isinstance(o, str) and o != current]
                if choices:
                    markers.append(str(choices[0]))
                    return choices[0]
                if key is not None:
                    markers.append(key)
                return current
            counter += 1
            marker = f"qxarg{seed}{index}{ordinal}{counter}"
            markers.append(marker)
            return current + marker
        if isinstance(current, (int, float)):
            if enum is not None:
                numbers = [
                    o
                    for o in enum
                    if isinstance(o, (int, float)) and not isinstance(o, bool) and o != current
                ]
                if numbers:
                    markers.append(str(numbers[0]))
                    return numbers[0]
            marker_number = _pick_number(current, sch.get("minimum"), sch.get("maximum"))
            if marker_number is None:
                if key is not None:
                    markers.append(key)
                return current
            markers.append(str(marker_number))
            return marker_number
        if isinstance(current, dict):
            properties = _schema_of(sch.get("properties"))
            return {
                key: mark(item, properties.get(str(key), {}), str(key))
                for key, item in current.items()
            }
        if isinstance(current, list):
            item_schema = sch.get("items", {})
            return [mark(item, item_schema, key) for item in current]
        return current

    return mark(value, schema, None), markers


def _verified(
    render: ChatTemplate,
    candidate: Sequence[ChatTurn],
    tools: Sequence[ToolDefinition] | None,
    supports_tool_calls: bool,
    null_content: bool,
    required_fragment: str | None = None,
) -> bool:
    """Check that the template really renders tools and tool exchanges.

    The candidate is rendered once more with every offered tool re-probed
    (renamed to a sentinel, and its first parameter renamed as well, so a
    template that ignores the tool list or the schemas is caught), with each
    call's function name replaced by a sentinel, and
    with sentinel markers injected into the tool messages and every value of
    the call arguments (strings get a suffix, numbers a unique replacement;
    booleans and null cannot carry a marker, so their keys are required to
    appear). The structure of the arguments is otherwise left untouched, so
    the marked call stays valid for templates that validate the arguments
    against the schema. The sentinels are unambiguous markers that survive
    every tool argument serialization (JSON, dict literal, ``<parameter>``
    blocks), so a template that drops the offered tools or a part of the
    tool history is detected here and falls back to the transcript. A
    sentinel appended to the system message catches templates that silently
    drop it (e.g. one that renders the system prompt only for a trailing
    user turn). ``required_fragment`` is text that has to appear in the
    output as well; the merged system/user candidate uses it to make sure
    the merged instruction survives the retry.
    """
    candidate = list(candidate)
    system_probe = None
    for index, message in enumerate(candidate):
        if message.role == "system" and message.content.strip():
            system_probe = f"qxsys{secrets.token_hex(4)}"
            candidate[index] = ChatTurn(
                "system",
                f"{message.content}\n{system_probe}",
                message.tool_calls,
                message.tool_name,
            )
            break
    probe_tools: list[ToolDefinition] | None = None
    tool_markers: list[str] = []
    rename: dict[str, tuple[str, dict[str, str], dict[str, Any] | None, str, str]] = {}
    if tools:
        # Every offered tool is probed: it is renamed to a sentinel, its
        # first parameter is renamed the same way, and its schema carries a
        # probe property. The probe history is rendered against this
        # schema, so a strict template validating the call against the
        # offered schema still accepts it. The calls in the history are
        # renamed in the same way, so the probe stays consistent.
        probe_tools = []
        for index, tool in enumerate(tools):
            function = tool.function
            name_marker = f"qxtool{secrets.token_hex(4)}{index}"
            probe_key = f"qxkey{secrets.token_hex(4)}{index}"
            probe_value = f"qxval{secrets.token_hex(4)}{index}"
            parameters = function.parameters
            parameter_marker: str | None = None
            parameter_map: dict[str, str] = {}
            properties = parameters.get("properties")
            properties = dict(properties) if isinstance(properties, dict) else {}
            if properties:
                first_key = next(iter(properties))
                parameter_marker = f"qxparam{secrets.token_hex(4)}{index}"
                properties[parameter_marker] = properties.pop(first_key)
                parameters = dict(parameters)
                parameters["properties"] = properties
                required = parameters.get("required")
                if isinstance(required, list):
                    parameters["required"] = [
                        parameter_marker if item == first_key else item for item in required
                    ]
                parameter_map = {first_key: parameter_marker}
            parameters = dict(parameters)
            parameters["properties"] = {
                **parameters.get("properties", {}),
                probe_key: {"type": "string"},
            }
            rename[function.name] = (
                name_marker,
                parameter_map,
                parameters,
                probe_key,
                probe_value,
            )
            tool_markers.append(name_marker)
            if parameter_marker is not None:
                tool_markers.append(parameter_marker)
            probe_tools.append(
                ToolDefinition(
                    function=ToolFunction(
                        name=name_marker,
                        description=function.description,
                        parameters=parameters,
                    )
                )
            )
    seed = secrets.token_hex(6)
    marked, markers = _mark_tool_messages(candidate, seed, rename)
    if probe_tools is None and not markers and system_probe is None:
        return True

    def render_and_check(offered: Sequence[ToolDefinition] | None, required: list[str]) -> bool:
        rendered = _render_with_template(render, marked, offered, supports_tool_calls, null_content)
        if rendered is None:
            return False
        squashed = _squash(_unescape_json(rendered)).lower()
        if (
            required_fragment is not None
            and _squash(_unescape_json(required_fragment)).lower() not in squashed
        ):
            return False
        if system_probe is not None and system_probe.lower() not in squashed:
            return False
        return all(marker.lower() in squashed for marker in required)

    if probe_tools is None:
        return render_and_check(None, markers)
    # A single render with the offer covers the full history: the schema
    # markers prove the offered tools survive, the history markers prove
    # the conversation (including calls and their arguments) survives.
    if not render_and_check(probe_tools, [*tool_markers, *markers]):
        return False
    # A second render with a minimal history proves the tools themselves
    # are rendered, independent of the conversation: a template that
    # prints the history but drops the tool definitions fails here.
    blank = [ChatTurn("user", f"qxprobeblank{secrets.token_hex(4)}")]
    rendered = _render_with_template(render, blank, probe_tools, supports_tool_calls, False)
    if rendered is None:
        return False
    squashed = _squash(_unescape_json(rendered)).lower()
    return all(marker.lower() in squashed for marker in tool_markers)


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
    messages = _order_tool_results(list(messages))
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


def _order_tool_results(messages: Sequence[ChatTurn]) -> list[ChatTurn]:
    """Reorder each run of tool results to match the order of the calls.

    A client may send the results of a parallel tool call batch in any
    order. A template that renders tool results without names or ids
    (Qwen2.5) can only associate them with the calls by position, so each
    maximal run of consecutive tool messages is reordered to follow the
    pending calls: a result whose ``tool_name`` matches a pending call goes
    in the call's position, the remaining results keep their relative
    order. Calls and results without names keep their client-given order.

    The reordering happens on the messages, before the calls are rendered
    or inlined, so it applies to every template — including the ones that
    render the calls inline as text.
    """
    out: list[ChatTurn] = []
    pending: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.tool_calls:
            pending.extend(call.function.name for call in message.tool_calls)
            out.append(message)
            index += 1
            continue
        if message.role != "tool":
            out.append(message)
            index += 1
            continue
        end = index
        while end < len(messages) and messages[end].role == "tool":
            end += 1
        run = list(messages[index:end])
        batch = pending
        pending = []
        used = [False] * len(run)
        assigned: list[int | None] = [None] * len(batch)
        # First pass: each named call claims its name-matching result.
        for call_index, name in enumerate(batch):
            if not name:
                continue
            for index_in_run, item in enumerate(run):
                if used[index_in_run]:
                    continue
                if (item.tool_name or "") == name:
                    used[index_in_run] = True
                    assigned[call_index] = index_in_run
                    break
        # Second pass: unmatched calls take the remaining results in order,
        # keeping the run's positions aligned with the calls.
        for call_index in range(len(batch)):
            if assigned[call_index] is not None:
                continue
            for index_in_run in range(len(run)):
                if not used[index_in_run]:
                    used[index_in_run] = True
                    assigned[call_index] = index_in_run
                    break
        for assigned_index in assigned:
            if assigned_index is not None:
                out.append(run[assigned_index])
                used[assigned_index] = True
        out.extend(item for position, item in enumerate(run) if not used[position])
        index = end
    return out


def _assign_call_ids(payload: list[dict[str, object]]) -> None:
    """Assign synthetic call ids to the tool calls of an assistant turn.

    The Mistral family of templates requires every tool call to carry a
    nine-character alphanumeric ``id`` and every tool result a matching
    ``tool_call_id``. Other templates ignore the keys. The runs of tool
    results have been reordered to match the calls by then (see
    :func:`_order_tool_results`); each tool result gets the id of the call
    it answers, matched by ``tool_name`` when the client sends it and by
    order of appearance otherwise. The ids are deterministic for a given
    history.
    """
    counter = 0
    pending: list[tuple[str, str]] = []  # (name, id), oldest first
    for message in payload:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for call in cast("list[dict[str, Any]]", tool_calls):
                counter += 1
                call_id = f"call{counter:05d}"[-9:]
                call["id"] = call_id
                function = call.get("function") or {}
                name = function.get("name") if isinstance(function, dict) else ""
                pending.append((str(name), call_id))
        if message.get("role") == "tool":
            if pending:
                tool_name = message.get("tool_name")
                index = -1
                if isinstance(tool_name, str) and tool_name:
                    for candidate, (name, _) in enumerate(pending):
                        if name == tool_name:
                            index = candidate
                            break
                if index == -1:
                    index = 0
                message["tool_call_id"] = pending.pop(index)[1]
            else:
                message["tool_call_id"] = "call00000"


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
