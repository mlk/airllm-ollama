"""Unit tests for the rendering of chat histories."""

from __future__ import annotations

import json
from typing import Any

from airllm_ollama.chat import ChatTurn, render_chat, render_completion, render_transcript
from airllm_ollama.schemas.tools import ToolCall, ToolCallFunction, ToolDefinition, ToolFunction
from fakes import SystemRoleTokenizer

MESSAGES = [
    ChatTurn("system", "Be brief."),
    ChatTurn("user", "Hello"),
    ChatTurn("assistant", "Hi!"),
]


def test_render_transcript_flattens_the_history() -> None:
    rendered = render_transcript(MESSAGES)

    assert rendered == "System: Be brief.\n\nUser: Hello\n\nAssistant: Hi!\n\nAssistant:"


def test_render_transcript_of_a_single_message() -> None:
    assert render_transcript([ChatTurn("user", "Hello")]) == "User: Hello\n\nAssistant:"


def test_render_transcript_of_an_empty_history() -> None:
    assert render_transcript([]) == "Assistant:"


def test_render_chat_uses_the_chat_template() -> None:
    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def __init__(self) -> None:
            """Record the rendered messages."""
            self.rendered: list[dict[str, str]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            self.rendered = messages
            return "<|im_start|>assistant\n"

    tokenizer = TemplateTokenizer()

    rendered = render_chat(tokenizer, MESSAGES)

    assert rendered.text == "<|im_start|>assistant\n"
    assert tokenizer.rendered == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_a_templated_prompt_is_tokenized_without_the_special_tokens() -> None:
    """The template renders the special tokens, the tokenizer must not add them."""

    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            return "<|begin_of_text|><|im_start|>assistant\n"

    rendered = render_chat(TemplateTokenizer(), MESSAGES)

    assert rendered.text.startswith("<|begin_of_text|>")
    assert rendered.add_special_tokens is False


def test_render_chat_falls_back_without_a_template() -> None:
    class PlainTokenizer:
        """Tokenizer double without a chat template."""

    rendered = render_chat(PlainTokenizer(), MESSAGES)

    assert rendered.text == render_transcript(MESSAGES)
    assert rendered.add_special_tokens is True


def test_render_chat_falls_back_when_the_template_fails() -> None:
    class BrokenTokenizer:
        """Tokenizer double whose template rejects every message."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            raise ValueError("no template for this model")

    rendered = render_chat(BrokenTokenizer(), MESSAGES)

    assert rendered.text == render_transcript(MESSAGES)
    assert rendered.add_special_tokens is True


def test_render_chat_falls_back_when_the_template_returns_nothing() -> None:
    class EmptyTokenizer:
        """Tokenizer double whose template returns an empty string."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Return nothing."""
            return "   "

    rendered = render_chat(EmptyTokenizer(), MESSAGES)

    assert rendered.text == render_transcript(MESSAGES)
    assert rendered.add_special_tokens is True


def test_a_template_that_refuses_the_system_role_gets_the_merged_messages() -> None:
    """A template without support for the system role is asked a second time."""
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    rendered = render_chat(tokenizer, [ChatTurn("system", "Be brief."), ChatTurn("user", "Hello")])

    assert rendered.text == "templated"
    assert rendered.add_special_tokens is False
    assert tokenizer.renderings == [[{"role": "user", "content": "Be brief.\n\nHello"}]]


def test_the_history_after_the_first_user_turn_is_kept() -> None:
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    render_chat(
        tokenizer,
        [
            ChatTurn("system", "Be brief."),
            ChatTurn("user", "Hello"),
            ChatTurn("assistant", "Hi!"),
            ChatTurn("user", "Again"),
        ],
    )

    assert tokenizer.renderings == [
        [
            {"role": "user", "content": "Be brief.\n\nHello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Again"},
        ]
    ]


def test_every_system_message_is_merged_into_the_first_user_turn() -> None:
    """A system message in the middle of the history is merged as well."""
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    render_chat(
        tokenizer,
        [ChatTurn("user", "Hello"), ChatTurn("system", "Be brief."), ChatTurn("user", "Again")],
    )

    assert tokenizer.renderings == [
        [
            {"role": "user", "content": "Be brief.\n\nHello"},
            {"role": "user", "content": "Again"},
        ]
    ]


def test_an_empty_system_message_is_dropped_by_the_merge() -> None:
    """An empty system message must not add newlines to the user turn."""
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    render_chat(tokenizer, [ChatTurn("system", "   "), ChatTurn("user", "Hello")])

    assert tokenizer.renderings == [[{"role": "user", "content": "Hello"}]]


def test_a_template_that_refuses_everything_falls_back_to_the_transcript() -> None:
    class BrokenTokenizer:
        """Tokenizer double whose template refuses every message."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            raise ValueError("no template for this model")

    messages = [ChatTurn("system", "Be brief."), ChatTurn("user", "Hello")]

    rendered = render_chat(BrokenTokenizer(), messages)

    assert rendered.text == render_transcript(messages)
    assert rendered.add_special_tokens is True


def test_a_history_without_a_user_turn_is_not_asked_again() -> None:
    """There is no user turn to merge the system message into."""

    class BrokenTokenizer:
        """Tokenizer double whose template refuses every message."""

        def __init__(self) -> None:
            """Count the calls of the template."""
            self.calls = 0

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            self.calls += 1
            raise ValueError("no template for this model")

    tokenizer = BrokenTokenizer()

    render_chat(tokenizer, [ChatTurn("system", "Be brief."), ChatTurn("assistant", "Hi!")])

    assert tokenizer.calls == 1


def test_render_completion_applies_the_chat_template_without_a_system_message() -> None:
    """Ollama applies the template to every prompt."""

    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            return f"templated:{messages}"

    rendered = render_completion(TemplateTokenizer(), "hello", None)

    assert rendered.text == "templated:[{'role': 'user', 'content': 'hello'}]"
    assert rendered.add_special_tokens is False


def test_render_completion_sends_a_raw_prompt_unchanged() -> None:
    """`raw` asks for the prompt without the template."""

    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            return "templated"

    rendered = render_completion(TemplateTokenizer(), "hello", "Be terse.", raw=True)

    assert rendered.text == "hello"
    assert rendered.add_special_tokens is True


def test_render_completion_uses_the_chat_template() -> None:
    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            return f"templated:{messages}"

    rendered = render_completion(TemplateTokenizer(), "hello", "Be terse.")

    assert rendered.text == (
        "templated:[{'role': 'system', 'content': 'Be terse.'}, "
        "{'role': 'user', 'content': 'hello'}]"
    )
    assert rendered.add_special_tokens is False


def test_render_completion_joins_the_system_message_without_a_template() -> None:
    """A model without a template gets the system message in front of the prompt."""

    class PlainTokenizer:
        """Tokenizer double without a chat template."""

    rendered = render_completion(PlainTokenizer(), "hello", "Be terse.")

    assert rendered.text == "Be terse.\n\nhello"
    assert rendered.add_special_tokens is True


def test_render_completion_keeps_the_prompt_without_a_template() -> None:
    class PlainTokenizer:
        """Tokenizer double without a chat template."""

    rendered = render_completion(PlainTokenizer(), "hello", None)

    assert rendered.text == "hello"
    assert rendered.add_special_tokens is True


def test_render_completion_joins_the_system_message_when_the_template_fails() -> None:
    class BrokenTokenizer:
        """Tokenizer double whose template rejects every message."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            raise ValueError("no template for this model")

    rendered = render_completion(BrokenTokenizer(), "hello", "Be terse.")

    assert rendered.text == "Be terse.\n\nhello"
    assert rendered.add_special_tokens is True


def test_render_completion_merges_the_system_message_when_the_template_refuses_it() -> None:
    """A template without support for the system role is asked a second time."""
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    rendered = render_completion(tokenizer, "hello", "Be terse.")

    assert rendered.text == "templated"
    assert rendered.add_special_tokens is False
    assert tokenizer.renderings == [[{"role": "user", "content": "Be terse.\n\nhello"}]]


TOOL_MESSAGES = [
    ChatTurn("user", "What is the weather in Paris?"),
    ChatTurn(
        "assistant",
        "",
        tool_calls=(
            ToolCall(function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})),
        ),
    ),
    ChatTurn("tool", "22°C", tool_name="get_weather"),
]

TOOLS = [
    ToolDefinition(
        function=ToolFunction(
            name="get_weather",
            description="Get the weather of a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    )
]


def test_render_transcript_renders_tool_calls_and_tool_results() -> None:
    rendered = render_transcript(TOOL_MESSAGES)

    assert rendered == (
        "User: What is the weather in Paris?\n\n"
        'Assistant: [tool_calls: get_weather({"city": "Paris"})]\n\n'
        "Tool[get_weather]: 22°C\n\n"
        "Assistant:"
    )


def test_render_transcript_lists_the_offered_tools() -> None:
    rendered = render_transcript([ChatTurn("user", "Hello")], tools=TOOLS)

    assert rendered.startswith('Tools: [{"type": "function"')
    assert rendered.endswith("User: Hello\n\nAssistant:")


def test_the_template_receives_the_tool_calls_and_the_tool_results() -> None:
    tokenizer = FakeToolTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assert tokenizer.renderings[-1] == [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    "id": "call00001",
                },
            ],
        },
        {
            "role": "tool",
            "content": "22°C",
            "tool_name": "get_weather",
            "name": "get_weather",
            "tool_call_id": "call00001",
        },
    ]


def test_the_template_receives_the_offered_tools() -> None:
    tokenizer = FakeToolTokenizer()

    render_chat(tokenizer, [ChatTurn("user", "Hello")], tools=TOOLS)

    assert tokenizer.tools == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather of a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_a_template_without_the_tools_keyword_still_renders_the_history() -> None:
    """A template that refuses the tools falls back to the transcript with the tools."""

    class ToolsRefusingTokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            self.calls += 1
            raise ValueError("unexpected keyword argument 'tools'")

    rendered = render_chat(ToolsRefusingTokenizer(), [ChatTurn("user", "Hello")], tools=TOOLS)

    assert rendered.text.startswith("Tools: [")
    assert "User: Hello" in rendered.text


class FakeToolTokenizer:
    """Tokenizer double recording the messages and the tools it was given."""

    def __init__(self) -> None:
        """Initialise the recording lists."""
        self.renderings: list[list[dict[str, Any]]] = []
        self.tools: Any = None

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render the messages the way a chat template would."""
        self.renderings.append(list(messages))
        self.tools = kwargs.get("tools")
        kwargs_tools = kwargs.get("tools") or []
        echoed = [
            str(message.get("content", ""))
            + "".join(
                str(call["function"]["name"]) + str(call["function"]["arguments"])
                for call in message.get("tool_calls", [])
            )
            for message in messages
            if isinstance(message, dict)
        ]
        echoed.append(json.dumps(kwargs_tools, default=str))
        return "<​|im_start|>assistant\n" + "\n".join(echoed)


class ContentOnlyTokenizer:
    """Tokenizer whose template renders the content and ignores tool calls."""

    def __init__(self) -> None:
        """Initialise the recording lists."""
        self.renderings: list[list[dict[str, Any]]] = []
        self.chat_template = "{{ message.content }}"

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render the content of the messages, like SmolLM3 does."""
        self.renderings.append(list(messages))
        return "\n".join(
            str(message.get("content", ""))
            + "".join(str(call["function"]["arguments"]) for call in message.get("tool_calls", []))
            for message in messages
            if isinstance(message, dict)
        )


def test_a_template_that_ignores_tool_calls_gets_them_inlined() -> None:
    """A content-only template sees the calls as part of the content."""
    tokenizer = ContentOnlyTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assistant = tokenizer.renderings[-1][1]
    assert "tool_calls" not in assistant
    assert (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        in (assistant["content"])
    )


def test_a_template_that_reads_tool_calls_gets_the_structured_payload() -> None:
    """A tool-aware template gets the structured payload, not the inlined one."""

    class ToolAwareTokenizer(ContentOnlyTokenizer):
        """Tokenizer whose template references `tool_calls`."""

        def __init__(self) -> None:
            """Set a template that renders the tool calls."""
            super().__init__()
            self.chat_template = "{{ message.tool_calls }}"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the calls like a template that reads them does."""
            self.renderings.append(list(messages))
            return "\n".join(
                str(message.get("content", ""))
                + "".join(
                    str(call["function"]["name"]) + str(call["function"]["arguments"])
                    for call in message.get("tool_calls", [])
                )
                + str(message.get("tool_name", ""))
                for message in messages
                if isinstance(message, dict)
            )

    tokenizer = ToolAwareTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assistant = tokenizer.renderings[-1][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert assistant["content"] == ""


def test_the_template_receives_an_empty_description_when_none_is_declared() -> None:
    tokenizer = FakeToolTokenizer()

    render_chat(
        tokenizer,
        [ChatTurn("user", "Hello")],
        tools=[ToolDefinition(function=ToolFunction(name="get_weather"))],
    )

    assert tokenizer.tools[0]["function"]["description"] == ""


class DictTemplateTokenizer:
    """Tokenizer double with a dict of named templates (Llama 3.1 style)."""

    def __init__(self) -> None:
        """Initialise the recording lists."""
        self.renderings: list[list[dict[str, Any]]] = []
        self.chat_template = {
            "default": "{{ message.content }}",
            "tool_use": "{{ message.tool_calls }}",
        }

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render the messages, like a named-template tokenizer would."""
        self.renderings.append(list(messages))
        branch = "tool_use" if kwargs.get("tools") else "default"
        return (
            f"{branch}:"
            + json.dumps(kwargs.get("tools") or [], default=str)
            + "|"
            + json.dumps(messages, default=str)
        )


def test_a_dict_template_without_tools_selects_the_default_branch() -> None:
    """The default branch renders content only, so the calls are inlined."""
    tokenizer = DictTemplateTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assistant = tokenizer.renderings[-1][1]
    assert "tool_calls" not in assistant
    assert "<tool_call>" in assistant["content"]


def test_a_dict_template_with_tools_selects_the_tool_use_branch() -> None:
    """The tool_use branch renders the calls, so they stay structured."""
    tokenizer = DictTemplateTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES, tools=TOOLS)

    assistant = tokenizer.renderings[-1][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert assistant["content"] == ""


class R1StyleTokenizer:
    """Tokenizer double mimicking the DeepSeek-R1 template.

    The R1 template renders an assistant turn's tool calls only when the
    content is null, and skips the turn otherwise — an empty string content
    silently drops the calls.
    """

    def __init__(self) -> None:
        """Initialise the recording list."""
        self.chat_template = (
            "{% if message['content'] is none %}{{ message.tool_calls }}{% endif %}"
        )
        self.rendered = ""

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render the messages the way the R1 template does."""
        parts = []
        if kwargs.get("tools"):
            parts.append(json.dumps(kwargs["tools"], default=str))
        for message in messages:
            if message["role"] == "assistant":
                if message["content"] is None:
                    for call in message.get("tool_calls", []):
                        name = call["function"]["name"]
                        arguments = json.dumps(call["function"]["arguments"])
                        parts.append(f"<tool_call>{name}{arguments}</tool_call>")
                else:
                    parts.append(str(message["content"]))
            else:
                parts.append(str(message["content"]))
        self.rendered = "|".join(parts)
        return self.rendered


def test_a_template_that_requires_null_content_renders_the_tool_calls() -> None:
    """The history is not silently emptied for content-is-none templates."""
    tokenizer = R1StyleTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assert "<tool_call>get_weather" in tokenizer.rendered


def test_an_assistant_message_with_text_and_tool_calls_keeps_its_content() -> None:
    tokenizer = ContentOnlyTokenizer()
    messages = [
        ChatTurn(
            "assistant",
            "Checking.",
            tool_calls=(
                ToolCall(
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})
                ),
            ),
        ),
    ]

    render_chat(tokenizer, messages)

    assistant = tokenizer.renderings[-1][0]
    assert assistant["content"].startswith("Checking.\n<tool_call>")


def test_a_named_template_dictionary_prefers_the_default_branch() -> None:
    """A tool_use entry is passed over when tools are offered."""

    class NamedTemplateTokenizer(DictTemplateTokenizer):
        """A dict template with an unrelated entry before default."""

        def __init__(self) -> None:
            """Initialise with a default that reads tool_calls."""
            super().__init__()
            self.chat_template = {
                "aardvark": "{{ message.content }}",
                "default": "{{ message.tool_calls }}",
                "tool_use": "{{ message.tool_calls }}",
            }

    tokenizer = NamedTemplateTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES, tools=TOOLS)

    assistant = tokenizer.renderings[-1][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"


def test_a_dict_template_without_tool_use_falls_back_to_default() -> None:
    """Without a tool_use branch, the default branch is the tool-aware one."""

    class NoToolUseTokenizer(DictTemplateTokenizer):
        """A dict template with no tool_use entry at all."""

        def __init__(self) -> None:
            """Initialise with a content-only entry before the default."""
            super().__init__()
            self.chat_template = {
                "aardvark": "{{ message.content }}",
                "default": "{{ message.tool_calls }}",
            }

    tokenizer = NoToolUseTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES, tools=TOOLS)

    assistant = tokenizer.renderings[-1][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"


class ToolBlindTokenizer:
    """Tokenizer double mimicking templates that render only system/user/assistant.

    Zephyr's template renders those three roles and silently ignores the
    tool role, so the tool result never reaches the model.
    """

    def __init__(self) -> None:
        """Initialise the recording list."""
        self.rendered: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render everything except tool messages."""
        self.rendered.append(list(messages))
        return "|".join(
            str(message["content"]) for message in messages if message["role"] != "tool"
        )


def test_a_template_that_drops_tool_messages_falls_back_to_the_transcript() -> None:
    tokenizer = ToolBlindTokenizer()

    rendered = render_chat(tokenizer, TOOL_MESSAGES)

    assert "Tool[get_weather]: 22°C" in rendered.text


class ParameterTemplateTokenizer:
    """Tokenizer double mimicking the Qwen3-Coder parameter blocks."""

    def __init__(self) -> None:
        """Initialise the recording list."""
        self.rendered: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render parameter blocks per argument and keep tool results."""
        self.rendered.append(list(messages))
        parts = []
        for message in messages:
            if message["role"] == "tool":
                parts.append(f"<tool_response>{message['content']}</tool_response>")
                continue
            if message.get("tool_calls"):
                body = "".join(
                    f"<parameter={key}>\n{value}\n</parameter>"
                    for call in message["tool_calls"]
                    for key, value in call["function"]["arguments"].items()
                )
                parts.append(
                    f"<tool_call>{message['tool_calls'][0]['function']['name']}</tool_call>{body}"
                )
                continue
            parts.append(str(message["content"]))
        return "|".join(parts)


def test_a_parameter_block_template_keeps_its_own_format() -> None:
    tokenizer = ParameterTemplateTokenizer()

    rendered = render_chat(tokenizer, TOOL_MESSAGES)

    assert "<tool_response>22°C</tool_response>" in rendered.text
    assert "<parameter=city>\nParis\n</parameter>" in rendered.text


def test_a_tool_result_matching_the_arguments_still_falls_back() -> None:
    """A result equal to an argument value must not vouch for itself."""

    class EchoTemplate(DictTemplateTokenizer):
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            self.renderings.append(list(messages))
            parts = []
            for message in messages:
                if message["role"] == "tool":
                    continue  # drops the tool result, renders the rest
                if message.get("tool_calls"):
                    call = message["tool_calls"][0]
                    parts.append(
                        f"{call['function']['name']} " + json.dumps(call["function"]["arguments"])
                    )
                else:
                    parts.append(str(message["content"]))
            return "|".join(parts)

    messages = [
        TOOL_MESSAGES[0],
        TOOL_MESSAGES[1],
        ChatTurn("tool", "Paris", tool_name="get_weather"),
    ]

    rendered = render_chat(EchoTemplate(), messages)

    assert "Tool[get_weather]: Paris" in rendered.text


def test_a_template_that_ignores_the_offered_tools_falls_back_to_the_transcript() -> None:
    class IgnoringTemplateTokenizer:
        """Render the messages but not the tools."""

        def __init__(self) -> None:
            """Initialise the recording list."""
            self.renderings: list[list[dict[str, Any]]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages, ignoring the tools keyword."""
            self.renderings.append(list(messages))
            return "<|im_start|>assistant\n"

    rendered = render_chat(IgnoringTemplateTokenizer(), [ChatTurn("user", "Hello")], tools=TOOLS)

    assert "Tools: [" in rendered.text
    assert "get_weather" in rendered.text


def test_a_template_that_concatenates_string_arguments_is_supported() -> None:
    """Templates like DeepSeek-R1 that stringify tool arguments keep their format."""

    class StringArgumentsTokenizer:
        """Tokenizer double that concatenates arguments into the output."""

        def __init__(self) -> None:
            """Initialise the recording list."""
            self.renderings: list[list[dict[str, Any]]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render, requiring arguments to be a string."""
            self.renderings.append(list(messages))
            parts = []
            for message in messages:
                for call in message.get("tool_calls", []):
                    arguments = call["function"]["arguments"]
                    if not isinstance(arguments, str):
                        raise TypeError("arguments must be a string")
                    parts.append(f"<call>{call['function']['name']}:{arguments}</call>")
            return "|".join(parts) or "empty"

    messages = [
        ChatTurn(
            "assistant",
            "",
            tool_calls=(
                ToolCall(
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})
                ),
            ),
        ),
    ]

    rendered = render_chat(StringArgumentsTokenizer(), messages, TOOLS)

    assert "get_weather" in rendered.text
    assert '{"city": "Paris"}' in rendered.text


class MistralTokenizer:
    """Tokenizer double enforcing the Mistral v0.3 tool-call constraints.

    The real template raises unless every tool call carries a nine
    alphanumeric character ``id`` and every tool result a matching
    ``tool_call_id`` of the same shape.
    """

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render the messages, raising without valid identifiers."""
        parts: list[str] = ["<s>"]
        for message in messages:
            if message["role"] == "user":
                parts.append(f"[INST] {message['content']} [/INST]")
                continue
            tool_calls = message.get("tool_calls")
            if tool_calls:
                parts.append("[TOOL_CALLS] [")
                for index, call in enumerate(tool_calls):
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or len(call_id) != 9 or not call_id.isalnum():
                        raise ValueError("tool call id must be 9 alphanumeric characters")
                    arguments = json.dumps(call["function"]["arguments"])
                    parts.append(
                        f'{{"name": "{call["function"]["name"]}", '
                        f'"arguments": {arguments}, "id": "{call_id}"}}'
                    )
                    if index != len(tool_calls) - 1:
                        parts.append(", ")
                parts.append("]")
                continue
            if message["role"] == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or len(call_id) != 9 or not call_id.isalnum():
                    raise ValueError("tool call id must be 9 alphanumeric characters")
                parts.append(f'[TOOL_RESULTS] {{"content": {message["content"]}, ')
                parts.append(f'"call_id": "{call_id}"}}[/TOOL_RESULTS]')
        return "".join(parts)


def test_a_mistral_style_template_with_identifiers_renders_the_history() -> None:
    """Synthetic ids satisfy the Mistral v0.3 tool-call constraints."""
    tokenizer = MistralTokenizer()
    messages = TOOL_MESSAGES

    rendered = render_chat(tokenizer, messages)

    assert "[TOOL_CALLS]" in rendered.text
    assert '"call_id": "call00001"' in rendered.text
    assert "call00001" in rendered.text


def test_a_glm_style_template_gets_an_empty_string_not_null() -> None:
    """A template interpolating content directly must not see the literal `None`."""

    class GLMTokenizer:
        """Tokenizer double mimicking GLM-4.7's direct interpolation."""

        def __init__(self) -> None:
            """Initialise the recording list."""
            self.rendered: list[list[dict[str, Any]]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render each message by interpolating its content directly."""
            self.rendered.append(list(messages))
            parts = []
            for message in messages:
                parts.append(f"[{message['content']}]")
                for call in message.get("tool_calls", []):
                    import json as _json

                    arguments = _json.dumps(call["function"]["arguments"])
                    parts.append(f"<call>{call['function']['name']}{arguments}</call>")
            return "".join(parts)

    messages = [
        ChatTurn(
            "assistant",
            "",
            tool_calls=(
                ToolCall(
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})
                ),
            ),
        ),
    ]

    rendered = render_chat(GLMTokenizer(), messages)

    assert "None" not in rendered.text
    assert '<call>get_weather{"city": "Paris"}</call>' in rendered.text
    assert " qxsentinel" not in rendered.text


def test_a_template_dropping_assistant_content_falls_back() -> None:
    """A template that renders tool_calls but not the content falls back."""

    class ContentBlindTokenizer:
        """Tokenizer double rendering tool_calls but ignoring content."""

        def __init__(self) -> None:
            """Initialise the recording list."""
            self.rendered: list[list[dict[str, Any]]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render only the tool calls of each message."""
            self.rendered.append(list(messages))
            return "|".join(
                f"<call>{call['function']['name']}</call>"
                for message in messages
                for call in message.get("tool_calls", [])
            )

    rendered = render_chat(ContentBlindTokenizer(), TOOL_MESSAGES)

    assert rendered.text.startswith("User:")
    assert "Tool[get_weather]: 22°C" in rendered.text


def test_an_empty_tool_result_is_still_visible_to_the_probe() -> None:
    """A Zephyr-style template that drops the tool role must be detected."""

    class ToolBlindTokenizer:
        """Tokenizer double that renders only non-tool messages."""

        def __init__(self) -> None:
            """Initialise the recording list."""
            self.renderings: list[list[dict[str, Any]]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render everything except tool messages."""
            self.renderings.append(list(messages))
            return "|".join(
                str(message["content"])
                for message in messages
                if isinstance(message, dict) and message.get("role") != "tool"
            )

    messages = [
        ChatTurn("user", "Weather?"),
        ChatTurn(
            "assistant",
            "",
            tool_calls=(
                ToolCall(
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})
                ),
            ),
        ),
        ChatTurn("tool", "", tool_name="get_weather"),
    ]

    rendered = render_chat(ToolBlindTokenizer(), messages, TOOLS)

    assert "Tool[get_weather]:" in rendered.text


def test_a_template_that_ignores_tools_falls_back_even_if_the_user_names_the_tool() -> None:
    class NameEchoTokenizer:
        """Render the messages and the user's mention without offering tools."""

        def __init__(self) -> None:
            """Initialise the recording list."""
            self.renderings: list[list[dict[str, Any]]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Echo the messages, silently dropping the tools keyword."""
            self.renderings.append(list(messages))
            return "|".join(
                str(message.get("content", "")) for message in messages if isinstance(message, dict)
            )

    messages = [ChatTurn("user", "Please call get_weather for Paris")]

    rendered = render_chat(NameEchoTokenizer(), messages, TOOLS)

    assert "Tools: [" in rendered.text
