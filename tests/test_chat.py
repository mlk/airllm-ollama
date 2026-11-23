"""Unit tests for the rendering of chat histories."""

from __future__ import annotations

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

    assert tokenizer.renderings[0] == [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                },
            ],
        },
        {
            "role": "tool",
            "content": "22°C",
            "tool_name": "get_weather",
            "name": "get_weather",
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
        return "<​|im_start|>assistant\n"


class ContentOnlyTokenizer:
    """Tokenizer whose template renders the content and ignores tool calls."""

    def __init__(self) -> None:
        """Initialise the recording lists."""
        self.renderings: list[list[dict[str, Any]]] = []
        self.chat_template = "{{ message.content }}"

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        """Render the content of the messages, like SmolLM3 does."""
        self.renderings.append(list(messages))
        return "rendered"


def test_a_template_that_ignores_tool_calls_gets_them_inlined() -> None:
    """A content-only template sees the calls as part of the content."""
    tokenizer = ContentOnlyTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assistant = tokenizer.renderings[0][1]
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

    tokenizer = ToolAwareTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assistant = tokenizer.renderings[0][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "<tool_call>" not in assistant["content"]


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
        return "rendered"


def test_a_dict_template_without_tools_selects_the_default_branch() -> None:
    """The default branch renders content only, so the calls are inlined."""
    tokenizer = DictTemplateTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES)

    assistant = tokenizer.renderings[0][1]
    assert "tool_calls" not in assistant
    assert "<tool_call>" in assistant["content"]


def test_a_dict_template_with_tools_selects_the_tool_use_branch() -> None:
    """The tool_use branch renders the calls, so they stay structured."""
    tokenizer = DictTemplateTokenizer()

    render_chat(tokenizer, TOOL_MESSAGES, tools=TOOLS)

    assistant = tokenizer.renderings[0][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "<tool_call>" not in assistant["content"]
