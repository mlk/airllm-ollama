"""Unit tests for the rendering of chat histories."""

from __future__ import annotations

from typing import Any

from airllm_ollama.chat import render_chat, render_completion, render_transcript
from fakes import SystemRoleTokenizer

MESSAGES = [
    ("system", "Be brief."),
    ("user", "Hello"),
    ("assistant", "Hi!"),
]


def test_render_transcript_flattens_the_history() -> None:
    rendered = render_transcript(MESSAGES)

    assert rendered == "System: Be brief.\n\nUser: Hello\n\nAssistant: Hi!\n\nAssistant:"


def test_render_transcript_of_a_single_message() -> None:
    assert render_transcript([("user", "Hello")]) == "User: Hello\n\nAssistant:"


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

    rendered = render_chat(tokenizer, [("system", "Be brief."), ("user", "Hello")])

    assert rendered.text == "templated"
    assert rendered.add_special_tokens is False
    assert tokenizer.renderings == [[{"role": "user", "content": "Be brief.\n\nHello"}]]


def test_the_history_after_the_first_user_turn_is_kept() -> None:
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    render_chat(
        tokenizer,
        [
            ("system", "Be brief."),
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "Again"),
        ],
    )

    assert tokenizer.renderings == [
        [
            {"role": "user", "content": "Be brief.\n\nHello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Again"},
        ]
    ]


def test_an_empty_system_message_is_dropped_by_the_merge() -> None:
    """An empty system message must not add newlines to the user turn."""
    tokenizer = SystemRoleTokenizer()
    tokenizer.template = "templated"

    render_chat(tokenizer, [("system", "   "), ("user", "Hello")])

    assert tokenizer.renderings == [[{"role": "user", "content": "Hello"}]]


def test_a_template_that_refuses_everything_falls_back_to_the_transcript() -> None:
    class BrokenTokenizer:
        """Tokenizer double whose template refuses every message."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            raise ValueError("no template for this model")

    messages = [("system", "Be brief."), ("user", "Hello")]

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

    render_chat(tokenizer, [("system", "Be brief."), ("assistant", "Hi!")])

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
