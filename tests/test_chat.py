"""Unit tests for the rendering of chat histories."""

from __future__ import annotations

from typing import Any

from airllm_ollama.chat import render_chat, render_completion, render_transcript

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

    assert rendered == "<|im_start|>assistant\n"
    assert tokenizer.rendered == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_render_chat_falls_back_without_a_template() -> None:
    class PlainTokenizer:
        """Tokenizer double without a chat template."""

    rendered = render_chat(PlainTokenizer(), MESSAGES)

    assert rendered == render_transcript(MESSAGES)


def test_render_chat_falls_back_when_the_template_fails() -> None:
    class BrokenTokenizer:
        """Tokenizer double whose template rejects every message."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            raise ValueError("no template for this model")

    rendered = render_chat(BrokenTokenizer(), MESSAGES)

    assert rendered == render_transcript(MESSAGES)


def test_render_chat_falls_back_when_the_template_returns_nothing() -> None:
    class EmptyTokenizer:
        """Tokenizer double whose template returns an empty string."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Return nothing."""
            return "   "

    assert render_chat(EmptyTokenizer(), MESSAGES) == render_transcript(MESSAGES)


def test_render_completion_without_a_system_message() -> None:
    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            return "templated"

    assert render_completion(TemplateTokenizer(), "hello", None) == "hello"
    assert render_completion(TemplateTokenizer(), "hello", "") == "hello"


def test_render_completion_uses_the_chat_template() -> None:
    class TemplateTokenizer:
        """Tokenizer double with a chat template."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Render the messages the way a chat template would."""
            return f"templated:{messages}"

    rendered = render_completion(TemplateTokenizer(), "hello", "Be terse.")

    assert rendered == (
        "templated:[{'role': 'system', 'content': 'Be terse.'}, "
        "{'role': 'user', 'content': 'hello'}]"
    )


def test_render_completion_falls_back_to_the_transcript() -> None:
    class PlainTokenizer:
        """Tokenizer double without a chat template."""

    rendered = render_completion(PlainTokenizer(), "hello", "Be terse.")

    assert rendered == "System: Be terse.\n\nUser: hello\n\nAssistant:"


def test_render_completion_falls_back_when_the_template_fails() -> None:
    class BrokenTokenizer:
        """Tokenizer double whose template rejects every message."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Reject the messages."""
            raise ValueError("no template for this model")

    rendered = render_completion(BrokenTokenizer(), "hello", "Be terse.")

    assert rendered == "System: Be terse.\n\nUser: hello\n\nAssistant:"
