"""Rendering of a chat history into a single prompt.

Ollama renders the messages of ``/api/chat`` with the chat template of the
model. Base models do not have one, so the history is flattened into a plain
transcript instead.
"""

from __future__ import annotations

from collections.abc import Sequence

ChatTurn = tuple[str, str]


def render_transcript(messages: Sequence[ChatTurn]) -> str:
    """Flatten a chat history into a plain transcript.

    The transcript is used for models without a chat template::

        System: Be brief.

        User: Hello

        Assistant:

    Args:
        messages: The chat history as ``(role, content)`` pairs.

    Returns:
        The rendered prompt.
    """
    lines = [f"{role.capitalize()}: {content}" for role, content in messages]
    lines.append("Assistant:")
    return "\n\n".join(lines)


def render_chat(tokenizer: object, messages: Sequence[ChatTurn]) -> str:
    """Render a chat history with the chat template of the model.

    Falls back to :func:`render_transcript` when the tokenizer has no chat
    template or when the template refuses the messages.

    Args:
        tokenizer: The tokenizer of the model.
        messages: The chat history as ``(role, content)`` pairs.

    Returns:
        The rendered prompt.
    """
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        return render_transcript(messages)
    payload = [{"role": role, "content": content} for role, content in messages]
    try:
        rendered = template(payload, tokenize=False, add_generation_prompt=True)
    except Exception:
        # The template may reject any message (unknown role, missing tool
        # support, ...); the transcript always works.
        return render_transcript(messages)
    if not isinstance(rendered, str) or not rendered.strip():
        return render_transcript(messages)
    return rendered
