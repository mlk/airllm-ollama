"""Rendering of a chat history into a single prompt.

Ollama renders the messages of ``/api/chat`` with the chat template of the
model. Base models do not have one, so the history is flattened into a plain
transcript instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

ChatTurn = tuple[str, str]


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


def render_completion(tokenizer: object, prompt: str, system: str | None) -> RenderedPrompt:
    """Render a completion prompt together with its system message.

    A model with a chat template expects the system message in the format of
    that template, so both are rendered as a chat — the same way
    ``/api/chat`` does it. Models without a template get the plain transcript.

    Args:
        tokenizer: The tokenizer of the model.
        prompt: The prompt of the user.
        system: The system message, if any.

    Returns:
        The rendered prompt and how it has to be tokenized.
    """
    if not system:
        # No template rendered the prompt, so the tokenizer adds the special
        # tokens. A `raw: true` request ends up here as well — a client that
        # rendered the template itself sends the BOS token with the prompt
        # (#31, #44).
        return RenderedPrompt(prompt, add_special_tokens=True)
    return render_chat(tokenizer, [("system", system), ("user", prompt)])


def render_chat(tokenizer: object, messages: Sequence[ChatTurn]) -> RenderedPrompt:
    """Render a chat history with the chat template of the model.

    Falls back to :func:`render_transcript` when the tokenizer has no chat
    template or when the template refuses the messages.

    Args:
        tokenizer: The tokenizer of the model.
        messages: The chat history as ``(role, content)`` pairs.

    Returns:
        The rendered prompt and how it has to be tokenized.
    """
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        return RenderedPrompt(render_transcript(messages), add_special_tokens=True)
    payload = [{"role": role, "content": content} for role, content in messages]
    try:
        rendered = template(payload, tokenize=False, add_generation_prompt=True)
    except Exception:
        # The template may reject any message (unknown role, missing tool
        # support, ...); the transcript always works.
        return RenderedPrompt(render_transcript(messages), add_special_tokens=True)
    if not isinstance(rendered, str) or not rendered.strip():
        return RenderedPrompt(render_transcript(messages), add_special_tokens=True)
    # The template rendered the special tokens of the model, so the tokenizer
    # must not add them a second time.
    return RenderedPrompt(rendered, add_special_tokens=False)
