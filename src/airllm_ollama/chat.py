"""Rendering of a chat history into a single prompt.

Ollama renders the messages of ``/api/chat`` with the chat template of the
model. Base models do not have one, so the history is flattened into a plain
transcript instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

ChatTurn = tuple[str, str]

# The system message is merged into the first user turn with this separator —
# the same one the plain join of a model without a template uses.
SEPARATOR = "\n\n"


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


def apply_chat_template(tokenizer: object, messages: Sequence[ChatTurn]) -> str | None:
    """Render the messages with the chat template of the model.

    A template that refuses the messages is asked a second time, with the
    system message merged into the first user turn. Templates without support
    for the system role (Gemma, older Mistral versions) raise, and the model
    would lose its own format completely — the merged messages keep it.

    Args:
        tokenizer: The tokenizer of the model.
        messages: The chat history as ``(role, content)`` pairs.

    Returns:
        The rendered prompt, or ``None`` when the model has no chat template
        or when the template refuses the messages.
    """
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        return None
    render = cast("ChatTemplate", template)
    rendered = _render_with_template(render, messages)
    if rendered is not None:
        return rendered
    merged = _merge_system_message(messages)
    if merged is None:
        return None
    return _render_with_template(render, merged)


def _render_with_template(template: ChatTemplate, messages: Sequence[ChatTurn]) -> str | None:
    """Render the messages with the template.

    Args:
        template: The chat template of the model.
        messages: The chat history as ``(role, content)`` pairs.

    Returns:
        The rendered prompt, or ``None`` when the template refuses the
        messages.
    """
    payload = [{"role": role, "content": content} for role, content in messages]
    try:
        rendered = template(payload, tokenize=False, add_generation_prompt=True)
    except Exception:
        # The template may reject any message (unknown role, missing tool
        # support, ...); the caller decides what to do instead.
        return None
    if not isinstance(rendered, str) or not rendered.strip():
        return None
    return rendered


def _merge_system_message(messages: Sequence[ChatTurn]) -> list[ChatTurn] | None:
    """Merge the system message into the first user turn.

    A template without support for the system role refuses the messages. The
    system message becomes part of the first user turn instead, so that the
    model still gets its own format.

    Args:
        messages: The chat history as ``(role, content)`` pairs.

    Returns:
        The messages with the system message merged, or ``None`` when there is
        no system message or no user turn to merge it into.
    """
    if len(messages) < 2 or messages[0][0] != "system":
        return None
    system = messages[0][1]
    merged: list[ChatTurn] = []
    merged_system = False
    for role, content in messages[1:]:
        if role == "user" and not merged_system:
            merged.append(("user", f"{system}{SEPARATOR}{content}"))
            merged_system = True
            continue
        merged.append((role, content))
    if not merged_system:
        return None
    return merged


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
    rendered = apply_chat_template(tokenizer, messages)
    if rendered is None:
        return RenderedPrompt(render_transcript(messages), add_special_tokens=True)
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
    messages: list[ChatTurn] = [("user", prompt)]
    if system:
        messages.insert(0, ("system", system))
    rendered = apply_chat_template(tokenizer, messages)
    if rendered is None:
        # No template: the system message is joined with the prompt, the way
        # Ollama does it for a model without a template.
        if system:
            return RenderedPrompt(f"{system}{SEPARATOR}{prompt}", add_special_tokens=True)
        return RenderedPrompt(prompt, add_special_tokens=True)
    return RenderedPrompt(rendered, add_special_tokens=False)
