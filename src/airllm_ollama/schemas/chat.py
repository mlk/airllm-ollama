"""Schemas for the Ollama ``/api/chat`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from airllm_ollama.schemas.generate import GenerateOptions
from airllm_ollama.schemas.tools import ToolCall, ToolDefinition


class ChatMessage(BaseModel):
    """A single message of a chat history.

    The role is free-form, like in Ollama: ``system``, ``user`` and
    ``assistant`` are the common ones, but a chat template may use others.
    Roles that the template does not know make the rendering fall back to the
    plain transcript.

    An assistant message that calls tools carries them in ``tool_calls``; its
    ``content`` is empty then. A ``tool`` role message carries the result of a
    call and names the tool it comes from in ``tool_name``.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None


class ChatRequest(BaseModel):
    """Request payload of Ollama's ``/api/chat``."""

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    format: str | None = None
    options: GenerateOptions | None = None
    keep_alive: str | None = None
    tools: list[ToolDefinition] | None = None


class ChatResponse(BaseModel):
    """Response payload of Ollama's ``/api/chat`` (non streaming)."""

    model: str
    created_at: str
    message: ChatMessage
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
