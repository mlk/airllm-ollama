"""Schemas for the Ollama ``/api/chat`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from airllm_ollama.schemas.generate import GenerateOptions


class ChatMessage(BaseModel):
    """A single message of a chat history."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request payload of Ollama's ``/api/chat``."""

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    format: str | None = None
    options: GenerateOptions | None = None
    keep_alive: str | None = None


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
