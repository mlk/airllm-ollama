"""Schemas for the OpenAI compatible endpoints.

The shapes mirror the OpenAI API, so that the official client can talk to this
service with ``base_url="http://host:port/v1"``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    """A model, as the OpenAI API describes it."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "airllm"


class ModelList(BaseModel):
    """The models the service serves."""

    object: str = "list"
    data: list[ModelCard]


class ErrorDetail(BaseModel):
    """The error object of the OpenAI API."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """The error payload of the OpenAI API."""

    error: ErrorDetail


class ChatMessage(BaseModel):
    """A message of a chat, as the OpenAI API describes it."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """The request payload of ``POST /v1/chat/completions``."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    # The newer name of `max_tokens`; the OpenAI SDK sends it for the models
    # that do not count the reasoning tokens.
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False


class AssistantMessage(BaseModel):
    """The message the assistant answers with."""

    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A choice of a chat completion."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """The tokens a completion used."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    """The response of ``POST /v1/chat/completions``."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class Delta(BaseModel):
    """The part of the answer a streamed chunk carries."""

    role: str | None = None
    content: str = ""


class ChatCompletionChunkChoice(BaseModel):
    """A choice of a streamed chat completion."""

    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """One chunk of a streamed chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)
