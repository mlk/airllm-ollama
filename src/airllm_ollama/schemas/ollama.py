"""Schema of the errors of the Ollama compatible endpoints.

Ollama answers an error with a JSON body of the shape ``{"error": "..."}``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """The error payload of the Ollama API.

    The OpenAI API nests the message in an object, this one does not — see
    ``airllm_ollama.schemas.openai.ErrorResponse``.
    """

    error: str


class SystemOneRequest(BaseModel):
    """The request payload of ``POST /v1/systemone``."""

    model: str
    # A non-empty string, or an object or an array that is serialized as JSON
    # text. It is not a chat history: the questions are asked about it.
    state: str | dict[str, Any] | list[Any]
    # Base64 encoded images, shared by all the questions. They need a model
    # with vision.
    images: list[str] = Field(default_factory=list)
    questions: dict[str, SystemOneQuestion]
    keep_alive: str | float | None = None


class SystemOneQuestion(BaseModel):
    """A question about the state of a request."""

    type: str
    instructions: str
    # The options of a choice, the two outcomes of a noul or the levels of a
    # score.
    criteria: dict[str, str | None] | list[str] | None = None


class SystemOneResponse(BaseModel):
    """The answers of the questions of a request."""

    model: str
    answers: dict[str, dict[str, Any]]
    usage: SystemOneUsage


class SystemOneUsage(BaseModel):
    """The tokens a decision used."""

    input_tokens: int = 0
    output_tokens: int = 0
