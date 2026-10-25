"""Schemas for the OpenAI compatible endpoints.

The shapes mirror the OpenAI API, so that the official client can talk to this
service with ``base_url="http://host:port/v1"``.
"""

from __future__ import annotations

from pydantic import BaseModel


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
