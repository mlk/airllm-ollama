"""OpenAI compatible API routes.

Ollama exposes the same routes, so that the official OpenAI client can be
pointed at this service with ``base_url="http://host:port/v1"``. The service
serves a single model, so the list has exactly one entry.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from airllm_ollama.config import get_settings
from airllm_ollama.engine import model_names_match
from airllm_ollama.model_metadata import collect_metadata
from airllm_ollama.schemas.openai import ErrorDetail, ModelCard, ModelList

openai_router = APIRouter(prefix="/v1", tags=["openai"])

# The OpenAI API reports the owner of a model; this service owns the model it
# serves.
OWNED_BY = "airllm"


class OpenAIError(Exception):
    """An error that is answered in the OpenAI format."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: The message of the error.
            status_code: The HTTP status code of the answer.
            error_type: The type of the error, e.g. ``invalid_request_error``.
            param: The request parameter the error is about.
            code: The machine readable code of the error.
        """
        super().__init__(message)
        self.status_code = status_code
        self.detail = ErrorDetail(message=message, type=error_type, param=param, code=code)


async def openai_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer an error in the OpenAI format."""
    error = cast("OpenAIError", exc)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": error.detail.model_dump()},
    )


@openai_router.get("/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List the models the service serves, as ``GET /v1/models`` does in OpenAI."""
    return ModelList(data=[_model_card()])


@openai_router.get("/models/{model_id:path}", response_model=ModelCard)
async def retrieve_model(model_id: str) -> ModelCard:
    """Return a single model, as ``GET /v1/models/{model_id}`` does in OpenAI.

    The id of a model may contain a slash (a Hugging Face repository id), so
    the path converter takes the rest of the path.

    Args:
        model_id: The model the client asked for.

    Returns:
        The card of the served model.

    Raises:
        OpenAIError: If the client asked for another model.
    """
    settings = get_settings()
    if not model_names_match(model_id, settings.model_name):
        raise OpenAIError(
            f"The model '{model_id}' does not exist",
            status_code=404,
            param="model",
            code="model_not_found",
        )
    return _model_card()


def _model_card() -> ModelCard:
    """Return the card of the served model."""
    settings = get_settings()
    metadata = collect_metadata(settings.resolved_model_path)
    return ModelCard(
        id=settings.model_name,
        created=_created(metadata.modified_at),
        owned_by=OWNED_BY,
    )


def _created(modified_at: str) -> int:
    """Return the Unix timestamp of an ISO timestamp.

    Args:
        modified_at: The timestamp of the checkpoint, as the metadata reports it.

    Returns:
        The timestamp in seconds since the epoch.
    """
    return int(datetime.fromisoformat(modified_at).timestamp())
