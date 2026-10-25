"""Exception handlers translating engine errors into Ollama style responses.

Ollama answers an unknown model with ``404`` and a JSON body of the shape
``{"error": "..."}``; the same format is used here so that Ollama clients can
handle the errors without special casing this service.

The ``/v1/*`` routes mirror the OpenAI API and answer with its error format
instead — an OpenAI client parses ``{"error": {"message": ...}}``.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# The routes that answer in the OpenAI format.
OPENAI_PREFIX = "/v1/"


async def model_mismatch_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer an unknown model with 404, like Ollama does."""
    return JSONResponse(status_code=404, content={"error": str(exc)})


async def model_initialization_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer a model that cannot be loaded with 503."""
    return JSONResponse(status_code=503, content={"error": str(exc)})


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer an invalid request body with 400, like Ollama does."""
    errors = cast("RequestValidationError", exc).errors()
    details = "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}"
        for error in errors
    )
    message = details or "invalid request"
    if request.url.path.startswith(OPENAI_PREFIX):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )
    return JSONResponse(status_code=400, content={"error": message})
