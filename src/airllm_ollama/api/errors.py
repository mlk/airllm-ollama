"""Exception handlers translating engine errors into Ollama style responses.

Ollama answers an unknown model with ``404`` and a JSON body of the shape
``{"error": "..."}``; the same format is used here so that Ollama clients can
handle the errors without special casing this service.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


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
    return JSONResponse(status_code=400, content={"error": details or "invalid request"})
