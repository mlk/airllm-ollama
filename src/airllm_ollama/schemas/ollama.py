"""Schema of the errors of the Ollama compatible endpoints.

Ollama answers an error with a JSON body of the shape ``{"error": "..."}``.
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorBody(BaseModel):
    """The error payload of the Ollama API.

    The OpenAI API nests the message in an object, this one does not — see
    ``airllm_ollama.schemas.openai.ErrorResponse``.
    """

    error: str
