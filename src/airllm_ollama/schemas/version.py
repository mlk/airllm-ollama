"""Schema for the Ollama ``/api/version`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class VersionResponse(BaseModel):
    """Payload returned by ``GET /api/version``."""

    version: str
