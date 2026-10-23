"""Schema for the health check endpoint."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Payload returned by ``GET /health``."""

    status: Literal["ok"] = "ok"
    version: str
    model: str
