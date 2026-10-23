"""Schema for the Ollama ``/api/tags`` endpoint.

The shapes mirror Ollama's own response so that standard Ollama clients can
consume them unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelDetails(BaseModel):
    """Details about a served model."""

    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] = Field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""
    context_length: int | None = None


class ModelInfo(BaseModel):
    """Metadata of a single served model."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails = Field(default_factory=ModelDetails)
    capabilities: list[str] = Field(default_factory=lambda: ["completion"])


class TagsResponse(BaseModel):
    """Payload returned by ``GET /api/tags``."""

    models: list[ModelInfo]
