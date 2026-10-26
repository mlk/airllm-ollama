"""Schema for the Ollama ``/api/ps`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from airllm_ollama.schemas.tags import ModelDetails


class RunningModel(BaseModel):
    """A model that is currently loaded into memory."""

    name: str
    model: str
    size: int = 0
    digest: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    expires_at: str = ""
    size_vram: int = 0


class PsResponse(BaseModel):
    """The models that are currently loaded."""

    models: list[RunningModel] = Field(default_factory=list)
