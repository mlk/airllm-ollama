"""Schema for the Ollama ``/api/show`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from airllm_ollama.schemas.tags import CAPABILITIES, ModelDetails


class ShowRequest(BaseModel):
    """The request payload of ``POST /api/show``."""

    model: str
    verbose: bool = False


class ShowResponse(BaseModel):
    """The information about a model, as ``POST /api/show`` reports it."""

    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails = Field(default_factory=ModelDetails)
    model_info: dict[str, object] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=lambda: list(CAPABILITIES))
