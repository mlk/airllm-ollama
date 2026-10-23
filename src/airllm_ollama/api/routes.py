"""API route definitions for the AirLLM Ollama wrapper."""

from fastapi import APIRouter

from airllm_ollama.config import get_settings
from airllm_ollama.schemas.health import HealthResponse

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Report service liveness and the configured model."""
    settings = get_settings()
    return HealthResponse(version=settings.version, model=settings.model_name)
