"""API route definitions for the AirLLM Ollama wrapper.

The ``/api/*`` routes mirror Ollama's REST interface, so standard Ollama
clients can talk to this service unchanged.
"""

from fastapi import APIRouter

from airllm_ollama.config import get_settings
from airllm_ollama.engine import get_engine
from airllm_ollama.model_metadata import collect_metadata
from airllm_ollama.schemas.health import HealthResponse
from airllm_ollama.schemas.tags import ModelDetails, ModelInfo, TagsResponse
from airllm_ollama.schemas.version import RootResponse, VersionResponse

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Report service liveness and the configured model."""
    settings = get_settings()
    return HealthResponse(
        version=settings.version,
        model=settings.model_name,
        model_loaded=get_engine().is_loaded,
    )


@api_router.get("/", response_model=RootResponse, tags=["meta"])
async def root() -> RootResponse:
    """Return basic information about the running service."""
    settings = get_settings()
    return RootResponse(
        name=settings.app_name,
        version=settings.version,
        model=settings.model_name,
    )


@api_router.get("/api/version", response_model=VersionResponse, tags=["ollama"])
async def version() -> VersionResponse:
    """Return the server version, as ``GET /api/version`` does in Ollama."""
    return VersionResponse(version=get_settings().version)


@api_router.get("/api/tags", response_model=TagsResponse, tags=["ollama"])
async def tags() -> TagsResponse:
    """Return the single served model, as ``GET /api/tags`` does in Ollama."""
    settings = get_settings()
    metadata = collect_metadata(settings.resolved_model_path)
    return TagsResponse(
        models=[
            ModelInfo(
                name=settings.model_name,
                model=settings.model_name,
                modified_at=metadata.modified_at,
                size=metadata.size,
                digest=metadata.digest,
                details=ModelDetails(
                    format=metadata.format,
                    family=metadata.family,
                    families=list(metadata.families),
                    parameter_size=metadata.parameter_size,
                    context_length=settings.max_seq_len,
                ),
            )
        ]
    )
