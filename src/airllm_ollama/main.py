"""FastAPI application factory for the AirLLM Ollama wrapper."""

from fastapi import FastAPI

from airllm_ollama.api.routes import api_router
from airllm_ollama.config import Settings, get_settings

DESCRIPTION = (
    "Ollama-compatible REST API backed by AirLLM layer-wise inference. "
    "Point any Ollama client at this service to run a single large model "
    "on constrained hardware."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        settings: Optional settings override; defaults to the process settings.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    resolved = settings or get_settings()
    application = FastAPI(
        title=resolved.app_name,
        version=resolved.version,
        description=DESCRIPTION,
    )
    application.include_router(api_router)
    return application


app = create_app()
