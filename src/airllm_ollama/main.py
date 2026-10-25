"""FastAPI application factory for the AirLLM Ollama wrapper."""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from airllm_ollama.api.errors import (
    model_initialization_handler,
    model_mismatch_handler,
    validation_error_handler,
)
from airllm_ollama.api.openai import OpenAIError, openai_error_handler, openai_router
from airllm_ollama.api.routes import api_router
from airllm_ollama.config import Settings, get_settings
from airllm_ollama.exceptions import ModelInitializationError, ModelMismatchError

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
    application.include_router(openai_router)
    application.add_exception_handler(ModelMismatchError, model_mismatch_handler)
    application.add_exception_handler(ModelInitializationError, model_initialization_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.add_exception_handler(OpenAIError, openai_error_handler)
    return application


app = create_app()
