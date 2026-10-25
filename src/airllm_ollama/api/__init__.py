"""HTTP routing layer."""

from airllm_ollama.api.openai import openai_router
from airllm_ollama.api.routes import api_router

__all__ = ["api_router", "openai_router"]
