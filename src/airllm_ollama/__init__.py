"""AirLLM Ollama API wrapper: an Ollama-compatible API powered by AirLLM."""

from airllm_ollama.config import Settings, get_settings
from airllm_ollama.version import __version__

__all__ = ["Settings", "__version__", "get_settings"]
