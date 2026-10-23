"""Runtime configuration for the AirLLM Ollama wrapper."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from airllm_ollama.version import __version__

ENV_PREFIX = "AIRLLM_OLLAMA_"


class Settings(BaseSettings):
    """Application settings loaded from the environment.

    All values can be overridden with ``AIRLLM_OLLAMA_``-prefixed environment
    variables, e.g. ``AIRLLM_OLLAMA_MODEL_NAME=my-model``.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "airllm-ollama-proxy"
    version: str = __version__
    host: str = "127.0.0.1"
    port: int = 11434
    log_level: str = "INFO"
    model_name: str = "airllm-model"
    model_path: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
