"""Runtime configuration for the AirLLM Ollama wrapper."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from airllm_ollama.version import __version__

ENV_PREFIX = "AIRLLM_OLLAMA_"

# Smallest possible download: a 4 MB randomly initialised Llama checkpoint.
# It is enough to exercise the whole request path; swap it for a real model
# (see README "Choosing a model") when you need meaningful completions.
DEFAULT_MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"


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

    # Name advertised through the Ollama API; also the only model the server serves.
    model_name: str = DEFAULT_MODEL_NAME
    # Local checkpoint path. Empty means "download <model_name> from Hugging Face".
    model_path: str = ""
    # Device handed to AirLLM (ignored by the MLX backend on macOS).
    device: str = "cuda:0"
    # Maximum context length used by AirLLM.
    max_seq_len: int = 512
    # Where AirLLM stores the per-layer shards it splits the checkpoint into.
    # Empty means "next to the checkpoint"; set it to keep the cache small.
    layer_shards_saving_path: str = ""
    # Default lifetime of the loaded model, e.g. "5m". Empty means the model
    # stays in memory until the process ends.
    keep_alive: str = ""

    @property
    def resolved_model_path(self) -> str:
        """Return the local checkpoint path, falling back to the model name."""
        return self.model_path or self.model_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
