"""Runtime configuration for the AirLLM Ollama wrapper."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from airllm_ollama.schemas.tags import CAPABILITIES
from airllm_ollama.version import __version__

ENV_PREFIX = "AIRLLM_OLLAMA_"

# Smallest possible download: a 4 MB randomly initialised Llama checkpoint.
# It is enough to exercise the whole request path; swap it for a real model
# (see README "Choosing a model") when you need meaningful completions.
DEFAULT_MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# The prompt the decision models are trained with: the state the question is
# about and the candidates the model chooses from. The newlines are written as
# "\n", so that the template stays a single line in the environment.
DEFAULT_DECISION_TEMPLATE = "Context: {state}\\nChoices: {choices}\\nSelected Option:"


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

    app_name: str = "airllm-ollama"
    version: str = __version__
    host: str = "127.0.0.1"
    port: int = 8000
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
    # Framing of the streaming responses: "ndjson" (like Ollama) or "sse".
    stream_format: str = "ndjson"
    # The prompt template of the decision models. The placeholders are the
    # state the question is about and the candidates the model chooses from;
    # the newlines are written as "\n".
    decision_template: str = DEFAULT_DECISION_TEMPLATE
    # The capabilities of the served model, as a comma separated list. The
    # service generates text, so a completion model is the default; a decision
    # model (the one of `POST /v1/systemone`) answers "decision".
    capabilities: Annotated[list[str], NoDecode] = Field(default_factory=lambda: list(CAPABILITIES))

    @field_validator("capabilities", mode="before")
    @classmethod
    def _split_capabilities(cls, value: object) -> object:
        """Accept the capabilities as a comma separated list.

        The environment holds a string (`AIRLLM_OLLAMA_CAPABILITIES=decision`),
        which is easier to write than a JSON list.

        Args:
            value: The value of the setting.

        Returns:
            The capabilities as a list.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                # A JSON list is accepted as well.
                try:
                    decoded = json.loads(text)
                except ValueError:
                    decoded = None
                if isinstance(decoded, list):
                    return decoded
            return [part.strip() for part in text.split(",") if part.strip()]
        return value

    @property
    def resolved_model_path(self) -> str:
        """Return the local checkpoint path, falling back to the model name."""
        return self.model_path or self.model_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def apply_overrides(overrides: Mapping[str, object]) -> None:
    """Apply the command line overrides to the environment.

    The settings are read from the environment, so the values of the command
    line have to reach it that way: they win over the environment and over the
    ``.env`` file. A value of ``None`` is ignored — it means that the argument
    was not given.

    The settings are cached, so this has to happen before they are read for
    the first time.

    Args:
        overrides: The values of the command line, keyed by the name of the
            setting (``model_name``, ``model_path``, ``device``, ``host``,
            ``port``).
    """
    for name, value in overrides.items():
        if value is None:
            continue
        os.environ[f"{ENV_PREFIX}{name.upper()}"] = str(value)
    get_settings.cache_clear()
