"""Shared pytest fixtures for the AirLLM Ollama wrapper test suite."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from airllm_ollama.config import Settings, get_settings
from airllm_ollama.main import create_app

DEFAULT_PARAMETERS = 135_000_000


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Ensure each test observes a freshly resolved settings object."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Return default settings for the current test."""
    return get_settings()


@pytest.fixture
def app() -> FastAPI:
    """Return a fresh FastAPI application instance."""
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a ``TestClient`` bound to the application under test."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def write_checkpoint() -> Callable[..., Path]:
    """Return a factory that creates a minimal but realistic checkpoint."""

    def factory(
        directory: Path,
        *,
        parameters: int = DEFAULT_PARAMETERS,
        architecture: str = "LlamaForCausalLM",
        model_type: str = "llama",
    ) -> Path:
        """Create a checkpoint directory with a config and a safetensors file.

        Args:
            directory: Directory to create the checkpoint in.
            parameters: Parameter count written into the safetensors header.
            architecture: Architecture name written into ``config.json``.
            model_type: Model type written into ``config.json``.

        Returns:
            The directory holding the checkpoint.
        """
        directory.mkdir(parents=True, exist_ok=True)
        config: dict[str, object] = {"architectures": [architecture]}
        if model_type:
            config["model_type"] = model_type
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

        header = {
            "__metadata__": {"format": "pt", "total_parameters": parameters, "total_size": 8},
            "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }
        blob = json.dumps(header).encode("utf-8")
        weights = directory / "model.safetensors"
        weights.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 8)
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
        return directory

    return factory
