"""Shared pytest fixtures for the AirLLM Ollama wrapper test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from airllm_ollama.config import Settings, get_settings
from airllm_ollama.main import create_app


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
