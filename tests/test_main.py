"""Tests for the FastAPI application factory and entry point."""

from __future__ import annotations

from importlib.metadata import version as distribution_version

import pytest
from fastapi import FastAPI

from airllm_ollama import __version__
from airllm_ollama.__main__ import main
from airllm_ollama.config import Settings
from airllm_ollama.main import app, create_app


def test_package_version_matches_distribution_metadata() -> None:
    assert distribution_version("airllm-ollama") == __version__


def test_create_app_returns_fastapi_instance() -> None:
    application = create_app()

    assert isinstance(application, FastAPI)


def test_create_app_uses_provided_settings() -> None:
    application = create_app(Settings(app_name="custom-service", version="9.9.9"))

    assert application.title == "custom-service"
    assert application.version == "9.9.9"


def test_create_app_defaults_to_process_settings() -> None:
    application = create_app()

    assert application.title == "airllm-ollama"
    assert application.version == __version__


def test_module_level_app_is_configured() -> None:
    assert isinstance(app, FastAPI)
    assert app.title == "airllm-ollama"


def test_entrypoint_runs_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("AIRLLM_OLLAMA_HOST", "0.0.0.0")
    monkeypatch.setenv("AIRLLM_OLLAMA_PORT", "9999")

    main()

    assert calls == [
        {
            "host": "0.0.0.0",
            "port": 9999,
            "log_level": "info",
        }
    ]
