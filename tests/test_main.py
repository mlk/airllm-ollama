"""Tests for the FastAPI application factory and entry point."""

from __future__ import annotations

import sys
from importlib.metadata import version as distribution_version

import pytest
from fastapi import FastAPI

from airllm_ollama import __version__
from airllm_ollama.__main__ import main
from airllm_ollama.config import Settings, get_settings
from airllm_ollama.main import app, create_app


def test_package_version_matches_distribution_metadata() -> None:
    assert distribution_version("airllm-ollama") == __version__


def test_the_version_is_written_by_the_build() -> None:
    """The build writes the version of the tags into the package."""
    from airllm_ollama import _version

    assert _version.__version__ == __version__


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

    # The arguments are passed explicitly: without them the entry point would
    # parse the arguments of the pytest process.
    main([])

    assert calls == [
        {
            "host": "0.0.0.0",
            "port": 9999,
            "log_level": "info",
        }
    ]


def test_entrypoint_parses_the_arguments_it_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["pytest", "--not-an-option"])

    assert main([]) == 0
    assert len(calls) == 1


def test_entrypoint_takes_the_model_options_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command line wins over the environment variables."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "from-the-environment")
    monkeypatch.setenv("AIRLLM_OLLAMA_DEVICE", "cpu")

    main(
        [
            "--model",
            "from-the-command-line",
            "--model-path",
            "/models/checkpoint",
            "--device",
            "cuda:0",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
        ]
    )

    assert calls == [
        {
            "host": "0.0.0.0",
            "port": 9999,
            "log_level": "info",
        }
    ]
    settings = get_settings()
    assert settings.model_name == "from-the-command-line"
    assert settings.model_path == "/models/checkpoint"
    assert settings.device == "cuda:0"


def test_entrypoint_keeps_the_environment_without_the_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An argument that was not given does not override the environment."""
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "from-the-environment")

    main([])

    assert get_settings().model_name == "from-the-environment"


def test_entrypoint_takes_the_capabilities_as_an_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capabilities of the served model are an argument as well."""
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    main(["--capabilities", "decision"])

    assert get_settings().capabilities == ["decision"]
