"""Tests for runtime configuration loading."""

from __future__ import annotations

import pytest

from airllm_ollama.config import DEFAULT_MODEL_NAME, Settings, get_settings


def test_default_settings_are_applied() -> None:
    settings = Settings()

    assert settings.app_name == "airllm-ollama-proxy"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.model_name == DEFAULT_MODEL_NAME
    assert settings.model_path == ""
    assert settings.device == "cuda:0"
    assert settings.max_seq_len == 512


def test_resolved_model_path_falls_back_to_model_name() -> None:
    assert Settings(model_name="my-model").resolved_model_path == "my-model"
    assert Settings(model_name="my-model", model_path="/models/x").resolved_model_path == (
        "/models/x"
    )


def test_settings_are_loaded_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "custom-model")
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_PATH", "/models/custom")
    monkeypatch.setenv("AIRLLM_OLLAMA_PORT", "8080")

    settings = Settings()

    assert settings.model_name == "custom-model"
    assert settings.model_path == "/models/custom"
    assert settings.port == 8080


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_UNKNOWN_SETTING", "value")

    settings = Settings()

    assert not hasattr(settings, "unknown_setting")


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "changed")
    second = get_settings()

    assert first is second
    assert second.model_name == DEFAULT_MODEL_NAME
