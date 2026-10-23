"""Tests for the health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from airllm_ollama.config import get_settings


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == get_settings().version
    assert payload["model"] == get_settings().model_name


def test_health_reports_configured_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "airllm-model-v2")
    get_settings.cache_clear()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model"] == "airllm-model-v2"


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
