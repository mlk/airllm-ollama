"""Tests for the OpenAI compatible ``/v1/models`` endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from airllm_ollama.config import get_settings

MODEL = "test-model"


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Serve the test model."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", MODEL)
    get_settings.cache_clear()
    return MODEL


def test_list_models_returns_the_openai_shape(client: TestClient, model: str) -> None:
    """The list of the models mirrors the OpenAI API."""
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    card = payload["data"][0]
    assert card["id"] == model
    assert card["object"] == "model"
    assert card["owned_by"] == "airllm"
    assert isinstance(card["created"], int)
    assert card["created"] > 0


def test_list_models_reports_the_configured_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id of the model is the configured one."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "other/model")
    get_settings.cache_clear()

    payload = client.get("/v1/models").json()

    assert payload["data"][0]["id"] == "other/model"


def test_retrieve_model_returns_the_model(client: TestClient, model: str) -> None:
    """A single model is returned in the OpenAI shape."""
    response = client.get(f"/v1/models/{model}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == model
    assert payload["object"] == "model"
    assert payload["owned_by"] == "airllm"


def test_retrieve_model_accepts_a_tag(client: TestClient, model: str) -> None:
    """A tagged reference resolves to the served model, like Ollama's does."""
    response = client.get(f"/v1/models/{model}:latest")

    assert response.status_code == 200
    assert response.json()["id"] == model


def test_retrieve_model_accepts_a_name_with_a_slash(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hugging Face repository id contains a slash."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_NAME", "org/model")
    get_settings.cache_clear()

    response = client.get("/v1/models/org/model")

    assert response.status_code == 200
    assert response.json()["id"] == "org/model"


def test_retrieve_model_rejects_another_model(client: TestClient, model: str) -> None:
    """An unknown model is answered with the OpenAI error format."""
    response = client.get("/v1/models/other-model")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["message"] == "The model 'other-model' does not exist"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "model"
    assert payload["error"]["code"] == "model_not_found"


def test_the_ollama_endpoints_keep_working(client: TestClient, model: str) -> None:
    """The OpenAI routes do not disturb the Ollama ones."""
    response = client.get("/api/version")

    assert response.status_code == 200
    assert "version" in response.json()
