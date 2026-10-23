"""Integration tests for the Ollama compatible metadata endpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from airllm_ollama.config import get_settings


@pytest.fixture
def local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_checkpoint: Callable[..., Path],
) -> Path:
    """Point the service at a local checkpoint directory."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_PATH", str(checkpoint))
    get_settings.cache_clear()
    return checkpoint


def test_root_reports_service_information(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "airllm-ollama-proxy"
    assert payload["version"] == get_settings().version
    assert payload["model"] == get_settings().model_name


def test_api_version_matches_the_server_version(client: TestClient) -> None:
    response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": get_settings().version}


def test_api_tags_lists_the_served_model(client: TestClient) -> None:
    response = client.get("/api/tags")

    assert response.status_code == 200
    payload = response.json()
    assert list(payload) == ["models"]
    assert len(payload["models"]) == 1
    model = payload["models"][0]
    assert model["name"] == get_settings().model_name
    assert model["model"] == get_settings().model_name


def test_api_tags_matches_the_ollama_schema(client: TestClient) -> None:
    response = client.get("/api/tags")

    model = response.json()["models"][0]
    assert set(model) == {
        "name",
        "model",
        "modified_at",
        "size",
        "digest",
        "details",
        "capabilities",
    }
    assert set(model["details"]) == {
        "parent_model",
        "format",
        "family",
        "families",
        "parameter_size",
        "quantization_level",
        "context_length",
    }
    assert isinstance(model["size"], int)
    assert isinstance(model["modified_at"], str)
    assert isinstance(model["digest"], str)
    assert model["capabilities"] == ["completion"]


def test_api_tags_reports_the_context_length(client: TestClient) -> None:
    response = client.get("/api/tags")

    assert response.json()["models"][0]["details"]["context_length"] == 512


def test_api_tags_reports_local_checkpoint_metadata(
    client: TestClient, local_checkpoint: Path
) -> None:
    response = client.get("/api/tags")

    model = response.json()["models"][0]
    assert model["size"] == sum(path.stat().st_size for path in local_checkpoint.rglob("*"))
    assert model["digest"].startswith("sha256:")
    assert model["details"]["format"] == "safetensors"
    assert model["details"]["family"] == "llama"
    assert model["details"]["families"] == ["llama"]
    assert model["details"]["parameter_size"] == "135M"


def test_api_tags_timestamp_is_rfc3339(client: TestClient) -> None:
    response = client.get("/api/tags")

    modified_at = response.json()["models"][0]["modified_at"]
    assert modified_at.endswith("Z")
    assert datetime.fromisoformat(modified_at.replace("Z", "+00:00")).tzinfo is not None


def test_api_tags_without_a_local_checkpoint(client: TestClient) -> None:
    """A model that has not been downloaded yet still has to be listed."""
    response = client.get("/api/tags")

    model = response.json()["models"][0]
    assert model["size"] == 0
    assert model["digest"] == ""
    assert model["details"]["family"] == ""


def test_openapi_documents_the_ollama_endpoints(client: TestClient) -> None:
    response = client.get("/openapi.json")

    paths = response.json()["paths"]
    assert {"/", "/health", "/api/version", "/api/tags"} <= set(paths)


def test_health_reports_the_model_state(client: TestClient, mocker: MockerFixture) -> None:
    engine = mocker.patch("airllm_ollama.api.routes.get_engine")
    engine.return_value.is_loaded = True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_defaults_to_an_unloaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False


def test_ollama_client_style_request(client: TestClient) -> None:
    """The payload has to survive a round trip through JSON unchanged."""
    response = client.get("/api/tags")

    assert json.loads(response.content) == response.json()
