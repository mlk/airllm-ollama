"""Integration tests for the Ollama compatible metadata endpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from airllm_ollama.config import get_settings
from airllm_ollama.schemas.tags import CAPABILITIES


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


def test_root_announces_a_running_service(client: TestClient, mocker: MockerFixture) -> None:
    engine = mocker.patch("airllm_ollama.api.routes.get_engine")
    engine.return_value.is_available = True

    response = client.get("/")

    assert response.status_code == 200
    assert response.text == "airllm-ollama is running"
    assert response.headers["content-type"].startswith("text/plain")


def test_root_announces_a_kinda_running_service(client: TestClient, mocker: MockerFixture) -> None:
    engine = mocker.patch("airllm_ollama.api.routes.get_engine")
    engine.return_value.is_available = False

    response = client.get("/")

    assert response.status_code == 200
    assert response.text == "airllm-ollama is kinda running"


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


def test_both_endpoints_report_the_same_capabilities(client: TestClient) -> None:
    """The capabilities of the model are reported in one place."""
    tags = client.get("/api/tags").json()["models"][0]["capabilities"]
    show = client.post("/api/show", json={"model": get_settings().model_name}).json()[
        "capabilities"
    ]

    assert tags == show == CAPABILITIES
    # The service generates text; the chat endpoint is a completion, Ollama has
    # no "chat" capability.
    assert CAPABILITIES == ["completion"]


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


def test_api_show_describes_the_served_model(
    client: TestClient, local_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint answers the information about the model."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_PATH", str(local_checkpoint))
    get_settings.cache_clear()

    response = client.post("/api/show", json={"model": get_settings().model_name})

    assert response.status_code == 200
    payload = response.json()
    assert payload["details"]["family"] == "llama"
    assert payload["details"]["families"] == ["llama"]
    assert payload["capabilities"] == ["completion"]
    assert "general.architecture" in payload["model_info"]
    assert f"FROM {local_checkpoint}" in payload["modelfile"]
    assert "PARAMETER num_ctx" in payload["modelfile"]
    assert "num_ctx" in payload["parameters"]


def test_api_show_rejects_another_model(client: TestClient) -> None:
    """Another model is answered with the Ollama error format."""
    response = client.post("/api/show", json={"model": "other-model"})

    assert response.status_code == 404
    assert "error" in response.json()


def test_api_ps_is_empty_while_the_model_is_not_loaded(client: TestClient) -> None:
    """The model is loaded lazily, so the list starts empty."""
    response = client.get("/api/ps")

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_api_ps_lists_the_loaded_model(
    client: TestClient,
    local_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """A loaded model is listed with its details."""
    monkeypatch.setenv("AIRLLM_OLLAMA_MODEL_PATH", str(local_checkpoint))
    get_settings.cache_clear()
    engine = mocker.patch("airllm_ollama.api.routes.get_engine")
    engine.return_value.is_loaded = True
    engine.return_value.unload_at = datetime(2026, 10, 25, 8, 0, tzinfo=UTC)

    response = client.get("/api/ps")

    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == 1
    model = models[0]
    assert model["name"] == get_settings().model_name
    assert model["details"]["family"] == "llama"
    assert model["expires_at"] == "2026-10-25T08:00:00Z"
    # AirLLM streams the model layer by layer, the weights are never in the
    # memory of the device at once.
    assert model["size_vram"] == 0
