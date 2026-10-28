"""Tests for the endpoints of the Ollama API that this service does not implement.

The service serves a single, pre-configured model and generates text, so the
endpoints that would manage a model registry or produce embeddings answer with
"not implemented" instead of "not found".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# The endpoints of Ollama that manage models, with the method they use.
MODEL_ENDPOINTS = [
    ("post", "/api/pull"),
    ("post", "/api/push"),
    ("post", "/api/create"),
    ("post", "/api/copy"),
    ("delete", "/api/delete"),
]

# The endpoints of Ollama that produce embeddings.
EMBEDDING_ENDPOINTS = ["/api/embed", "/api/embeddings"]


@pytest.mark.parametrize(("method", "path"), MODEL_ENDPOINTS)
def test_a_model_management_endpoint_is_not_implemented(
    client: TestClient, method: str, path: str
) -> None:
    """The endpoints exist and answer with a clear error."""
    response = client.request(method.upper(), path, json={"model": "some-model"})

    assert response.status_code == 501
    assert "error" in response.json()


@pytest.mark.parametrize(("method", "path"), MODEL_ENDPOINTS)
def test_a_model_management_endpoint_explains_itself(
    client: TestClient, method: str, path: str
) -> None:
    """The error names the model the service serves."""
    response = client.request(method.upper(), path, json={"model": "some-model"})

    error = response.json()["error"]
    assert "single" in error
    assert "AIRLLM_OLLAMA_MODEL_NAME" in error


def test_the_model_management_endpoints_are_documented(client: TestClient) -> None:
    """The endpoints are part of the OpenAPI schema."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/pull" in paths
    assert "/api/push" in paths
    assert "/api/create" in paths
    assert "/api/copy" in paths
    assert "/api/delete" in paths
    assert "501" in paths["/api/pull"]["post"]["responses"]


@pytest.mark.parametrize(("method", "path"), MODEL_ENDPOINTS)
def test_the_error_of_the_endpoints_is_documented(
    client: TestClient, method: str, path: str
) -> None:
    """The shape of the error is part of the OpenAPI schema."""
    schema = client.get("/openapi.json").json()

    response = schema["paths"][path][method.lower()]["responses"]["501"]
    reference = response["content"]["application/json"]["schema"]["$ref"]
    assert reference == "#/components/schemas/ErrorBody"


@pytest.mark.parametrize("path", EMBEDDING_ENDPOINTS)
def test_an_embedding_endpoint_is_not_implemented(client: TestClient, path: str) -> None:
    """The service generates text, it does not produce embeddings."""
    response = client.post(path, json={"model": "some-model", "input": "a text"})

    assert response.status_code == 501
    assert response.json() == {
        "error": "This service generates text; it does not produce embeddings."
    }


def test_the_embedding_endpoints_are_documented(client: TestClient) -> None:
    """The endpoints are part of the OpenAPI schema."""
    paths = client.get("/openapi.json").json()["paths"]

    for path in EMBEDDING_ENDPOINTS:
        assert path in paths
        response = paths[path]["post"]["responses"]["501"]
        reference = response["content"]["application/json"]["schema"]["$ref"]
        assert reference == "#/components/schemas/ErrorBody"


def test_the_ollama_endpoints_still_work(client: TestClient) -> None:
    """The stubs do not disturb the endpoints that work."""
    response = client.get("/api/tags")

    assert response.status_code == 200
    assert "models" in response.json()


def test_a_blob_cannot_be_pushed(client: TestClient) -> None:
    """The service stores no files."""
    response = client.post("/api/blobs/sha256:0123", content=b"a file")

    assert response.status_code == 501
    assert response.json() == {
        "error": (
            "This service serves a single, pre-configured model; "
            "it cannot push the files of a model."
        )
    }


def test_a_blob_cannot_be_checked(client: TestClient) -> None:
    """A HEAD answer has no body, so the status code is the whole answer."""
    response = client.head("/api/blobs/sha256:0123")

    assert response.status_code == 501
    assert response.text == ""


def test_the_blob_endpoints_are_documented(client: TestClient) -> None:
    """The endpoints are part of the OpenAPI schema."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/blobs/{digest}" in paths
    for method in ("post", "head"):
        response = paths["/api/blobs/{digest}"][method]["responses"]["501"]
        reference = response["content"]["application/json"]["schema"]["$ref"]
        assert reference == "#/components/schemas/ErrorBody"
