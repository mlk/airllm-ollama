"""Tests for the OpenAPI schema of the service.

The schema is part of the contract: a client that reads ``/openapi.json`` (or
the interactive documentation) has to see the shape of the answers, including
the errors.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# The endpoints that answer with an error, and the status codes they use.
GENERATION_ENDPOINTS = ["/api/generate", "/api/chat"]
GENERATION_ERRORS = ["400", "404", "503"]


@pytest.mark.parametrize("path", GENERATION_ENDPOINTS)
def test_the_errors_of_the_generation_endpoints_are_documented(
    client: TestClient, path: str
) -> None:
    """The errors of the generation endpoints are part of the OpenAPI schema."""
    schema = client.get("/openapi.json").json()

    responses = schema["paths"][path]["post"]["responses"]
    assert set(responses) >= set(GENERATION_ERRORS)
    for status in GENERATION_ERRORS:
        reference = responses[status]["content"]["application/json"]["schema"]["$ref"]
        assert reference == "#/components/schemas/ErrorBody"


def test_the_error_body_is_documented(client: TestClient) -> None:
    """The shape of the error of the Ollama endpoints is documented."""
    schema = client.get("/openapi.json").json()

    error_body = schema["components"]["schemas"]["ErrorBody"]
    assert error_body["type"] == "object"
    assert error_body["properties"]["error"]["type"] == "string"
    assert error_body["required"] == ["error"]
