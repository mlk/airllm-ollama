"""Tests for the error handlers of the API layer."""

from __future__ import annotations

from fastapi import Request

from airllm_ollama.api.errors import error_body, is_openai_route


def _request(path: str) -> Request:
    """Return a request for the given path.

    Args:
        path: The path of the request.

    Returns:
        The request.
    """
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )


def test_the_openai_routes_are_recognised() -> None:
    """The format of the error depends on the route of the request."""
    assert is_openai_route(_request("/v1/chat/completions")) is True
    assert is_openai_route(_request("/v1")) is True
    assert is_openai_route(_request("/api/chat")) is False


def test_the_error_body_follows_the_format_of_the_route() -> None:
    """The Ollama routes answer with a string, the OpenAI ones with an object."""
    assert error_body(_request("/api/chat"), "broken") == {"error": "broken"}

    assert error_body(_request("/v1/chat/completions"), "broken") == {
        "error": {
            "message": "broken",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }
