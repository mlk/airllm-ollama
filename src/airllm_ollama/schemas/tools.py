"""Tool call schemas, in the shape Ollama uses.

The tool calls of a request and the ones of a response share the same
shape: a list of ``{"function": {"name": ..., "arguments": {...}}}`` objects.
The arguments are a JSON **object** — unlike the OpenAI API, which encodes
them as a JSON string.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    """The definition of the function a tool call invokes."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """A tool the client offers to the model."""

    type: str = "function"
    function: ToolFunction


class ToolCallFunction(BaseModel):
    """The function a tool call asks to invoke."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A single tool call of the assistant."""

    function: ToolCallFunction
