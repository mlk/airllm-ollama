"""Pydantic request/response schemas."""

from airllm_ollama.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from airllm_ollama.schemas.generate import GenerateOptions, GenerateRequest, GenerateResponse
from airllm_ollama.schemas.health import HealthResponse
from airllm_ollama.schemas.tags import ModelDetails, ModelInfo, TagsResponse
from airllm_ollama.schemas.version import VersionResponse

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "GenerateOptions",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "ModelDetails",
    "ModelInfo",
    "TagsResponse",
    "VersionResponse",
]
