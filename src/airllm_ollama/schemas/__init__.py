"""Pydantic request/response schemas."""

from airllm_ollama.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from airllm_ollama.schemas.generate import GenerateOptions, GenerateRequest, GenerateResponse
from airllm_ollama.schemas.health import HealthResponse
from airllm_ollama.schemas.openai import ErrorDetail, ErrorResponse, ModelCard, ModelList
from airllm_ollama.schemas.tags import ModelDetails, ModelInfo, TagsResponse
from airllm_ollama.schemas.version import VersionResponse

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ErrorDetail",
    "ErrorResponse",
    "GenerateOptions",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "ModelCard",
    "ModelDetails",
    "ModelInfo",
    "ModelList",
    "TagsResponse",
    "VersionResponse",
]
