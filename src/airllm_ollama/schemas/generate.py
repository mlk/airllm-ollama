"""Schemas for the Ollama ``/api/generate`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateOptions(BaseModel):
    """Sampling options, mirroring Ollama's ``options`` object.

    Only a subset of the options can be applied to a Hugging Face model. The
    unsupported ones (presence/frequency penalties, Mirostat, ...) are accepted
    and ignored, so that clients sending the full Ollama option set keep
    working.
    """

    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    tfs_z: float | None = None
    num_predict: int | None = None
    num_ctx: int | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Request payload of Ollama's ``/api/generate``."""

    model: str
    prompt: str = ""
    system: str | None = None
    suffix: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = False
    raw: bool = False
    format: str | None = None
    keep_alive: str | None = None
    options: GenerateOptions | None = None


class GenerateResponse(BaseModel):
    """Response payload of Ollama's ``/api/generate`` (non streaming)."""

    model: str
    created_at: str
    response: str
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
