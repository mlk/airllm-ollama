"""OpenAI compatible API routes.

Ollama exposes the same routes, so that the official OpenAI client can be
pointed at this service with ``base_url="http://host:port/v1"``. The service
serves a single model, so the list has exactly one entry.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime
from typing import cast
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from airllm_ollama.config import get_settings
from airllm_ollama.engine import AirLLMEngine, get_engine, model_names_match
from airllm_ollama.model_metadata import collect_metadata
from airllm_ollama.options import StopSequenceFilter, apply_stop_sequences, map_options
from airllm_ollama.options import done_reason as generation_done_reason
from airllm_ollama.schemas.generate import GenerateOptions
from airllm_ollama.schemas.openai import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    Delta,
    ErrorDetail,
    ModelCard,
    ModelList,
    Usage,
)

openai_router = APIRouter(prefix="/v1", tags=["openai"])

# The OpenAI API reports the owner of a model; this service owns the model it
# serves.
OWNED_BY = "airllm"

# The media type of a streamed answer, as the OpenAI API uses it.
SSE_MEDIA_TYPE = "text/event-stream"

# The last event of a streamed answer.
DONE = "data: [DONE]\n\n"


class OpenAIError(Exception):
    """An error that is answered in the OpenAI format."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: The message of the error.
            status_code: The HTTP status code of the answer.
            error_type: The type of the error, e.g. ``invalid_request_error``.
            param: The request parameter the error is about.
            code: The machine readable code of the error.
        """
        super().__init__(message)
        self.status_code = status_code
        self.detail = ErrorDetail(message=message, type=error_type, param=param, code=code)


async def openai_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer an error in the OpenAI format."""
    error = cast("OpenAIError", exc)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": error.detail.model_dump()},
    )


@openai_router.get("/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List the models the service serves, as ``GET /v1/models`` does in OpenAI."""
    return ModelList(data=[_model_card()])


@openai_router.get("/models/{model_id:path}", response_model=ModelCard)
async def retrieve_model(model_id: str) -> ModelCard:
    """Return a single model, as ``GET /v1/models/{model_id}`` does in OpenAI.

    The id of a model may contain a slash (a Hugging Face repository id), so
    the path converter takes the rest of the path.

    Args:
        model_id: The model the client asked for.

    Returns:
        The card of the served model.

    Raises:
        OpenAIError: If the client asked for another model.
    """
    settings = get_settings()
    if not model_names_match(model_id, settings.model_name):
        raise OpenAIError(
            f"The model '{model_id}' does not exist",
            status_code=404,
            param="model",
            code="model_not_found",
        )
    return _model_card()


def _model_card() -> ModelCard:
    """Return the card of the served model."""
    settings = get_settings()
    metadata = collect_metadata(settings.resolved_model_path)
    return ModelCard(
        id=settings.model_name,
        created=_created(metadata.modified_at),
        owned_by=OWNED_BY,
    )


def _created(modified_at: str) -> int:
    """Return the Unix timestamp of an ISO timestamp.

    Args:
        modified_at: The timestamp of the checkpoint, as the metadata reports it.

    Returns:
        The timestamp in seconds since the epoch.
    """
    return int(datetime.fromisoformat(modified_at).timestamp())


@openai_router.post("/chat/completions", response_model=ChatCompletion)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletion | StreamingResponse:
    """Answer a chat, as ``POST /v1/chat/completions`` does in OpenAI.

    The messages are rendered with the chat template of the model, the same way
    ``/api/chat`` renders them. A request with ``stream`` set is answered with
    Server-Sent Events, like the OpenAI API does.

    Args:
        request: The request of the client.

    Returns:
        The completion, or a stream of the chunks of the completion.

    Raises:
        OpenAIError: If the client asked for another model.
    """
    settings = get_settings()
    if not model_names_match(request.model, settings.model_name):
        raise OpenAIError(
            f"Model '{request.model}' not found. Server only supports '{settings.model_name}'.",
            status_code=400,
            param="model",
            code="model_not_found",
        )
    stop = _stop_sequences(request.stop)
    options = map_options(
        GenerateOptions(
            temperature=request.temperature,
            num_predict=request.max_tokens,
            stop=stop,
        )
    )
    messages = [(message.role, message.content) for message in request.messages]
    if request.stream:
        return StreamingResponse(
            _chat_events(get_engine(), messages, request.model, options, stop),
            media_type=SSE_MEDIA_TYPE,
        )
    result = await run_in_threadpool(
        get_engine().generate_chat, messages, model=request.model, **options
    )
    text = apply_stop_sequences(result.text, stop)
    return ChatCompletion(
        id=_completion_id(),
        created=int(time.time()),
        model=settings.model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(role="assistant", content=text),
                finish_reason=generation_done_reason(result, options),
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.generated_tokens,
            total_tokens=result.prompt_tokens + result.generated_tokens,
        ),
    )


def _chat_events(
    engine: AirLLMEngine,
    messages: list[tuple[str, str]],
    model: str,
    options: dict[str, object],
    stop: list[str],
) -> Iterator[str]:
    """Yield the Server-Sent Events of a chat completion.

    Args:
        engine: The engine that runs the generation.
        messages: The chat history as ``(role, content)`` pairs.
        model: The model the client asked for.
        options: The translated generation options.
        stop: The stop sequences requested by the client.

    Yields:
        The events of the stream, the last one being ``[DONE]``.
    """
    completion_id = _completion_id()
    created = int(time.time())
    stop_filter = StopSequenceFilter(stop)
    first = True
    for chunk in engine.generate_chat_stream(messages, model=model, **options):
        if chunk.result is not None:
            # The generation is over: the filter releases the tail it held
            # back, it cannot become a stop sequence any more.
            yield from _content_event(completion_id, created, model, stop_filter.flush())
            yield _chunk_event(
                completion_id,
                created,
                model,
                finish_reason=generation_done_reason(chunk.result, options),
            )
            yield DONE
            return
        text = stop_filter.feed(chunk.text)
        if text:
            yield _chunk_event(
                completion_id,
                created,
                model,
                content=text,
                role="assistant" if first else None,
            )
            first = False
        if stop_filter.stopped:
            # The stop sequence was found: the answer is complete, but the
            # client still expects the last chunk.
            yield _chunk_event(completion_id, created, model, finish_reason="stop")
            yield DONE
            return
    yield from _content_event(completion_id, created, model, stop_filter.flush())
    yield DONE


def _content_event(completion_id: str, created: int, model: str, content: str) -> Iterator[str]:
    """Yield the event of a piece of text, if there is one.

    Args:
        completion_id: The id of the completion.
        created: The timestamp of the completion.
        model: The model the client asked for.
        content: The text of the chunk.

    Yields:
        The event of the text, nothing when the text is empty.
    """
    if not content:
        return
    yield _chunk_event(completion_id, created, model, content=content)


def _chunk_event(
    completion_id: str,
    created: int,
    model: str,
    content: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
) -> str:
    """Render a chunk of the completion as a Server-Sent Event.

    Args:
        completion_id: The id of the completion.
        created: The timestamp of the completion.
        model: The model the client asked for.
        content: The text of the chunk.
        role: The role of the answer, in the first chunk.
        finish_reason: The reason the answer ended, in the last chunk.

    Returns:
        The event, ready to be written to the stream.
    """
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=Delta(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def _completion_id() -> str:
    """Return the id of a completion."""
    return f"chatcmpl-airllm-{uuid4().hex[:12]}"


def _stop_sequences(stop: str | list[str] | None) -> list[str]:
    """Return the stop sequences of a request.

    Args:
        stop: The stop sequences as the client sent them: one or a list.

    Returns:
        The stop sequences, without the empty ones.
    """
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [sequence for sequence in stop if sequence]
