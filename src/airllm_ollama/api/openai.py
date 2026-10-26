"""OpenAI compatible API routes.

Ollama exposes the same routes, so that the official OpenAI client can be
pointed at this service with ``base_url="http://host:port/v1"``. The service
serves a single model, so the list has exactly one entry.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from airllm_ollama.config import get_settings
from airllm_ollama.decision import (
    CHOICE,
    FALSE,
    NOUL,
    SCORE,
    TRUE,
    best,
    build_prompt,
    confidence,
    normalize,
    score,
)
from airllm_ollama.engine import AirLLMEngine, Decision, get_engine, model_names_match
from airllm_ollama.model_metadata import collect_metadata
from airllm_ollama.options import (
    StopSequenceFilter,
    apply_stop_sequences,
    map_options,
    parse_keep_alive,
)
from airllm_ollama.options import done_reason as generation_done_reason
from airllm_ollama.schemas.generate import GenerateOptions
from airllm_ollama.schemas.ollama import (
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneUsage,
)
from airllm_ollama.schemas.openai import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    CompletionChoice,
    CompletionChunkChoice,
    CompletionRequest,
    Delta,
    ErrorDetail,
    ModelCard,
    ModelList,
    TextCompletion,
    TextCompletionChunk,
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

# The status code of an endpoint this service does not implement.
NOT_IMPLEMENTED = 501


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


@openai_router.post("/embeddings", status_code=NOT_IMPLEMENTED, tags=["openai"])
async def embeddings() -> None:
    """Answer the embeddings of the inputs, as ``POST /v1/embeddings`` does in OpenAI.

    This service generates text — it does not produce embeddings. The endpoint
    exists so that a client gets a clear answer instead of a "not found".

    Raises:
        OpenAIError: Always — the endpoint is not implemented.
    """
    raise OpenAIError(
        "This service generates text; it does not produce embeddings.",
        status_code=NOT_IMPLEMENTED,
        code="not_implemented",
    )


@openai_router.post("/systemone", response_model=SystemOneResponse, tags=["openai"])
async def systemone(request: SystemOneRequest) -> SystemOneResponse:
    """Answer the questions of the request, as ``POST /v1/systemone`` does in Ollama.

    Every question is asked separately: the model is asked once per question,
    with the state and the candidates of that question, and the probabilities
    of the candidates become the answer.

    Args:
        request: The request of the client.

    Returns:
        The answers of the questions.

    Raises:
        OpenAIError: If the client asked for another model, if the request
            cannot be answered or if the model failed.
    """
    settings = get_settings()
    if not model_names_match(request.model, settings.model_name):
        raise OpenAIError(
            f"Model '{request.model}' not found. Server only supports '{settings.model_name}'.",
            status_code=400,
            param="model",
            code="model_not_found",
        )
    if request.images:
        raise OpenAIError(
            "This service cannot judge images; they need a checkpoint with a projector.",
            status_code=400,
            param="images",
            code="unsupported",
        )
    engine = get_engine()
    keep_alive = parse_keep_alive(
        str(request.keep_alive) if request.keep_alive is not None else settings.keep_alive
    )
    answers: dict[str, dict[str, Any]] = {}
    input_tokens = 0
    output_tokens = 0
    for name, question in request.questions.items():
        candidates = _candidates(question)
        prompt = build_prompt(
            settings.decision_template,
            _state(request.state, question.instructions),
            candidates,
        )
        decision = await run_in_threadpool(engine.decide, prompt, candidates, model=request.model)
        answers[name] = _answer(question, decision)
        input_tokens += decision.prompt_tokens
        output_tokens += 1
    if keep_alive is not None:
        engine.schedule_unload(keep_alive)
    return SystemOneResponse(
        model=settings.model_name,
        answers=answers,
        usage=SystemOneUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _candidates(question: SystemOneQuestion) -> list[str]:
    """Return the candidates of a question.

    Args:
        question: The question of the client.

    Returns:
        The candidates the model chooses from.

    Raises:
        OpenAIError: If the question is not one Ollama knows.
    """
    if question.type == CHOICE:
        criteria = question.criteria
        if not isinstance(criteria, dict) or not criteria:
            raise OpenAIError(
                "A choice needs the options it chooses from.",
                status_code=400,
                param="questions",
                code="invalid_request",
            )
        return list(criteria)
    if question.type == NOUL:
        return [FALSE, TRUE]
    if question.type == SCORE:
        criteria = question.criteria
        if not isinstance(criteria, list) or not criteria:
            raise OpenAIError(
                "A score needs the levels of its scale.",
                status_code=400,
                param="questions",
                code="invalid_request",
            )
        return list(criteria)
    raise OpenAIError(
        f"'{question.type}' is not a question type Ollama knows.",
        status_code=400,
        param="questions",
        code="invalid_request",
    )


def _state(state: str | dict[str, Any] | list[Any], instructions: str) -> str:
    """Return the text a question is about.

    The state is serialized as JSON text when it is not a string, and the
    instructions of the question come first.

    Args:
        state: The state of the request.
        instructions: The instructions of the question.

    Returns:
        The text of the question.
    """
    text = state if isinstance(state, str) else json.dumps(state)
    return f"{instructions}\n\n{text}"


def _answer(question: SystemOneQuestion, decision: Decision) -> dict[str, Any]:
    """Return the answer of a question.

    Args:
        question: The question of the client.
        decision: The probabilities the model answered with.

    Returns:
        The answer, in the shape Ollama reports it.
    """
    probabilities = _rounded(normalize(decision.probabilities))
    if question.type == NOUL:
        return {"type": NOUL, "noul": probabilities.get(TRUE, 0.0)}
    if question.type == SCORE:
        levels = list(probabilities)
        return {
            "type": SCORE,
            "score": score(probabilities, levels),
            "legend": {str(index): level for index, level in enumerate(levels)},
            "probabilities": {
                str(index): probability for index, probability in enumerate(probabilities.values())
            },
            "confidence": _rounded_value(confidence(probabilities)),
        }
    return {
        "type": CHOICE,
        "choice": best(probabilities),
        "probabilities": probabilities,
        "confidence": _rounded_value(confidence(probabilities)),
    }


def _rounded(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Round the probabilities, the way Ollama reports them."""
    return {name: round(value, 4) for name, value in probabilities.items()}


def _rounded_value(value: float) -> float:
    """Round a single number, the way Ollama reports it."""
    return round(value, 4)


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
            top_p=request.top_p,
            num_predict=_max_tokens(request),
            stop=stop,
        )
    )
    messages = [(message.role, message.content) for message in request.messages]
    engine = get_engine()
    keep_alive = parse_keep_alive(settings.keep_alive)
    if request.stream:
        # The model is loaded before the headers are sent: a broken checkpoint
        # becomes a 503 instead of a truncated stream.
        await run_in_threadpool(engine.prepare, request.model)
        return StreamingResponse(
            _chat_events(engine, messages, request.model, options, stop, keep_alive),
            media_type=SSE_MEDIA_TYPE,
        )
    result = await run_in_threadpool(engine.generate_chat, messages, model=request.model, **options)
    text = apply_stop_sequences(result.text, stop)
    if keep_alive is not None:
        engine.schedule_unload(keep_alive)
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
    keep_alive: float | None = None,
) -> Iterator[str]:
    """Yield the Server-Sent Events of a chat completion.

    Args:
        engine: The engine that runs the generation.
        messages: The chat history as ``(role, content)`` pairs.
        model: The model the client asked for.
        options: The translated generation options.
        stop: The stop sequences requested by the client.
        keep_alive: The lifetime of the loaded model, if one is configured.

    Yields:
        The events of the stream, the last one being ``[DONE]``.
    """
    try:
        yield from _completion_events(engine, messages, model, options, stop)
    finally:
        if keep_alive is not None:
            # The model stays loaded for the configured idle time.
            engine.schedule_unload(keep_alive)


def _completion_events(
    engine: AirLLMEngine,
    messages: list[tuple[str, str]],
    model: str,
    options: dict[str, object],
    stop: list[str],
) -> Iterator[str]:
    """Yield the Server-Sent Events of the answer of the model.

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
    # The stream ended without the statistics of the generation: the client
    # still expects the last chunk.
    yield _chunk_event(completion_id, created, model, finish_reason="stop")
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


@openai_router.post("/completions", response_model=TextCompletion)
async def completions(request: CompletionRequest) -> TextCompletion | StreamingResponse:
    """Answer a text completion, as ``POST /v1/completions`` does in OpenAI.

    The legacy endpoint takes a prompt instead of a chat history. Every prompt
    of the request is completed, and the answer has one choice per prompt.

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
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    stop = _stop_sequences(request.stop)
    options = map_options(
        GenerateOptions(
            temperature=request.temperature,
            top_p=request.top_p,
            num_predict=request.max_tokens,
            stop=stop,
        )
    )
    engine = get_engine()
    keep_alive = parse_keep_alive(settings.keep_alive)
    if request.stream:
        # The model is loaded before the headers are sent: a broken checkpoint
        # becomes a 503 instead of a truncated stream.
        await run_in_threadpool(engine.prepare, request.model)
        return StreamingResponse(
            _text_events(engine, prompts, request.model, options, stop, keep_alive),
            media_type=SSE_MEDIA_TYPE,
        )
    choices: list[CompletionChoice] = []
    prompt_tokens = 0
    completion_tokens = 0
    for index, prompt in enumerate(prompts):
        result = await run_in_threadpool(
            engine.generate_result, prompt, model=request.model, system=None, raw=False, **options
        )
        choices.append(
            CompletionChoice(
                text=apply_stop_sequences(result.text, stop),
                index=index,
                finish_reason=generation_done_reason(result, options),
            )
        )
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.generated_tokens
    if keep_alive is not None:
        engine.schedule_unload(keep_alive)
    return TextCompletion(
        id=_text_completion_id(),
        created=int(time.time()),
        model=settings.model_name,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@dataclass(frozen=True)
class _TextStream:
    """The state that every chunk of a text completion shares."""

    id: str
    created: int
    model: str


def _text_events(
    engine: AirLLMEngine,
    prompts: Sequence[str],
    model: str,
    options: dict[str, object],
    stop: list[str],
    keep_alive: float | None = None,
) -> Iterator[str]:
    """Yield the Server-Sent Events of the completions of the prompts.

    Args:
        engine: The engine that runs the generation.
        prompts: The prompts to complete.
        model: The model the client asked for.
        options: The translated generation options.
        stop: The stop sequences requested by the client.
        keep_alive: The lifetime of the loaded model, if one is configured.

    Yields:
        The events of the stream, the last one being ``[DONE]``.
    """
    stream = _TextStream(id=_text_completion_id(), created=int(time.time()), model=model)
    try:
        for index, prompt in enumerate(prompts):
            yield from _text_prompt_events(engine, prompt, index, stream, options, stop)
        yield DONE
    finally:
        if keep_alive is not None:
            # The model stays loaded for the configured idle time.
            engine.schedule_unload(keep_alive)


def _text_prompt_events(
    engine: AirLLMEngine,
    prompt: str,
    index: int,
    stream: _TextStream,
    options: dict[str, object],
    stop: list[str],
) -> Iterator[str]:
    """Yield the Server-Sent Events of the answer of one prompt.

    Args:
        engine: The engine that runs the generation.
        prompt: The prompt to complete.
        index: The number of the prompt, the index of the choice.
        stream: The state that every chunk shares.
        options: The translated generation options.
        stop: The stop sequences requested by the client.

    Yields:
        The events of the answer of the prompt.
    """
    stop_filter = StopSequenceFilter(stop)
    for chunk in engine.generate_stream(
        prompt, model=stream.model, system=None, raw=False, **options
    ):
        if chunk.result is not None:
            # The generation is over: the filter releases the tail it held
            # back, it cannot become a stop sequence any more.
            yield from _text_content_event(stream, index, stop_filter.flush())
            yield _text_chunk_event(
                stream,
                index,
                finish_reason=generation_done_reason(chunk.result, options),
            )
            return
        text = stop_filter.feed(chunk.text)
        if text:
            yield _text_chunk_event(stream, index, content=text)
        if stop_filter.stopped:
            # The stop sequence was found: the answer is complete, but the
            # client still expects the last chunk.
            yield _text_chunk_event(stream, index, finish_reason="stop")
            return
    yield from _text_content_event(stream, index, stop_filter.flush())
    # The stream ended without the statistics of the generation: the client
    # still expects the last chunk.
    yield _text_chunk_event(stream, index, finish_reason="stop")


def _text_content_event(stream: _TextStream, index: int, content: str) -> Iterator[str]:
    """Yield the event of a piece of text, if there is one.

    Args:
        stream: The state that every chunk shares.
        index: The number of the choice.
        content: The text of the chunk.

    Yields:
        The event of the text, nothing when the text is empty.
    """
    if not content:
        return
    yield _text_chunk_event(stream, index, content=content)


def _text_chunk_event(
    stream: _TextStream,
    index: int,
    content: str = "",
    finish_reason: str | None = None,
) -> str:
    """Render a chunk of a text completion as a Server-Sent Event.

    Args:
        stream: The state that every chunk shares.
        index: The number of the choice.
        content: The text of the chunk.
        finish_reason: The reason the answer ended, in the last chunk.

    Returns:
        The event, ready to be written to the stream.
    """
    chunk = TextCompletionChunk(
        id=stream.id,
        created=stream.created,
        model=stream.model,
        choices=[
            CompletionChunkChoice(
                text=content,
                index=index,
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def _text_completion_id() -> str:
    """Return the id of a text completion."""
    return f"cmpl-airllm-{uuid4().hex[:12]}"


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


def _max_tokens(request: ChatCompletionRequest) -> int | None:
    """Return the number of tokens the client asked for.

    The newer OpenAI clients send ``max_completion_tokens`` instead of
    ``max_tokens``; the newer name wins when both are sent.

    Args:
        request: The request of the client.

    Returns:
        The number of tokens, or ``None`` when the client did not ask for a
        limit.
    """
    if request.max_completion_tokens is not None:
        return request.max_completion_tokens
    return request.max_tokens
