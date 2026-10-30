"""API route definitions for the AirLLM Ollama wrapper.

The ``/api/*`` routes mirror Ollama's REST interface, so standard Ollama
clients can talk to this service unchanged.
"""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from airllm_ollama.api.disconnect import stream_events
from airllm_ollama.api.stream import chat_events, generate_events, media_type
from airllm_ollama.config import Settings, get_settings
from airllm_ollama.engine import AirLLMEngine, get_engine, model_names_match
from airllm_ollama.exceptions import ModelMismatchError
from airllm_ollama.model_metadata import CheckpointMetadata, collect_metadata
from airllm_ollama.options import (
    apply_stop_sequences,
    map_options,
    parse_keep_alive,
)
from airllm_ollama.options import done_reason as generation_done_reason
from airllm_ollama.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from airllm_ollama.schemas.generate import GenerateRequest, GenerateResponse
from airllm_ollama.schemas.health import HealthResponse
from airllm_ollama.schemas.ollama import ErrorBody
from airllm_ollama.schemas.ps import PsResponse, RunningModel
from airllm_ollama.schemas.show import ShowRequest, ShowResponse
from airllm_ollama.schemas.tags import ModelDetails, ModelInfo, TagsResponse
from airllm_ollama.schemas.version import VersionResponse

api_router = APIRouter()

# The reason why the embedding endpoints are not implemented.
EMBEDDINGS_ERROR = "This service generates text; it does not produce embeddings."

# The model management endpoints of Ollama are not implemented: this service
# serves a single, pre-configured model.
NOT_IMPLEMENTED = 501

# The answer of those endpoints, documented in the OpenAPI schema.
NOT_IMPLEMENTED_RESPONSE: dict[int | str, dict[str, Any]] = {
    NOT_IMPLEMENTED: {
        "model": ErrorBody,
        "description": "This service does not implement the endpoint.",
    }
}

# The errors the generation endpoints can answer with, documented in the
# OpenAPI schema.
GENERATION_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorBody, "description": "The request body is invalid."},
    404: {"model": ErrorBody, "description": "The model is not the one this service serves."},
    503: {"model": ErrorBody, "description": "The model could not be loaded."},
}


@api_router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Report service liveness and the configured model."""
    settings = get_settings()
    return HealthResponse(
        version=settings.version,
        model=settings.model_name,
        model_loaded=get_engine().is_loaded,
    )


@api_router.get("/", response_class=PlainTextResponse, tags=["meta"])
async def root() -> PlainTextResponse:
    """Announce the service, in the spirit of Ollama's own root response.

    The service reports that it is only *kinda* running when the AirLLM
    runtime is missing: the API answers, but inference is not possible.
    """
    status = "running" if get_engine().is_available else "kinda running"
    return PlainTextResponse(f"airllm-ollama is {status}")


@api_router.get("/api/version", response_model=VersionResponse, tags=["ollama"])
async def version() -> VersionResponse:
    """Return the server version, as ``GET /api/version`` does in Ollama."""
    return VersionResponse(version=get_settings().version)


@api_router.post(
    "/api/generate",
    response_model=GenerateResponse,
    responses=GENERATION_ERRORS,
    tags=["ollama"],
)
async def generate(request: GenerateRequest) -> GenerateResponse | StreamingResponse:
    """Generate a completion, as ``POST /api/generate`` does in Ollama.

    With ``stream: true`` the answer is streamed as newline delimited JSON (or
    Server-Sent Events), otherwise it is returned as a single JSON object with
    ``done`` set to ``true``.
    """
    engine = get_engine()
    options = map_options(request.options)
    stop = request.options.stop if request.options else None
    if request.stream:
        engine.prepare(request.model)
        # The event stops the generation when the client disconnects.
        abort = threading.Event()
        return _stream_response(
            generate_events(
                engine,
                request.prompt,
                system=request.system,
                raw=request.raw,
                model=request.model,
                options=options,
                stop=stop,
                abort=abort,
            ),
            engine,
            request.keep_alive,
            abort,
        )
    result = engine.generate_result(
        request.prompt,
        system=request.system,
        raw=request.raw,
        model=request.model,
        stop=stop or (),
        **options,
    )
    text = apply_stop_sequences(result.text, stop)

    settings = get_settings()
    keep_alive = parse_keep_alive(
        request.keep_alive if request.keep_alive is not None else settings.keep_alive
    )
    if keep_alive is not None:
        engine.schedule_unload(keep_alive)

    return GenerateResponse(
        model=request.model,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        response=text,
        done=True,
        done_reason=generation_done_reason(result, options),
        total_duration=result.total_duration_ns,
        load_duration=result.load_duration_ns,
        prompt_eval_count=result.prompt_tokens,
        prompt_eval_duration=result.prompt_duration_ns,
        eval_count=result.generated_tokens,
        eval_duration=result.eval_duration_ns,
    )


def _stream_response(
    events: Iterator[str], engine: AirLLMEngine, keep_alive: str | None, abort: threading.Event
) -> StreamingResponse:
    """Return a streaming response and schedule the unload of the model.

    Args:
        events: The events of the answer.
        engine: The engine that runs the generation.
        keep_alive: The idle time after which the model is unloaded.
        abort: The event that stops the generation when the client is gone.

    Returns:
        The streaming response.
    """
    settings = get_settings()
    lifetime = parse_keep_alive(keep_alive if keep_alive is not None else settings.keep_alive)
    if lifetime is not None:
        engine.schedule_unload(lifetime)
    return StreamingResponse(stream_events(events, abort), media_type=media_type())


@api_router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses=GENERATION_ERRORS,
    tags=["ollama"],
)
async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
    """Answer a chat, as ``POST /api/chat`` does in Ollama.

    The messages are rendered with the chat template of the model; models
    without a template get a plain transcript. With ``stream: true`` the
    answer is streamed as newline delimited JSON (or Server-Sent Events),
    otherwise it is returned as a single JSON object with ``done`` set to
    ``true``.
    """
    engine = get_engine()
    messages = [(message.role, message.content) for message in request.messages]
    options = map_options(request.options)
    stop = request.options.stop if request.options else None
    if request.stream:
        engine.prepare(request.model)
        # The event stops the generation when the client disconnects.
        abort = threading.Event()
        return _stream_response(
            chat_events(
                engine,
                messages,
                model=request.model,
                options=options,
                stop=stop,
                abort=abort,
            ),
            engine,
            request.keep_alive,
            abort,
        )
    result = engine.generate_chat(messages, model=request.model, stop=stop or (), **options)
    text = apply_stop_sequences(result.text, stop)

    settings = get_settings()
    keep_alive = parse_keep_alive(
        request.keep_alive if request.keep_alive is not None else settings.keep_alive
    )
    if keep_alive is not None:
        engine.schedule_unload(keep_alive)

    return ChatResponse(
        model=request.model,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        message=ChatMessage(role="assistant", content=text),
        done=True,
        done_reason=generation_done_reason(result, options),
        total_duration=result.total_duration_ns,
        load_duration=result.load_duration_ns,
        prompt_eval_count=result.prompt_tokens,
        prompt_eval_duration=result.prompt_duration_ns,
        eval_count=result.generated_tokens,
        eval_duration=result.eval_duration_ns,
    )


@api_router.get("/api/tags", response_model=TagsResponse, tags=["ollama"])
async def tags() -> TagsResponse:
    """Return the single served model, as ``GET /api/tags`` does in Ollama."""
    settings = get_settings()
    metadata = collect_metadata(settings.resolved_model_path)
    return TagsResponse(
        models=[
            ModelInfo(
                name=settings.model_name,
                model=settings.model_name,
                modified_at=metadata.modified_at,
                size=metadata.size,
                digest=metadata.digest,
                details=ModelDetails(
                    format=metadata.format,
                    family=metadata.family,
                    families=list(metadata.families),
                    parameter_size=metadata.parameter_size,
                    context_length=settings.max_seq_len,
                ),
                capabilities=list(settings.capabilities),
            )
        ]
    )


@api_router.post(
    "/api/show", response_model=ShowResponse, responses=GENERATION_ERRORS, tags=["ollama"]
)
async def show(request: ShowRequest) -> ShowResponse:
    """Return the information about the model, as ``POST /api/show`` does in Ollama.

    The information is derived from the local checkpoint — the endpoint never
    loads the model and never downloads anything.

    Args:
        request: The request of the client.

    Returns:
        The information about the served model.

    Raises:
        ModelMismatchError: If the client asked for another model.
    """
    settings = get_settings()
    if not model_names_match(request.model, settings.model_name):
        raise ModelMismatchError(
            f"Model '{request.model}' is not available. This server serves a single model: "
            f"'{settings.model_name}'."
        )
    metadata = collect_metadata(settings.resolved_model_path)
    return ShowResponse(
        modelfile=_modelfile(settings, metadata),
        parameters=_parameters(settings),
        template=metadata.template,
        details=_details(metadata, settings),
        model_info=metadata.model_info,
        capabilities=list(settings.capabilities),
    )


@api_router.get("/api/ps", response_model=PsResponse, tags=["ollama"])
async def ps() -> PsResponse:
    """Return the models that are loaded, as ``GET /api/ps`` does in Ollama.

    The service loads its model lazily, so the list is empty until the first
    request.
    """
    settings = get_settings()
    engine = get_engine()
    if not engine.is_loaded:
        return PsResponse(models=[])
    metadata = collect_metadata(settings.resolved_model_path)
    expires_at = engine.unload_at
    return PsResponse(
        models=[
            RunningModel(
                name=settings.model_name,
                model=settings.model_name,
                size=metadata.size,
                digest=metadata.digest,
                details=_details(metadata, settings),
                expires_at=_timestamp(expires_at) if expires_at else "",
                # AirLLM streams the model layer by layer, the weights are
                # never in the memory of the device at once.
                size_vram=0,
            )
        ]
    )


def _details(metadata: CheckpointMetadata, settings: Settings) -> ModelDetails:
    """Return the details of the served model."""
    return ModelDetails(
        format=metadata.format,
        family=metadata.family,
        families=list(metadata.families),
        parameter_size=metadata.parameter_size,
        context_length=settings.max_seq_len,
    )


def _modelfile(settings: Settings, metadata: CheckpointMetadata) -> str:
    """Return the Modelfile of the served model.

    A checkpoint does not hold a Modelfile, so the service describes the model
    it serves — the way ``ollama show`` does it.
    """
    lines = [
        '# Modelfile generated by "airllm-ollama show"',
        "# To build a new Modelfile based on this one, replace the FROM line with:",
        f"# FROM {settings.model_name}",
        "",
        f"FROM {settings.resolved_model_path}",
    ]
    if metadata.template:
        lines.append(f'TEMPLATE """{metadata.template}"""')
    lines.append(f"PARAMETER num_ctx {settings.max_seq_len}")
    return "\n".join(lines) + "\n"


def _parameters(settings: Settings) -> str:
    """Return the parameters of the served model, the way Ollama shows them."""
    return f"{'num_ctx':<32}{settings.max_seq_len}\n"


@api_router.post(
    "/api/embed",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def embed() -> JSONResponse:
    """Answer the embeddings of the inputs, as ``POST /api/embed`` does in Ollama.

    This service generates text — it does not produce embeddings. The endpoint
    exists so that a client gets a clear answer instead of a "not found".
    """
    return _not_implemented(EMBEDDINGS_ERROR)


@api_router.post(
    "/api/embeddings",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def embeddings() -> JSONResponse:
    """Answer the embedding of a prompt, as ``POST /api/embeddings`` does in Ollama.

    The legacy endpoint, superseded by ``/api/embed``. This service generates
    text — it does not produce embeddings.
    """
    return _not_implemented(EMBEDDINGS_ERROR)


@api_router.post(
    "/api/pull",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def pull() -> JSONResponse:
    """Pull a model, as ``POST /api/pull`` does in Ollama.

    This service serves a single, pre-configured model — it cannot pull
    models. The endpoint exists so that a client gets a clear answer instead
    of a "not found".
    """
    return _not_implemented(_model_management_error("pull"))


@api_router.post(
    "/api/push",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def push() -> JSONResponse:
    """Push a model, as ``POST /api/push`` does in Ollama.

    This service serves a single, pre-configured model — it cannot push
    models.
    """
    return _not_implemented(_model_management_error("push"))


@api_router.post(
    "/api/create",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def create() -> JSONResponse:
    """Create a model, as ``POST /api/create`` does in Ollama.

    This service serves a single, pre-configured model — it cannot create
    models.
    """
    return _not_implemented(_model_management_error("create"))


@api_router.post(
    "/api/blobs/{digest}",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def push_blob(digest: str) -> JSONResponse:
    """Push a blob, as ``POST /api/blobs/:digest`` does in Ollama.

    The blobs are the files a model is created from. This service serves a
    single, pre-configured model — it stores no blobs. The body of the
    request is not read: the answer comes before the file is transferred.

    Args:
        digest: The SHA256 digest of the file the client sends.

    Returns:
        The answer with the status code ``501``.
    """
    return _not_implemented(_blob_error("push"))


@api_router.head(
    "/api/blobs/{digest}",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def check_blob(digest: str) -> Response:
    """Check a blob, as ``HEAD /api/blobs/:digest`` does in Ollama.

    A ``HEAD`` answer has no body, so the status code is the whole answer —
    the body of the equivalent ``GET`` would carry the message.

    Args:
        digest: The SHA256 digest of the file the client asks about.

    Returns:
        The answer with the status code ``501``.
    """
    return Response(status_code=NOT_IMPLEMENTED)


@api_router.post(
    "/api/copy",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def copy() -> JSONResponse:
    """Copy a model, as ``POST /api/copy`` does in Ollama.

    This service serves a single, pre-configured model — it cannot copy
    models.
    """
    return _not_implemented(_model_management_error("copy"))


@api_router.delete(
    "/api/delete",
    status_code=NOT_IMPLEMENTED,
    responses=NOT_IMPLEMENTED_RESPONSE,
    tags=["ollama"],
)
async def delete() -> JSONResponse:
    """Delete a model, as ``DELETE /api/delete`` does in Ollama.

    This service serves a single, pre-configured model — it cannot delete
    models.
    """
    return _not_implemented(_model_management_error("delete"))


def _timestamp(moment: datetime) -> str:
    """Format a moment the way Ollama formats its timestamps."""
    return moment.isoformat().replace("+00:00", "Z")


def _blob_error(action: str) -> str:
    """Return the reason why a blob endpoint is not implemented.

    Args:
        action: The action of the endpoint, e.g. ``push``.

    Returns:
        The message of the error.
    """
    return (
        "This service serves a single, pre-configured model; "
        f"it cannot {action} the files of a model."
    )


def _not_implemented(reason: str) -> JSONResponse:
    """Return the answer of an endpoint this service does not implement.

    Args:
        reason: The reason of the answer, in a full sentence.

    Returns:
        The answer with the status code ``501``.
    """
    return JSONResponse(status_code=NOT_IMPLEMENTED, content={"error": reason})


def _model_management_error(action: str) -> str:
    """Return the reason why a model management endpoint is not implemented.

    Args:
        action: The action of the endpoint, e.g. ``pull``.

    Returns:
        The message of the error.
    """
    return (
        "This service serves a single, pre-configured model "
        f"(AIRLLM_OLLAMA_MODEL_NAME); it cannot {action} models."
    )
