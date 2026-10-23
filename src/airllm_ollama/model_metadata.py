"""Metadata about the served model, derived from the local checkpoint.

Everything in this module works on files that are already on disk: nothing is
downloaded, and a model that has not been fetched yet simply reports empty
metadata instead of triggering a download.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.bin")
CHUNK_SIZE = 1024 * 1024
_STARTED_AT = datetime.now(UTC)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata of a model checkpoint, mirroring Ollama's ``/api/tags`` fields."""

    size: int
    digest: str
    modified_at: str
    family: str
    families: tuple[str, ...]
    parameter_size: str
    format: str


def collect_metadata(model_path: str) -> CheckpointMetadata:
    """Return the metadata of the checkpoint at ``model_path``.

    The result is cached: hashing a multi-hundred-megabyte checkpoint for every
    request would be wasteful.

    Args:
        model_path: Local checkpoint directory. Repository ids simply yield
            empty metadata.

    Returns:
        The metadata of the checkpoint.
    """
    return _collect_metadata(model_path)


@lru_cache(maxsize=4)
def _collect_metadata(model_path: str) -> CheckpointMetadata:
    """Collect the metadata of a checkpoint, caching the result."""
    directory = Path(model_path)
    if not directory.is_dir():
        return CheckpointMetadata(
            size=0,
            digest="",
            modified_at=_timestamp(_STARTED_AT),
            family="",
            families=(),
            parameter_size="",
            format="",
        )

    weights = _weights_file(directory)
    return CheckpointMetadata(
        size=_total_size(directory),
        digest=_digest(weights) if weights is not None else "",
        modified_at=_timestamp(_modified_at(directory)),
        family=_family(directory),
        families=_families(directory),
        parameter_size=_parameter_size(directory),
        format=_format(directory),
    )


def _weights_file(directory: Path) -> Path | None:
    """Return the file holding the model weights, if the checkpoint has one."""
    for name in WEIGHT_FILES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _total_size(directory: Path) -> int:
    """Return the total size of all files in ``directory``."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _digest(weights: Path) -> str:
    """Return the ``sha256:`` digest of the weights file."""
    hasher = hashlib.sha256()
    with weights.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _modified_at(directory: Path) -> datetime:
    """Return the modification time of the newest file in ``directory``."""
    timestamps = [path.stat().st_mtime for path in directory.rglob("*") if path.is_file()]
    if not timestamps:
        return _STARTED_AT
    return datetime.fromtimestamp(max(timestamps), tz=UTC)


def _timestamp(moment: datetime) -> str:
    """Format ``moment`` the way Ollama formats its timestamps."""
    return moment.isoformat().replace("+00:00", "Z")


def _config(directory: Path) -> dict[str, object]:
    """Return the parsed ``config.json`` of the checkpoint, if present."""
    config_path = directory / "config.json"
    if not config_path.is_file():
        return {}
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _family(directory: Path) -> str:
    """Return the model family, e.g. ``llama``."""
    config = _config(directory)
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type.lower()
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        first = architectures[0]
        if isinstance(first, str):
            return _family_from_architecture(first)
    return ""


def _families(directory: Path) -> tuple[str, ...]:
    """Return the model families, e.g. ``("llama",)``."""
    family = _family(directory)
    return (family,) if family else ()


def _family_from_architecture(architecture: str) -> str:
    """Derive the family name from a Hugging Face architecture name."""
    for suffix in (
        "ForConditionalGeneration",
        "ForSequenceClassification",
        "ForTokenClassification",
        "ForCausalLM",
        "ForMaskedLM",
        "Model",
        "LM",
    ):
        if architecture.endswith(suffix):
            architecture = architecture[: -len(suffix)]
            break
    return architecture.split("For")[0].lower()


def _parameter_size(directory: Path) -> str:
    """Return the parameter count of the checkpoint, e.g. ``135M``."""
    weights = _weights_file(directory)
    if weights is None or weights.suffix != ".safetensors":
        return ""
    count = _safetensors_parameter_count(weights)
    return _format_parameter_size(count) if count else ""


def _safetensors_parameter_count(weights: Path) -> int | None:
    """Read ``total_parameters`` from the header of a safetensors file."""
    try:
        with weights.open("rb") as handle:
            (header_size,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_size))
    except (OSError, ValueError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        return None
    total = metadata.get("total_parameters")
    return total if isinstance(total, int) else None


def _format_parameter_size(count: int) -> str:
    """Format a parameter count the way Ollama does, e.g. ``135M`` or ``1.3B``."""
    if count >= 1_000_000_000:
        value = f"{count / 1_000_000_000:.1f}"
        if value.endswith(".0"):
            value = value[:-2]
        return f"{value}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


def _format(directory: Path) -> str:
    """Return the checkpoint format, e.g. ``safetensors``."""
    for name, format_name in (
        ("model.safetensors", "safetensors"),
        ("pytorch_model.bin", "pytorch"),
        ("model.bin", "pytorch"),
    ):
        if (directory / name).is_file():
            return format_name
    return ""
