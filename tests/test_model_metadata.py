"""Unit tests for the checkpoint metadata helpers."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path

from airllm_ollama.model_metadata import (
    _family_from_architecture,
    _format_parameter_size,
    clear_metadata_cache,
    collect_metadata,
)
from conftest import DEFAULT_PARAMETERS


def test_metadata_of_a_missing_checkpoint_is_empty(tmp_path: Path) -> None:
    metadata = collect_metadata(str(tmp_path / "does-not-exist"))

    assert metadata.size == 0
    assert metadata.digest == ""
    assert metadata.family == ""
    assert metadata.families == ()
    assert metadata.parameter_size == ""
    assert metadata.format == ""


def test_metadata_of_a_local_checkpoint(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.size == sum(path.stat().st_size for path in checkpoint.rglob("*"))
    assert metadata.digest.startswith("sha256:")
    assert metadata.family == "llama"
    assert metadata.families == ("llama",)
    assert metadata.parameter_size == "135M"
    assert metadata.format == "safetensors"
    assert metadata.modified_at.endswith("Z")


def test_metadata_of_an_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    metadata = collect_metadata(str(empty))

    assert metadata.size == 0
    assert metadata.digest == ""
    assert metadata.modified_at.endswith("Z")
    assert metadata.family == ""
    assert metadata.format == ""


def test_metadata_without_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.size > 0
    assert metadata.digest == ""
    assert metadata.format == ""
    assert metadata.parameter_size == ""
    assert metadata.family == "llama"


def test_metadata_is_cached(tmp_path: Path, write_checkpoint: Callable[..., Path]) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")

    first = collect_metadata(str(checkpoint))
    second = collect_metadata(str(checkpoint))

    assert first is second


def test_metadata_cache_can_be_cleared(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")

    first = collect_metadata(str(checkpoint))
    clear_metadata_cache()
    second = collect_metadata(str(checkpoint))

    assert first is not second
    assert second == first


def test_family_is_derived_from_the_architecture(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint", model_type="")
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"]}), encoding="utf-8"
    )

    metadata = collect_metadata(str(checkpoint))

    assert metadata.family == "qwen2"
    assert metadata.families == ("qwen2",)


def test_broken_config_is_ignored(tmp_path: Path, write_checkpoint: Callable[..., Path]) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "config.json").write_text("{not json", encoding="utf-8")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.family == ""
    assert metadata.parameter_size == "135M"


def test_broken_safetensors_header_is_ignored(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").write_bytes(b"broken")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.parameter_size == ""
    assert metadata.format == "safetensors"


def test_pytorch_checkpoint_is_detected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.format == "pytorch"
    assert metadata.parameter_size == ""


def test_parameter_size_formatting() -> None:
    assert _format_parameter_size(135_000_000) == "135M"
    assert _format_parameter_size(1_300_000_000) == "1.3B"
    assert _format_parameter_size(7_000_000_000) == "7B"
    assert _format_parameter_size(10_000_000_000) == "10B"
    assert _format_parameter_size(2_500) == "2K"
    assert _format_parameter_size(512) == "512"


def test_family_from_architecture_names() -> None:
    assert _family_from_architecture("LlamaForCausalLM") == "llama"
    assert _family_from_architecture("Qwen2ForCausalLM") == "qwen2"
    assert _family_from_architecture("MistralForCausalLM") == "mistral"
    assert _family_from_architecture("LlamaModel") == "llama"
    assert _family_from_architecture("Custom") == "custom"
    assert _family_from_architecture("CustomForCausalLM") == "custom"


def test_unexpected_architecture_entries_are_ignored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": [42]}), encoding="utf-8")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.family == ""


def test_safetensors_with_a_malformed_header_is_ignored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    blob = b"[1, 2, 3]"
    (checkpoint / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * 8
    )

    metadata = collect_metadata(str(checkpoint))

    assert metadata.parameter_size == ""


def test_safetensors_without_metadata_is_ignored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}})
    blob = header.encode("utf-8")
    (checkpoint / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * 8
    )

    metadata = collect_metadata(str(checkpoint))

    assert metadata.parameter_size == ""
    assert metadata.format == "safetensors"


def test_safetensors_header_is_read_without_loading_the_weights(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint", parameters=1_300_000_000)
    header_size = struct.unpack("<Q", (checkpoint / "model.safetensors").read_bytes()[:8])[0]

    metadata = collect_metadata(str(checkpoint))

    assert header_size > 0
    assert metadata.parameter_size == "1.3B"


def test_model_info_holds_the_configuration(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    """The information about the model is the configuration of the checkpoint."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.model_info["model_type"] == "llama"
    assert metadata.model_info["general.architecture"] == "llama"
    assert metadata.model_info["general.parameter_count"] == DEFAULT_PARAMETERS


def test_model_info_of_a_missing_checkpoint_is_empty(tmp_path: Path) -> None:
    metadata = collect_metadata(str(tmp_path / "does-not-exist"))

    assert metadata.model_info == {}
    assert metadata.template == ""


def test_the_template_is_read_from_the_tokenizer_config(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    """The chat template of the checkpoint is part of the metadata."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages[0].content }}"}), encoding="utf-8"
    )

    metadata = collect_metadata(str(checkpoint))

    assert metadata.template == "{{ messages[0].content }}"


def test_the_template_is_read_from_the_jinja_file(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    """The current place of the chat template is a file of its own."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.template == "{{ messages }}"


def test_a_template_that_is_a_mapping_uses_the_default(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    """Some checkpoints hold a mapping of named templates."""
    checkpoint = write_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": {"default": "{{ default }}", "tool": "{{ tool }}"}}),
        encoding="utf-8",
    )

    metadata = collect_metadata(str(checkpoint))

    assert metadata.template == "{{ default }}"


def test_a_checkpoint_without_a_template_has_none(
    tmp_path: Path, write_checkpoint: Callable[..., Path]
) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint")

    metadata = collect_metadata(str(checkpoint))

    assert metadata.template == ""
