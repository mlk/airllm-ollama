"""Tests for the download command.

The Hub is never contacted: the download is mocked, so the suite stays offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from huggingface_hub import errors
from pytest_mock import MockerFixture

from airllm_ollama.download import DEFAULT_PATTERNS, download_model, main


def _response() -> httpx2.Response:
    """Return the response of a request the Hub refused."""
    return httpx2.Response(404, request=httpx2.Request("GET", "https://huggingface.co"))


def _write_checkpoint(directory: Path) -> Path:
    """Write a checkpoint with a config and a weight file.

    Args:
        directory: The directory of the checkpoint.

    Returns:
        The directory of the checkpoint.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (directory / "model.safetensors").write_bytes(b"\x00" * 8)
    return directory


def test_the_download_holds_the_files_the_model_needs(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Only the configuration, the tokenizer and the weights are downloaded."""
    calls: list[dict[str, Any]] = []

    def snapshot_download(repo: str, **kwargs: Any) -> str:
        """Record the call and write a checkpoint."""
        calls.append({"repo": repo, **kwargs})
        return str(_write_checkpoint(tmp_path / "checkpoint"))

    mocker.patch("huggingface_hub.snapshot_download", side_effect=snapshot_download)

    download = download_model("some/model", tmp_path / "checkpoint")

    assert calls[0]["repo"] == "some/model"
    assert calls[0]["allow_patterns"] == list(DEFAULT_PATTERNS)
    assert calls[0]["local_dir"] == str(tmp_path / "checkpoint")
    assert download.files == ("config.json", "model.safetensors")
    assert download.size == 8 + len("""{"architectures": ["LlamaForCausalLM"]}""")


def test_the_download_reports_where_the_checkpoint_is(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints the files and how to serve the model."""
    mocker.patch(
        "huggingface_hub.snapshot_download",
        side_effect=lambda repo, **kwargs: str(_write_checkpoint(tmp_path / "checkpoint")),
    )

    assert main(["--model", "some/model", "--path", str(tmp_path / "checkpoint")]) == 0

    output = capsys.readouterr().out
    assert "Downloaded some/model" in output
    assert "model.safetensors" in output
    assert "--model-path" in output
    assert "--model some/model" in output


def test_the_short_arguments_are_accepted(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """The model and the path have a short form."""
    mocker.patch(
        "huggingface_hub.snapshot_download",
        side_effect=lambda repo, **kwargs: str(_write_checkpoint(tmp_path / "checkpoint")),
    )

    assert main(["-m", "some/model", "-p", str(tmp_path / "checkpoint")]) == 0

    assert "Downloaded some/model" in capsys.readouterr().out


def test_extra_patterns_are_downloaded(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pattern of the command line is added to the default ones."""
    calls: list[dict[str, Any]] = []

    def snapshot_download(repo: str, **kwargs: Any) -> str:
        """Record the call and write a checkpoint."""
        calls.append(kwargs)
        return str(_write_checkpoint(tmp_path / "checkpoint"))

    mocker.patch("huggingface_hub.snapshot_download", side_effect=snapshot_download)

    main(
        [
            "-m",
            "some/model",
            "-p",
            str(tmp_path / "checkpoint"),
            "--include",
            "*.bin",
        ]
    )

    assert calls[0]["allow_patterns"] == [*DEFAULT_PATTERNS, "*.bin"]
    assert capsys.readouterr()


def test_a_missing_model_is_reported(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model that does not exist is reported, with the hint about the token."""
    mocker.patch(
        "huggingface_hub.snapshot_download",
        side_effect=errors.RepositoryNotFoundError("not found", response=_response()),
    )

    assert main(["-m", "some/model", "-p", str(tmp_path / "checkpoint")]) == 1

    error = capsys.readouterr().err
    assert "some/model" in error
    assert "HF_TOKEN" in error


def test_a_gated_model_is_reported(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gated model needs a token."""
    mocker.patch(
        "huggingface_hub.snapshot_download",
        side_effect=errors.GatedRepoError("gated", response=_response()),
    )

    assert main(["-m", "some/model", "-p", str(tmp_path / "checkpoint")]) == 1

    assert "HF_TOKEN" in capsys.readouterr().err
