"""Downloading the checkpoint of a model.

The service serves one model, and the checkpoint comes from the Hugging Face
Hub. The repositories of the models carry much more than the service needs:
the ONNX exports of `SmolLM2-1.7B-Instruct` alone are 13 GB, and AirLLM reads
none of them. The command downloads the files a checkpoint is made of — the
configuration, the tokenizer and the weights — and nothing else.

The Hub client is imported inside the function: the service itself never
downloads anything.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

# The files a checkpoint needs: the configuration, the tokenizer and the
# weights. Everything else in the repository (the ONNX exports, the training
# results) is of no use to AirLLM.
DEFAULT_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.safetensors",
    "*.model",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "*.tiktoken",
)


@dataclass(frozen=True)
class Download:
    """The result of a download."""

    model: str
    path: Path
    files: tuple[str, ...]
    size: int

    @property
    def megabytes(self) -> float:
        """Return the size of the download in megabytes."""
        return self.size / 1_000_000


def download_model(
    model: str,
    path: str | Path,
    *,
    token: str | None = None,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> Download:
    """Download the checkpoint of a model.

    Args:
        model: The repository of the model, e.g. ``Qwen/Qwen2.5-0.5B-Instruct``.
        path: The directory the checkpoint is written to.
        token: The Hugging Face token of an account with access to the model.
            Without it, the token of the environment (``HF_TOKEN``) or of the
            cached login is used.
        patterns: The patterns of the files to download.

    Returns:
        The download: where the checkpoint is and what it holds.
    """
    from huggingface_hub import snapshot_download

    directory = Path(
        snapshot_download(
            model,
            allow_patterns=list(patterns),
            local_dir=str(path),
            token=token,
        )
    )
    # Only the files of the checkpoint are reported: the directory can hold
    # more (the shards of AirLLM, for example).
    files = tuple(
        sorted(
            name
            for name in (
                item.relative_to(directory).as_posix()
                for item in directory.rglob("*")
                if item.is_file() and ".cache" not in item.parts
            )
            if _matches(name, patterns)
        )
    )
    size = sum((directory / name).stat().st_size for name in files)
    return Download(model=model, path=directory, files=files, size=size)


def _matches(name: str, patterns: Sequence[str]) -> bool:
    """Return whether a file of the checkpoint matches one of the patterns.

    Args:
        name: The name of the file, relative to the checkpoint.
        patterns: The patterns of the files to download.

    Returns:
        ``True`` when the file is one of the checkpoint's files.
    """
    return any(fnmatch(name, pattern) for pattern in patterns)


def main(argv: Sequence[str] | None = None) -> int:
    """Download the checkpoint of a model.

    Args:
        argv: The arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when the checkpoint was downloaded, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="download-model",
        description="Download the checkpoint of a model from the Hugging Face Hub.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="repository of the model, e.g. Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="directory the checkpoint is written to",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="an extra pattern of files to download (can be given more than once)",
    )
    args = parser.parse_args(argv)

    try:
        download = download_model(
            args.model,
            args.path,
            patterns=(*DEFAULT_PATTERNS, *args.include),
        )
    except Exception as exc:
        print(_message(args.model, exc), file=sys.stderr)
        return 1
    _report(download)
    return 0


def _report(download: Download) -> None:
    """Print what was downloaded and how to serve it.

    Args:
        download: The result of the download.
    """
    print(f"Downloaded {download.model} to {download.path} ({download.megabytes:.1f} MB)")
    for name in download.files:
        print(f"  {name}")
    print("Serve it with:")
    print(
        f"  uv run --extra airllm python -m airllm_ollama "
        f"--model-path {download.path} --model {download.model}"
    )


def _message(model: str, exc: Exception) -> str:
    """Return the message of a failed download.

    Args:
        model: The repository of the model.
        exc: The error the Hub raised.

    Returns:
        The message for the operator.
    """
    from huggingface_hub import errors

    if isinstance(exc, errors.GatedRepoError):
        return (
            f"The model '{model}' needs a token: set HF_TOKEN to the token of an account with "
            "access to it."
        )
    if isinstance(exc, errors.RepositoryNotFoundError):
        # The Hub answers a private model like a missing one, so both are
        # possible.
        return (
            f"The model '{model}' was not found on the Hugging Face Hub. If it is private or "
            "gated, set HF_TOKEN to the token of an account with access to it."
        )
    return f"Could not download the model '{model}': {exc}"
