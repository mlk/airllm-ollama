"""Tests for the workflows of the repository.

The workflows are only run when a tag is pushed or when a pull request is
opened — a broken one would be discovered at the worst moment. The tests here
pin the parts that matter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOWS = Path(__file__).parent.parent / ".github" / "workflows"
ROOT = WORKFLOWS.parent.parent


def _workflow_paths() -> list[Path]:
    """Return the workflow files of the repository.

    GitHub Actions loads the workflows of both extensions, so both are read.

    Returns:
        The workflow files.
    """
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _workflow(name: str) -> tuple[dict[str, Any], str]:
    """Return the parsed workflow of the given name and its text.

    Args:
        name: The file name of the workflow.

    Returns:
        The workflow and the text it was parsed from.
    """
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    # The base loader keeps every scalar a string: the key `on` would become
    # a boolean otherwise (the YAML 1.1 rule that GitHub does not follow).
    workflow = cast("dict[str, Any]", yaml.load(text, Loader=yaml.BaseLoader))
    return workflow, text


def _steps(workflow: dict[str, Any], job_name: str | None = None) -> list[dict[str, Any]]:
    """Return the steps of a job of a workflow.

    Args:
        workflow: The workflow.
        job_name: The name of the job; the first one when it is not given.

    Returns:
        The steps of the job.
    """
    jobs = cast("dict[str, Any]", workflow["jobs"])
    job = cast("dict[str, Any]", jobs[job_name] if job_name else next(iter(jobs.values())))
    return cast("list[dict[str, Any]]", job["steps"])


def test_the_publish_workflow_publishes_a_tag() -> None:
    """A tag builds the artifacts and publishes them to PyPI."""
    workflow, text = _workflow("publish.yml")

    assert workflow["on"]["push"]["tags"] == ["*"]
    # A publication in flight is never cancelled.
    assert str(workflow["concurrency"]["cancel-in-progress"]).lower() == "false"
    # The workflow reads the repository, it does not write to it.
    assert workflow["permissions"]["contents"] == "read"
    # The trusted publishing of PyPI needs a token of the workflow itself.
    assert workflow["permissions"]["id-token"] == "write"

    steps = _steps(workflow, "publish")
    runs = [str(step.get("run", "")) for step in steps]
    uses = [str(step.get("uses", "")) for step in steps]
    assert any("uv build" in run for run in runs)
    assert any("pypa/gh-action-pypi-publish" in use for use in uses)

    # The version of the package is derived from the git tags.
    assert "fetch-depth: 0" in text
    # A development version is not a release: the workflow refuses it.
    assert "is not a release version" in text


def test_the_quality_gates_run_on_every_pull_request() -> None:
    """The quality gates are the gate of every change."""
    workflow, _ = _workflow("ci.yml")

    assert "pull_request" in workflow["on"]
    assert workflow["on"]["push"]["branches"] == ["main"]


def test_the_review_is_run_by_the_ollama_action() -> None:
    """The pull requests are reviewed by an Ollama Cloud model."""
    workflow, text = _workflow("ollama-review.yml")

    assert "pull_request" in workflow["on"]
    # The workflow only reads the repository and writes to the pull request.
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["pull-requests"] == "write"
    # A push to a draft or a pull request from a fork does not spend a model call.
    guard = str(workflow["jobs"]["review"]["if"])
    assert "draft == false" in guard
    assert "head.repo.full_name == github.repository" in guard
    # Dependabot gets a read-only token and no secrets: its pull requests are
    # not reviewed at all.
    assert "dependabot[bot]" in guard
    # A stuck model call fails in minutes, not in the six hours of the runner.
    assert str(workflow["jobs"]["review"]["timeout-minutes"]) == "15"

    steps = _steps(workflow, "review")
    uses = [str(step.get("uses", "")) for step in steps]
    assert any(use.startswith("mlk/ollama-review-action@") for use in uses)

    # The action is pinned to a commit SHA: a tag can be moved.
    action = next(use for use in uses if use.startswith("mlk/ollama-review-action@"))
    sha = action.removeprefix("mlk/ollama-review-action@").split("#")[0].strip()
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"the action is not pinned to a commit SHA: {sha}"

    # The API key of Ollama Cloud is a secret of the repository.
    assert "OLLAMA_API_KEY" in text
    assert "secrets.OLLAMA_API_KEY" in text
    assert "JUNIE_API_KEY" not in text

    # A model is always named: the action ships without a default.
    with_ = [step.get("with", {}) for step in steps]
    assert any(str(inputs.get("model", "")) for inputs in with_)


def test_the_review_reads_the_conventions_of_the_repository() -> None:
    """The model is handed the conventions of the repository as context."""
    workflow, _ = _workflow("ollama-review.yml")

    steps = _steps(workflow, "review")
    inputs = next(step["with"] for step in steps if "with" in step)
    context_files = str(inputs["context_files"]).split(",")
    assert "AGENTS.md" in context_files
    assert "README.md" in context_files
    assert ".github/review-guidelines.md" in context_files
    # The guidelines of the review live in the repository, not in the action.
    assert (ROOT / ".github" / "review-guidelines.md").is_file()

    # The cap of the action is per file. It has to cover the longest of them,
    # or the model would review without the conventions of the repository.
    cap = int(str(inputs["context_max_chars"]))
    for name in context_files:
        length = len((ROOT / name).read_text(encoding="utf-8"))
        assert cap >= length, f"the context of {name} would be truncated ({length} characters)"

    # The cap of the action is per file. It has to cover the longest of them,
    # or the model would review without the conventions of the repository.
    cap = int(str(inputs["context_max_chars"]))
    for name in context_files:
        length = len((ROOT / name).read_text(encoding="utf-8"))
        assert cap >= length, f"the context of {name} would be truncated ({length} characters)"


def test_the_junie_review_is_gone() -> None:
    """The review does not run on JetBrains Junie any more."""
    # Every workflow is scanned: a Junie review under another name would keep
    # running on every pull request.
    for path in _workflow_paths():
        assert "junie" not in path.read_text(encoding="utf-8").lower(), path.name
    assert not (ROOT / ".junie").exists()
