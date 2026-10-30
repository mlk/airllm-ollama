"""Tests for the workflows of the repository.

The workflows are only run when a tag is pushed or when a pull request is
opened — a broken one would be discovered at the worst moment. The tests here
pin the parts that matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOWS = Path(__file__).parent.parent / ".github" / "workflows"


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
