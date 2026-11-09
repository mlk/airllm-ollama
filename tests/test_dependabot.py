"""Tests for the Dependabot configuration of the repository.

Dependabot opens the pull requests that keep the dependencies up to date. A
broken configuration would either stop the updates or raise them for the wrong
packages, so the parts that matter are pinned here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

CONFIG = Path(__file__).parent.parent / ".github" / "dependabot.yml"
ROOT = CONFIG.parent.parent


def _config() -> dict[str, Any]:
    """Return the parsed configuration of Dependabot.

    Returns:
        The configuration of Dependabot.
    """
    text = CONFIG.read_text(encoding="utf-8")
    # The base loader keeps every scalar a string, the way GitHub reads it.
    return cast("dict[str, Any]", yaml.load(text, Loader=yaml.BaseLoader))


def _ecosystems(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the configuration of every package ecosystem.

    Args:
        config: The configuration of Dependabot.

    Returns:
        The configuration of every package ecosystem, by its name.
    """
    updates = cast("list[dict[str, Any]]", config["updates"])
    return {update["package-ecosystem"]: update for update in updates}


def test_the_configuration_uses_the_current_syntax() -> None:
    """The configuration uses the current syntax of Dependabot."""
    config = _config()

    assert config["version"] == "2"
    assert _ecosystems(config)


def test_the_python_packages_are_updated_weekly() -> None:
    """The Python packages are kept up to date by the `uv` ecosystem."""
    update = _ecosystems(_config())["uv"]

    assert update["directory"] == "/"
    assert update["schedule"]["interval"] == "weekly"
    # The manifest and the lock file are in the root of the repository.
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "uv.lock").is_file()


def test_the_actions_are_updated_weekly_and_grouped() -> None:
    """The actions of the workflows are kept up to date, in one pull request."""
    update = _ecosystems(_config())["github-actions"]

    assert update["directory"] == "/"
    assert update["schedule"]["interval"] == "weekly"
    # The actions of the workflows are updated together.
    assert update["groups"]["actions"]["patterns"] == ["*"]


def test_the_commit_messages_follow_the_conventions_of_the_repository() -> None:
    """The pull requests of Dependabot use the commit style of the repository."""
    ecosystems = _ecosystems(_config())

    assert ecosystems["uv"]["commit-message"]["prefix"] == "chore(deps)"
    assert ecosystems["uv"]["commit-message"]["prefix-development"] == "chore(deps-dev)"
    assert ecosystems["github-actions"]["commit-message"]["prefix"] == "chore(ci)"
