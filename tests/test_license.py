"""Tests for the license of the package.

The build of the package succeeds even when the file with the text of the
license is missing — the wheel would just lose its license text, and the
metadata would still promise one. The test is the guard against that.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).parent.parent
LICENSE = ROOT / "LICENSE"
PYPROJECT = ROOT / "pyproject.toml"

# The phrases that every copy of the MIT license holds.
MIT = (
    "MIT License",
    "Permission is hereby granted, free of charge",
    "The above copyright notice and this permission notice shall be included",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
)


def _project() -> dict[str, Any]:
    """Return the metadata of the package.

    Returns:
        The ``[project]`` table of ``pyproject.toml``.
    """
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return cast("dict[str, Any]", document["project"])


def test_the_license_file_is_in_the_repository() -> None:
    """The text of the license is in the repository."""
    assert LICENSE.is_file()

    text = LICENSE.read_text(encoding="utf-8")
    for phrase in MIT:
        assert phrase in text


def test_the_metadata_declares_the_same_license() -> None:
    """The metadata of the package declares the license of the file."""
    project = _project()

    assert project["license"] == "MIT"
    # The file of the metadata has to be the file of the repository: a name
    # that points to nothing would leave the artifacts without a license.
    assert project["license-files"] == [LICENSE.name]


def test_the_copyright_names_the_author_of_the_package() -> None:
    """The copyright of the license names the author of the package."""
    project = _project()
    author = cast("list[dict[str, str]]", project["authors"])[0]["name"]

    text = LICENSE.read_text(encoding="utf-8")
    assert re.search(rf"Copyright \(c\) \d{{4}} {re.escape(author)}", text)
