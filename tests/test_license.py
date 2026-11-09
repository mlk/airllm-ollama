"""Tests for the license of the package.

The metadata of the package promises the MIT license: the file with its text
has to be in the repository, it has to be the license of the metadata, and the
README has to document it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).parent.parent
LICENSE = ROOT / "LICENSE"
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

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
    assert "LICENSE" in project["license-files"]


def test_the_copyright_names_the_author_of_the_package() -> None:
    """The copyright of the license names the author of the package."""
    project = _project()
    author = cast("list[dict[str, str]]", project["authors"])[0]["name"]

    text = LICENSE.read_text(encoding="utf-8")
    assert re.search(rf"Copyright \(c\) \d{{4}} {re.escape(author)}", text)


def test_the_readme_documents_the_license() -> None:
    """The README tells which license the package is published under."""
    readme = README.read_text(encoding="utf-8")

    assert "### License" in readme
    assert "MIT" in readme.split("### License", 1)[1]
