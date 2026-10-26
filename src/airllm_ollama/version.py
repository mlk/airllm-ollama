"""The version of the package.

The version is derived from the git tags by ``hatch-vcs`` and written to
``_version.py`` when the package is built. A source checkout without that file
falls back to the metadata of the installed package.
"""

try:
    from airllm_ollama._version import __version__
except ImportError:  # pragma: no cover - the file is written when the package is built
    from importlib.metadata import version as _distribution_version

    __version__ = _distribution_version("airllm-ollama")

__all__ = ["__version__"]
