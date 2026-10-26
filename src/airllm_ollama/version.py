"""The version of the package.

The version is derived from the git tags by ``hatch-vcs`` and written to
``_version.py`` when the package is built. A source checkout without that file
falls back to the metadata of the installed package — and to a placeholder
when the package was not installed at all.
"""

try:
    from airllm_ollama._version import __version__
except ImportError:  # pragma: no cover - the file is written when the package is built
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _distribution_version

    try:
        __version__ = _distribution_version("airllm-ollama")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
