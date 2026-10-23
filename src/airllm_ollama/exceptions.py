"""Exception hierarchy for the AirLLM Ollama wrapper."""

from __future__ import annotations


class AirLLMError(Exception):
    """Base class for every error raised by this package."""


class ModelMismatchError(AirLLMError):
    """Raised when a client requests a model this server does not serve."""


class ModelInitializationError(AirLLMError):
    """Raised when the AirLLM model cannot be loaded or is unavailable."""
