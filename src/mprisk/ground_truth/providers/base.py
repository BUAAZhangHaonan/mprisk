"""Provider-neutral contracts for GT Description generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class TransientProviderError(RuntimeError):
    """A retryable provider or transport failure."""


class PermanentProviderError(RuntimeError):
    """A non-retryable provider response failure."""


@dataclass(frozen=True)
class GTDescriptionProviderRequest:
    """The complete provider-neutral request for one GT description."""

    model: str
    system_prompt: str
    model_input: Mapping[str, Any]


@dataclass(frozen=True)
class GTDescriptionProviderResponse:
    """Normalized response returned by every provider adapter."""

    response_id: str | None
    response_model: str
    finish_reason: str
    content: str
    usage: Mapping[str, Any]
    provider_metadata: Mapping[str, Any]


class GTDescriptionProvider(Protocol):
    """Provider adapter used by the generic GT Description task."""

    async def complete(
        self, request: GTDescriptionProviderRequest
    ) -> GTDescriptionProviderResponse: ...

    async def close(self) -> None: ...
