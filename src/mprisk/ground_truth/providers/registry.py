"""Explicit provider registry for GT Description generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from mprisk.ground_truth.providers.base import GTDescriptionProvider
from mprisk.ground_truth.providers.deepseek import (
    build_deepseek_provider,
    validate_deepseek_settings,
)

ProviderFactory = Callable[[str, Mapping[str, Any]], GTDescriptionProvider]
SettingsValidator = Callable[[Mapping[str, Any]], None]

_PROVIDERS: dict[str, tuple[SettingsValidator, ProviderFactory]] = {
    "deepseek": (validate_deepseek_settings, build_deepseek_provider),
}


def validate_provider_settings(
    provider_key: str, provider_settings: Mapping[str, Any]
) -> None:
    """Validate settings with the selected adapter without opening a client."""

    _require_mapping(provider_settings)
    validator, _ = _resolve(provider_key)
    validator(provider_settings)


def get_provider(
    provider_key: str,
    gt_generator_model: str,
    provider_settings: Mapping[str, Any],
) -> GTDescriptionProvider:
    """Construct exactly the requested provider; unknown keys never fall back."""

    _require_mapping(provider_settings)
    _, factory = _resolve(provider_key)
    return factory(gt_generator_model, provider_settings)


def _resolve(provider_key: str) -> tuple[SettingsValidator, ProviderFactory]:
    if not isinstance(provider_key, str) or not provider_key.strip():
        raise ValueError("provider_key must be a non-empty string")
    registration = _PROVIDERS.get(provider_key)
    if registration is None:
        raise ValueError(f"Unknown GT Description provider: {provider_key!r}")
    return registration


def _require_mapping(provider_settings: Mapping[str, Any]) -> None:
    if not isinstance(provider_settings, Mapping):
        raise TypeError("provider_settings must be a mapping")
