"""Strict DeepSeek adapter for GT Description generation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.data.generated_archive_freeze import _canonical_json
from mprisk.ground_truth.providers.base import (
    GTDescriptionProviderRequest,
    GTDescriptionProviderResponse,
    PermanentProviderError,
    TransientProviderError,
)


class DeepSeekProviderSettings(BaseModel):
    """All settings specific to the DeepSeek chat-completions adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_url: str
    api_key_env: str
    env_file: Path
    temperature: Literal[0]
    max_tokens: int
    thinking: Literal["disabled"]
    request_timeout_seconds: float

    @field_validator("api_url", "api_key_env")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DeepSeek text settings must be non-empty")
        return value

    @field_validator("api_url")
    @classmethod
    def api_url_must_be_https(cls, value: str) -> str:
        if not value.startswith("https://") or "/chat/completions" not in value:
            raise ValueError("api_url must be an HTTPS chat-completions endpoint")
        return value

    @field_validator("api_key_env")
    @classmethod
    def api_key_env_must_be_identifier(cls, value: str) -> str:
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", value) is None:
            raise ValueError("api_key_env must be an uppercase environment variable name")
        return value

    @field_validator("max_tokens")
    @classmethod
    def positive_max_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_tokens must be positive")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        return value


def validate_deepseek_settings(provider_settings: Mapping[str, Any]) -> None:
    DeepSeekProviderSettings.model_validate(dict(provider_settings))


def load_api_key(settings: DeepSeekProviderSettings) -> str:
    value = os.environ.get(settings.api_key_env)
    if value:
        return value
    value = _read_env_file(settings.env_file).get(settings.api_key_env)
    if value:
        return value
    raise ValueError(
        f"{settings.api_key_env} is required for the DeepSeek GT Description provider"
    )


class DeepSeekProvider:
    """DeepSeek chat-completions implementation of the provider contract."""

    def __init__(
        self,
        model: str,
        settings: DeepSeekProviderSettings,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("gt_generator_model must be non-empty")
        self.model = model
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds), transport=transport
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def close(self) -> None:
        await self.client.aclose()

    async def complete(
        self, request: GTDescriptionProviderRequest
    ) -> GTDescriptionProviderResponse:
        if request.model != self.model:
            raise PermanentProviderError("Provider request model does not match adapter model")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": _canonical_json(dict(request.model_input))},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.settings.thinking},
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        try:
            response = await self.client.post(
                self.settings.api_url, headers=self.headers, json=body
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientProviderError(type(exc).__name__) from exc
        if response.status_code in {408, 409, 429} or response.status_code >= 500:
            raise TransientProviderError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise PermanentProviderError(f"HTTP {response.status_code}")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise PermanentProviderError("API response is not JSON") from exc
        return _validate_response_envelope(payload, self.model)


def build_deepseek_provider(
    gt_generator_model: str, provider_settings: Mapping[str, Any]
) -> DeepSeekProvider:
    settings = DeepSeekProviderSettings.model_validate(dict(provider_settings))
    return DeepSeekProvider(gt_generator_model, settings, load_api_key(settings))


def _validate_response_envelope(
    payload: Any, model: str
) -> GTDescriptionProviderResponse:
    if not isinstance(payload, dict) or payload.get("model") != model:
        raise PermanentProviderError("API returned an unexpected model")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise PermanentProviderError("API must return exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        raise PermanentProviderError(f"Unexpected finish_reason: {finish_reason!r}")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("reasoning_content") not in (None, ""):
        raise PermanentProviderError("Thinking was not disabled")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise PermanentProviderError("API returned empty content")
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        raise PermanentProviderError("API usage must be an object")
    response_id = payload.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise PermanentProviderError("API response id must be a string or null")
    return GTDescriptionProviderResponse(
        response_id=response_id,
        response_model=payload["model"],
        finish_reason=choice["finish_reason"],
        content=content,
        usage=usage,
        provider_metadata={"system_fingerprint": payload.get("system_fingerprint")},
    )


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            raise ValueError(f"Invalid env line in {path}")
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values
