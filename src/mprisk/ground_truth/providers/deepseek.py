"""DeepSeek transport for GT Description generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from mprisk.data.generated_archive_freeze import _canonical_json
from mprisk.ground_truth.description_generation import (
    GTDescriptionGenerationConfig,
    GTDescriptionGenerationTask,
)


class TransientProviderError(RuntimeError):
    """A retryable provider or transport failure."""


class PermanentProviderError(RuntimeError):
    """A non-retryable provider response failure."""


def load_api_key(config: GTDescriptionGenerationConfig) -> str:
    value = os.environ.get(config.api_key_variable)
    if value:
        return value
    value = _read_env_file(config.env_file).get(config.api_key_variable)
    if value:
        return value
    raise ValueError("DEEPSEEK_API_KEY is required for GT Description generation")


class DeepSeekClient:
    """DeepSeek chat-completions client for the generic GT Description task."""

    def __init__(
        self,
        config: GTDescriptionGenerationConfig,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_seconds), transport=transport
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def close(self) -> None:
        await self.client.aclose()

    async def complete(self, task: GTDescriptionGenerationTask) -> dict[str, Any]:
        body = {
            "model": self.config.gt_generator_model,
            "messages": [
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": _canonical_json(task.model_input)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        try:
            response = await self.client.post(
                self.config.api_url, headers=self.headers, json=body
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
        return _validate_response_envelope(payload, self.config.gt_generator_model)


def _validate_response_envelope(payload: Any, model: str) -> dict[str, Any]:
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
    return {
        "response_id": payload.get("id"),
        "response_model": payload.get("model"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "usage": payload.get("usage") or {},
    }


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
