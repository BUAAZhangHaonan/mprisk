"""Prompt cache manifest contract and lookup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from mprisk.data.protocol_views import normalize_protocol
from mprisk.utils.io import write_jsonl


REQUIRED_PROMPT_CACHE_FIELDS = (
    "model_key",
    "prompt_set_key",
    "prompt_id",
    "protocol",
    "cache_key",
)


class PromptCacheManifest:
    """JSONL-backed prompt cache lookup keyed by model, prompt set, prompt, and protocol."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows = [_validate_prompt_cache_row(row) for row in rows]
        self._rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in self.rows:
            self._rows_by_key.setdefault(_row_key(row), row)

    def lookup(
        self,
        *,
        model_key: str,
        prompt_set_key: str,
        prompt_id: str,
        protocol: str,
    ) -> dict[str, Any] | None:
        return self._rows_by_key.get(
            _key(
                model_key=model_key,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
                protocol=protocol,
            )
        )

    def missing_prompt_ids(
        self,
        *,
        model_key: str,
        prompt_set_key: str,
        prompt_ids: Iterable[str],
        protocol: str,
    ) -> list[str]:
        return [
            prompt_id
            for prompt_id in prompt_ids
            if self.lookup(
                model_key=model_key,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
                protocol=protocol,
            )
            is None
        ]

    def rows_for_prompt_ids(
        self,
        *,
        model_key: str,
        prompt_set_key: str,
        prompt_ids: Iterable[str],
        protocol: str,
    ) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for prompt_id in prompt_ids:
            row = self.lookup(
                model_key=model_key,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
                protocol=protocol,
            )
            if row is not None:
                rows[prompt_id] = row
        return rows


def read_prompt_cache_rows(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                rows.append(_validate_prompt_cache_row(row))
            except ValueError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
    return rows


def load_prompt_cache_manifest(path: str | Path) -> PromptCacheManifest:
    return PromptCacheManifest(read_prompt_cache_rows(path))


def write_prompt_cache_manifest(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    return write_jsonl(path, [_validate_prompt_cache_row(row) for row in rows])


def _validate_prompt_cache_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("prompt cache row must be a JSON object")
    clean = dict(row)
    for field in REQUIRED_PROMPT_CACHE_FIELDS:
        value = clean.get(field)
        if value is None or value == "":
            raise ValueError(f"prompt cache row missing required field {field}")
        clean[field] = str(value)
    clean["protocol"] = normalize_protocol(clean["protocol"])
    return clean


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return _key(
        model_key=row["model_key"],
        prompt_set_key=row["prompt_set_key"],
        prompt_id=row["prompt_id"],
        protocol=row["protocol"],
    )


def _key(
    *,
    model_key: str,
    prompt_set_key: str,
    prompt_id: str,
    protocol: str,
) -> tuple[str, str, str, str]:
    return (
        str(model_key),
        str(prompt_set_key),
        str(prompt_id),
        normalize_protocol(protocol),
    )
