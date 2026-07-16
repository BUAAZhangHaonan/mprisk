"""JSONL manifest reading and writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from mprisk.data.protocol_views import normalize_protocol


class ManifestViews(BaseModel):
    model_config = ConfigDict(extra="allow")

    M1: dict[str, Any]
    M2: dict[str, Any]
    M12: dict[str, Any]

    @field_validator("M1", "M2", "M12")
    @classmethod
    def view_must_be_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("manifest views must be JSON objects")
        return value


class FinalManifestRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    source_dataset: str
    source_id: str
    protocol: str
    sample_type: str
    split_group_id: str
    views: ManifestViews
    media_paths: dict[str, str]
    use_in_main: bool

    @field_validator("sample_id", "source_dataset", "source_id", "sample_type", "split_group_id")
    @classmethod
    def text_field_must_not_be_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @field_validator("protocol", mode="before")
    @classmethod
    def protocol_must_be_supported(cls, value: object) -> str:
        if value is None:
            raise ValueError("protocol is required")
        return normalize_protocol(str(value))

    @field_validator("sample_type")
    @classmethod
    def sample_type_must_be_supported(cls, value: str) -> str:
        if value not in {"Conflict", "Ambiguous", "Aligned"}:
            raise ValueError("sample_type must be one of Conflict, Ambiguous, Aligned")
        return value

    @field_validator("media_paths")
    @classmethod
    def media_paths_must_be_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ValueError("media_paths must be a JSON object")
        return value


def is_manifest_placeholder(row: dict[str, Any]) -> bool:
    return "schema" in row and "sample_id" not in row


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_final_manifest(
    path: str | Path,
    *,
    sample_type: str | None = None,
    protocol: str | None = None,
    source_dataset: str | None = None,
    use_in_main: bool | None = None,
) -> list[FinalManifestRow]:
    rows: list[FinalManifestRow] = []
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if is_manifest_placeholder(raw):
                continue
            try:
                rows.append(FinalManifestRow.model_validate(raw))
            except ValueError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
    return filter_manifest_rows(
        rows,
        sample_type=sample_type,
        protocol=protocol,
        source_dataset=source_dataset,
        use_in_main=use_in_main,
    )


def filter_manifest_rows(
    rows: list[FinalManifestRow],
    *,
    sample_type: str | None = None,
    protocol: str | None = None,
    source_dataset: str | None = None,
    use_in_main: bool | None = None,
) -> list[FinalManifestRow]:
    normalized_protocol = normalize_protocol(protocol) if protocol is not None else None
    return [
        row
        for row in rows
        if (sample_type is None or row.sample_type == sample_type)
        and (normalized_protocol is None or row.protocol == normalized_protocol)
        and (source_dataset is None or row.source_dataset == source_dataset)
        and (use_in_main is None or row.use_in_main is use_in_main)
    ]


read_manifest = read_final_manifest
