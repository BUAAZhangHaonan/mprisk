"""GT Description generation plan primitives and constants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

CONFIG_SCHEMA = "mprisk_gt_description_generation_config_v3"
OUTPUT_SCHEMA = "mprisk_gt_description_v1"
PROVENANCE_SCHEMA = "mprisk_gt_description_generation_provenance_v3"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GTDescriptionGenerationConfig(_StrictModel):
    schema_name: Literal["mprisk_gt_description_generation_config_v3"]
    run_id: str
    provider_key: str
    gt_generator_model: str
    provider_settings: dict[str, Any]
    concurrency: int
    retry_delays_seconds: list[float]
    min_words: int
    max_words: int
    gt_input_schema_version: Literal["gt_annotation_input_v1"]
    input_manifest: Path
    input_manifest_sha256: str
    expected_count: int
    output_root: Path
    conflict_prompt_path: Path
    aligned_prompt_path: Path

    @field_validator("run_id")
    @classmethod
    def run_id_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id must be non-empty")
        return value

    @field_validator("provider_key", "gt_generator_model")
    @classmethod
    def provider_identity_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider_key and gt_generator_model must be non-empty")
        return value

    @field_validator("concurrency", "expected_count")
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("concurrency and expected_count must be positive")
        return value

    @field_validator("retry_delays_seconds")
    @classmethod
    def retry_delays_must_be_non_negative(cls, value: list[float]) -> list[float]:
        if any(delay < 0 for delay in value):
            raise ValueError("retry delays must be non-negative")
        return value

    @field_validator("max_words")
    @classmethod
    def word_range_must_be_valid(cls, value: int, info: Any) -> int:
        min_words = info.data.get("min_words")
        if not isinstance(min_words, int) or min_words <= 0 or value < min_words:
            raise ValueError("word limits must satisfy 0 < min_words <= max_words")
        return value

    @field_validator("input_manifest_sha256")
    @classmethod
    def manifest_hash_must_be_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("input_manifest_sha256 must be a lowercase SHA-256 digest")
        return value


@dataclass(frozen=True)
class GTDescriptionGenerationTask:
    order: int
    sample_id: str
    sample_type: Literal["Conflict", "Aligned"]
    source_archive: str
    input_hash: str
    prompt_hash: str
    system_prompt: str
    model_input: dict[str, Any]
    annotation_input_row: dict[str, Any]
    ledger_signature: dict[str, Any]


@dataclass(frozen=True)
class GTDescriptionGenerationResult:
    total: int
    completed: int
    failed: int
    pending: int
    output_root: Path


class GTDescriptionValidationError(ValueError):
    pass
