"""GT Description generation content validation and export verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mprisk.data.generated_archive_freeze import _canonical_json, _sha256
from mprisk.ground_truth.annotation_inputs import GT_INPUT_SCHEMA_VERSION
from mprisk.utils.io import read_jsonl as _read_jsonl

from mprisk.ground_truth._plan import (
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    GTDescriptionGenerationConfig,
    GTDescriptionGenerationResult,
    GTDescriptionValidationError,
)
from mprisk.ground_truth._planner import (
    _resolve_repo_path,
    load_config,
    prepare_tasks,
)


def validate_gt_description_content(content: Any, *, min_words: int, max_words: int) -> str:
    if not isinstance(content, str) or not content:
        raise GTDescriptionValidationError("response content must be a non-empty string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GTDescriptionValidationError("response content must be exact JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"GT_DESCRIPTION"}:
        raise GTDescriptionValidationError(
            "response JSON must contain exactly GT_DESCRIPTION"
        )
    value = payload["GT_DESCRIPTION"]
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value:
        raise GTDescriptionValidationError(
            "GT_DESCRIPTION must be one non-empty unpadded line"
        )
    if not value.endswith(".") or any(mark in value for mark in "?!"):
        raise GTDescriptionValidationError(
            "GT_DESCRIPTION must be a declarative sentence ending in a period"
        )
    if len(re.findall(r"[.!?](?=\s|$)", value)) != 1:
        raise GTDescriptionValidationError("GT_DESCRIPTION must contain exactly one sentence")
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", value)
    if not min_words <= len(words) <= max_words:
        raise GTDescriptionValidationError(
            f"GT_DESCRIPTION must contain {min_words}-{max_words} English words"
        )
    return value


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def verify_gt_description_generation(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    require_complete: bool,
) -> GTDescriptionGenerationResult:
    root = Path(repo_root).resolve()
    config = load_config(config_path)
    tasks = prepare_tasks(root, config)
    output_root = _resolve_repo_path(root, config.output_root)
    manifest = _read_jsonl(output_root / "gt_manifest.jsonl")
    sidecar = _read_jsonl(output_root / "review_status.jsonl")
    failures = _read_jsonl(output_root / "failures.jsonl")
    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    completed = len(manifest)
    if require_complete and (
        completed != config.expected_count
        or failures
        or len(sidecar) != config.expected_count
    ):
        raise ValueError(
            f"Complete GT export must contain {config.expected_count} rows and no failures"
        )
    input_by_id = {task.sample_id: task.annotation_input_row for task in tasks}
    task_ids = set(input_by_id)
    manifest_ids = [_text(row, "sample_id") for row in manifest]
    sidecar_ids = [_text(row, "sample_id") for row in sidecar]
    failure_ids = [_text(row, "sample_id") for row in failures]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("GT manifest contains duplicate sample_id values")
    if len(sidecar_ids) != len(set(sidecar_ids)):
        raise ValueError("GT review sidecar contains duplicate sample_id values")
    if not set(manifest_ids + sidecar_ids + failure_ids) <= task_ids:
        raise ValueError("GT export contains unknown sample_id values")
    if set(sidecar_ids) != set(manifest_ids):
        raise ValueError("GT review sidecar must exactly match completed manifest rows")
    for row in manifest:
        sample_id = _text(row, "sample_id")
        expected = input_by_id[sample_id]
        if set(row) != set(expected) | {"run_id", "GT_DESCRIPTION"}:
            raise ValueError(f"Final GT manifest fields differ for {sample_id}")
        expected_output = {
            **expected,
            "schema_name": OUTPUT_SCHEMA,
            "run_id": config.run_id,
        }
        if {key: row[key] for key in expected_output} != expected_output:
            raise ValueError(f"Final GT manifest changed annotation input fields for {sample_id}")
        validate_gt_description_content(
            _canonical_json({"GT_DESCRIPTION": row["GT_DESCRIPTION"]}),
            min_words=config.min_words,
            max_words=config.max_words,
        )
    for row in sidecar:
        if set(row) != {
            "sample_id",
            "annotation_status",
            "human_review_status",
            "gt_description_sha256",
        }:
            raise ValueError(f"Unexpected GT review sidecar fields: {row.get('sample_id')}")
        if row["annotation_status"] != "preliminary_ai_draft":
            raise ValueError(f"Unexpected annotation status: {row['sample_id']}")
        if row["human_review_status"] != "pending_human":
            raise ValueError(f"Unexpected human review status: {row['sample_id']}")
    if provenance.get("schema_name") != PROVENANCE_SCHEMA:
        raise ValueError("Unexpected GT provenance schema")
    if (
        provenance.get("run_id") != config.run_id
        or provenance.get("provider_key") != config.provider_key
        or provenance.get("gt_generator_model") != config.gt_generator_model
        or provenance.get("gt_description_schema_name") != OUTPUT_SCHEMA
        or provenance.get("gt_input_schema_version") != GT_INPUT_SCHEMA_VERSION
        or provenance.get("provider_settings_sha256")
        != hashlib.sha256(_canonical_json(config.provider_settings).encode()).hexdigest()
    ):
        raise ValueError("GT provenance run identity mismatch")
    for artifact in provenance["artifacts"].values():
        path = root / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"GT artifact hash mismatch: {path}")
    return GTDescriptionGenerationResult(
        total=config.expected_count,
        completed=completed,
        failed=len(failures),
        pending=config.expected_count - completed - len(failures),
        output_root=output_root,
    )
