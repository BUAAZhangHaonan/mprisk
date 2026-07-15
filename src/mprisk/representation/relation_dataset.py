"""Build sample-level three-condition relation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.utils.io import write_json, write_jsonl

CONDITIONS = ("M1", "M2", "M12")
LABEL_TO_ID = {"Aligned": 0, "Conflict": 1}
FORBIDDEN_FIELDS = frozenset({"misread", "binary_label", "final_decision"})


@dataclass(frozen=True)
class RelationDatasetBuildResult:
    dataset_path: Path
    summary_path: Path
    sample_count: int
    row_count: int


def build_relation_dataset(
    *,
    bundle_manifest_path: str | Path,
    output_dir: str | Path,
) -> RelationDatasetBuildResult:
    bundles = _read_jsonl(bundle_manifest_path)
    rows: list[dict[str, Any]] = []
    model_keys: set[str] = set()
    sample_ids: set[str] = set()
    for bundle in bundles:
        _reject_forbidden_fields(bundle)
        sample_id = _required_text(bundle, "sample_id")
        sample_type = _required_text(bundle, "sample_type")
        if sample_type not in LABEL_TO_ID:
            raise ValueError("relation samples must use Conflict or Aligned sample labels")
        model_key = _required_text(bundle, "model_key")
        model_keys.add(model_key)
        sample_ids.add(sample_id)
        metadata = bundle.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("bundle metadata must be a JSON object")
        prompt_ids = _prompt_ids(bundle)
        _require_synchronized_prompts(bundle, prompt_ids)
        for prompt_id in prompt_ids:
            rows.append(
                {
                    "schema": "mprisk_relation_sample_v1",
                    "row_id": f"{sample_id}:{prompt_id}",
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "label_id": LABEL_TO_ID[sample_type],
                    "model_key": model_key,
                    "protocol": _required_text(bundle, "protocol"),
                    "prompt_set_key": _required_text(bundle, "prompt_set_key"),
                    "prompt_id": prompt_id,
                    "split_group_id": str(metadata.get("split_group_id") or sample_id),
                    "master_split": str(metadata.get("master_split") or ""),
                    "calibration_split": str(
                        metadata.get("calibration_split")
                        or bundle.get("calibration_split")
                        or ""
                    ),
                    "conditions": {
                        condition: dict(
                            bundle["views"][condition]["prompts"][prompt_id][
                                "prompt_conditioned_state"
                            ]
                        )
                        for condition in CONDITIONS
                    },
                }
            )
    if len(model_keys) > 1:
        raise ValueError("each relation dataset must contain exactly one backbone model_key")
    output_root = Path(output_dir)
    dataset_path = write_jsonl(output_root / "relation_dataset.jsonl", rows)
    summary_path = write_json(
        output_root / "relation_dataset_summary.json",
        {
            "schema": "mprisk_relation_dataset_summary_v1",
            "bundle_manifest": str(bundle_manifest_path),
            "model_key": next(iter(model_keys), None),
            "sample_count": len(sample_ids),
            "row_count": len(rows),
            "label_counts": {
                label: sum(row["sample_type"] == label for row in rows)
                for label in LABEL_TO_ID
            },
        },
    )
    return RelationDatasetBuildResult(dataset_path, summary_path, len(sample_ids), len(rows))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: bundle must be a JSON object")
            rows.append(row)
    return rows


def _reject_forbidden_fields(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_FIELDS or "misread" in normalized:
                raise ValueError(f"Misread fields are forbidden in relation data: {path}.{key}")
            _reject_forbidden_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=f"{path}[{index}]")


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"bundle field {field} must be non-empty text")
    return value


def _prompt_ids(bundle: dict[str, Any]) -> list[str]:
    prompts = bundle.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("bundle prompts must be a non-empty list")
    prompt_ids = [
        str(prompt.get("prompt_id", ""))
        for prompt in prompts
        if isinstance(prompt, dict)
    ]
    if len(prompt_ids) != len(prompts) or any(not value for value in prompt_ids):
        raise ValueError("every bundle prompt must have a prompt_id")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("bundle prompt IDs must be unique")
    return prompt_ids


def _require_synchronized_prompts(bundle: dict[str, Any], prompt_ids: list[str]) -> None:
    views = bundle.get("views")
    if not isinstance(views, dict) or set(views) != set(CONDITIONS):
        raise ValueError("bundle views must contain exactly M1, M2, and M12")
    expected = set(prompt_ids)
    for condition in CONDITIONS:
        prompts = views[condition].get("prompts") if isinstance(views[condition], dict) else None
        if not isinstance(prompts, dict) or set(prompts) != expected:
            raise ValueError("all three conditions must use synchronized prompt IDs")
        for prompt_id, payload in prompts.items():
            state = payload.get("prompt_conditioned_state") if isinstance(payload, dict) else None
            if not isinstance(state, dict):
                raise ValueError(f"missing prompt-conditioned state for {condition}:{prompt_id}")
