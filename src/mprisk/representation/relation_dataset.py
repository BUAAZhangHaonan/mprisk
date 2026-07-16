"""Build sample-level three-condition relation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.utils.io import write_json, write_jsonl

CONDITIONS = ("M1", "M2", "M12")
LABEL_TO_ID = {"Aligned": 0, "Conflict": 1}
FORBIDDEN_FIELDS = frozenset(
    {"misread", "binary_label", "final_decision", "misread_label", "misread_binary_label"}
)


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
    prompt_set_key: str,
    prompt_set_artifact_sha256: str,
    expected_prompt_count: int,
    expected_prompt_ids: tuple[str, ...] | list[str],
) -> RelationDatasetBuildResult:
    expected_ids = set(expected_prompt_ids)
    if (
        expected_prompt_count <= 0
        or len(expected_prompt_ids) != expected_prompt_count
        or len(expected_ids) != expected_prompt_count
    ):
        raise ValueError("prompt contract requires exact unique expected_prompt_ids")
    if len(prompt_set_artifact_sha256) != 64:
        raise ValueError("prompt_set_artifact_sha256 must be a SHA-256 digest")
    bundles = _read_jsonl(bundle_manifest_path)
    if not bundles:
        raise ValueError("bundle manifest is empty")
    rows: list[dict[str, Any]] = []
    model_keys: set[str] = set()
    sample_ids: set[str] = set()
    split_assignment_keys: set[str] = set()
    split_assignment_checksums: set[str] = set()
    for bundle in bundles:
        _reject_forbidden_fields(bundle)
        sample_id = _required_text(bundle, "sample_id")
        sample_type = _required_text(bundle, "sample_type")
        if sample_type not in LABEL_TO_ID:
            raise ValueError("relation samples must use Conflict or Aligned sample labels")
        model_key = _required_text(bundle, "model_key")
        if _required_text(bundle, "prompt_set_key") != prompt_set_key:
            raise ValueError("bundle prompt_set_key does not match the prompt contract")
        model_keys.add(model_key)
        sample_ids.add(sample_id)
        metadata = bundle.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("bundle metadata must be a JSON object")
        master_split = _required_metadata_text(metadata, "master_split")
        representation_split = _required_metadata_text(metadata, "representation_split")
        split_group_id = _required_metadata_text(metadata, "split_group_id")
        split_assignment_key = _required_metadata_text(metadata, "split_assignment_key")
        split_assignment_sha256 = _required_metadata_text(
            metadata, "split_assignment_sha256"
        )
        split_assignment_keys.add(split_assignment_key)
        split_assignment_checksums.add(split_assignment_sha256)
        if len(split_assignment_sha256) != 64:
            raise ValueError("split_assignment_sha256 must be a SHA-256 digest")
        expected_master = {
            "relation_train": "train",
            "relation_val": "val",
            "aligned_calibration": "val",
            "official_test": "test",
        }
        if expected_master.get(representation_split) != master_split:
            raise ValueError("representation_split mismatches master_split")
        calibration_split = str(metadata.get("calibration_split") or "")
        expected_calibration = (
            "aligned_calibration" if representation_split == "aligned_calibration" else ""
        )
        if calibration_split != expected_calibration:
            raise ValueError("calibration_split mismatches representation_split")
        prompt_ids = _prompt_ids(bundle)
        if len(prompt_ids) != expected_prompt_count or set(prompt_ids) != expected_ids:
            raise ValueError(
                f"sample {sample_id} must use exactly the configured "
                f"{expected_prompt_count} prompt IDs"
            )
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
                    "prompt_set_artifact_sha256": prompt_set_artifact_sha256,
                    "prompt_id": prompt_id,
                    "split_group_id": split_group_id,
                    "master_split": master_split,
                    "representation_split": representation_split,
                    "calibration_split": calibration_split,
                    "split_assignment_key": split_assignment_key,
                    "split_assignment_sha256": split_assignment_sha256,
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
    if len(split_assignment_keys) != 1 or len(split_assignment_checksums) != 1:
        raise ValueError("relation dataset must use one registered split assignment")
    output_root = Path(output_dir)
    dataset_path = write_jsonl(output_root / "relation_dataset.jsonl", rows)
    summary_path = write_json(
        output_root / "relation_dataset_summary.json",
        {
            "schema": "mprisk_relation_dataset_summary_v1",
            "bundle_manifest": str(bundle_manifest_path),
            "model_key": next(iter(model_keys), None),
            "prompt_set_key": prompt_set_key,
            "prompt_set_artifact_sha256": prompt_set_artifact_sha256,
            "expected_prompt_count": expected_prompt_count,
            "expected_prompt_ids": list(expected_prompt_ids),
            "sample_count": len(sample_ids),
            "row_count": len(rows),
            "label_counts": {
                label: sum(row["sample_type"] == label for row in rows)
                for label in LABEL_TO_ID
            },
            "representation_split_counts": {
                split: sum(row["representation_split"] == split for row in rows)
                for split in sorted({row["representation_split"] for row in rows})
            },
            "split_assignment_key": next(iter(split_assignment_keys)),
            "split_assignment_sha256": next(iter(split_assignment_checksums)),
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


def _required_metadata_text(metadata: dict[str, Any], field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"bundle metadata {field} must be non-empty text")
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
