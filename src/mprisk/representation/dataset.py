"""Build representation-training datasets from state bundle manifests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mprisk.data.protocol_views import VIEW_KEYS
from mprisk.utils.io import write_json, write_jsonl


VALID_LABELS = {"positive", "negative", "neutral"}
VALID_SAMPLE_TYPES = {"Conflict", "Aligned"}


@dataclass(frozen=True)
class RepresentationDatasetBuildResult:
    dataset_path: Path
    summary_path: Path
    total_input_bundles: int
    exported_rows: int
    skipped_rows: int


def build_representation_dataset(
    *,
    bundle_manifest_path: str | Path,
    output_dir: str | Path,
) -> RepresentationDatasetBuildResult:
    bundles = list(iter_bundle_manifest(bundle_manifest_path))
    output_root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    skipped_rows = 0

    for bundle in bundles:
        for view_key in VIEW_KEYS:
            for prompt_id in _prompt_ids(bundle):
                row = _representation_row(bundle, view_key=view_key, prompt_id=prompt_id)
                if _keep_row(row):
                    rows.append(row)
                else:
                    skipped_rows += 1

    dataset_path = write_jsonl(output_root / "representation_dataset.jsonl", rows)
    summary_path = write_json(
        output_root / "representation_dataset_summary.json",
        {
            "total_input_bundles": len(bundles),
            "exported_rows": len(rows),
            "skipped_rows": skipped_rows,
            "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "sample_type_counts": dict(
                sorted(Counter(row["sample_type"] for row in rows).items())
            ),
        },
    )
    return RepresentationDatasetBuildResult(
        dataset_path=dataset_path,
        summary_path=summary_path,
        total_input_bundles=len(bundles),
        exported_rows=len(rows),
        skipped_rows=skipped_rows,
    )


def iter_bundle_manifest(bundle_manifest_path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(bundle_manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: bundle row must be a JSON object")
            yield row


def _prompt_ids(bundle: dict[str, Any]) -> list[str]:
    prompts = bundle.get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError(f"bundle {bundle.get('sample_id', '<unknown>')}: prompts must be a list")
    prompt_ids: list[str] = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or not prompt.get("prompt_id"):
            raise ValueError(
                f"bundle {bundle.get('sample_id', '<unknown>')}: prompt missing prompt_id"
            )
        prompt_ids.append(str(prompt["prompt_id"]))
    return prompt_ids


def _representation_row(
    bundle: dict[str, Any],
    *,
    view_key: str,
    prompt_id: str,
) -> dict[str, Any]:
    sample_id = str(bundle["sample_id"])
    view_labels = bundle.get("view_labels", {})
    if not isinstance(view_labels, dict):
        raise ValueError(f"bundle {sample_id}: view_labels must be a JSON object")
    view_label = view_labels.get(view_key, {})
    if not isinstance(view_label, dict):
        raise ValueError(f"bundle {sample_id}: view_labels.{view_key} must be a JSON object")
    metadata = bundle.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"bundle {sample_id}: metadata must be a JSON object")

    prompt_conditioned_state = _prompt_conditioned_state(
        bundle, sample_id=sample_id, view_key=view_key, prompt_id=prompt_id
    )
    return {
        "row_id": f"{sample_id}:{view_key}:{prompt_id}",
        "sample_id": sample_id,
        "sample_type": bundle.get("sample_type", ""),
        "model_key": bundle.get("model_key", ""),
        "protocol": bundle.get("protocol", ""),
        "view_key": view_key,
        "prompt_id": prompt_id,
        "prompt_set_key": bundle.get("prompt_set_key", ""),
        "label": view_label.get("label"),
        "specific_affect": view_label.get("specific_affect"),
        "is_clear": view_label.get("is_clear", False),
        "prompt_conditioned_state": prompt_conditioned_state,
        "split_group_id": metadata.get("split_group_id") or sample_id,
        "source_dataset": metadata.get("source_dataset") or "",
    }


def _prompt_conditioned_state(
    bundle: dict[str, Any],
    *,
    sample_id: str,
    view_key: str,
    prompt_id: str,
) -> dict[str, Any]:
    try:
        view = bundle["views"][view_key]
        prompt = view["prompts"][prompt_id]
        state = prompt["prompt_conditioned_state"]
    except KeyError as exc:
        raise ValueError(
            f"bundle {sample_id}: missing prompt_conditioned_state for {view_key}:{prompt_id}"
        ) from exc
    if not isinstance(state, dict):
        raise ValueError(
            f"bundle {sample_id}: prompt_conditioned_state for {view_key}:{prompt_id} "
            "must be a JSON object"
        )
    return dict(state)


def _keep_row(row: dict[str, Any]) -> bool:
    return (
        row["is_clear"] is True
        and row["label"] in VALID_LABELS
        and row["sample_type"] in VALID_SAMPLE_TYPES
    )
