"""Build the relation dataset from a completed cache and pick official_test rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mprisk.data.manifests import read_final_manifest, read_jsonl
from mprisk.data.representation_splits import load_representation_split_assignment
from mprisk.experiments._io_utils import _json_sha256, _one, _sha256
from mprisk.experiments.jobs import (
    CALIBRATION,
    CONDITIONS,
    OFFICIAL_TEST,
    TRAINING_SPLITS,
    CacheJob,
)
from mprisk.representation.relation_dataset import LABEL_TO_ID
from mprisk.representation.training import load_training_config
from mprisk.utils.io import write_json, write_jsonl


def build_relation_dataset_from_cache(
    job: CacheJob,
    *,
    split_assignment_path: str | Path,
    training_config_path: str | Path,
    output_dir: str | Path,
    cache_gate: dict[str, Any],
) -> tuple[Path, Path]:
    config = load_training_config(training_config_path)
    if (
        config.model_key != job.model_key
        or config.protocol != job.protocol
        or config.prompt_set_key != job.prompt_set_key
    ):
        raise ValueError("training config identity does not match completed cache")
    if config.prompt_set_artifact_sha256 != _sha256(job.prompt_set):
        raise ValueError("training config prompt artifact SHA does not match the prompt YAML")
    prompt_ids = tuple(config.expected_prompt_ids)
    source_rows = [
        row
        for row in read_final_manifest(job.source_manifest, protocol=job.protocol)
        if row.sample_type in LABEL_TO_ID
    ]
    assignments = load_representation_split_assignment(split_assignment_path)
    split_sha = _sha256(Path(split_assignment_path))
    cache_rows = read_jsonl(job.cache_root / "manifest.jsonl")
    by_key = {
        (str(row["sample_id"]), str(row["prompt_id"]), str(row["condition"])): row
        for row in cache_rows
    }
    relation_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    seen_samples: set[str] = set()
    for source in source_rows:
        assignment = assignments.get(source.split_group_id)
        if assignment is None or source.sample_id not in set(map(str, assignment["sample_ids"])):
            raise ValueError(f"sample {source.sample_id} is absent from the registered split")
        master_split = str(assignment["master_split"])
        representation_split = str(assignment["representation_split"])
        if str(source.model_dump().get("split")) != master_split:
            raise ValueError("source master split disagrees with the registered assignment")
        calibration_split = CALIBRATION if representation_split == CALIBRATION else ""
        seen_samples.add(source.sample_id)
        split_counts[representation_split] += 1
        type_counts[source.sample_type] += 1
        for prompt_id in prompt_ids:
            conditions = {
                condition: by_key[(source.sample_id, prompt_id, condition)]
                for condition in CONDITIONS
            }
            relation_rows.append(
                {
                    "schema": "mprisk_relation_sample_v1",
                    "row_id": f"{source.sample_id}:{prompt_id}",
                    "sample_id": source.sample_id,
                    "sample_type": source.sample_type,
                    "label_id": LABEL_TO_ID[source.sample_type],
                    "model_key": job.model_key,
                    "protocol": job.protocol,
                    "prompt_set_key": job.prompt_set_key,
                    "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
                    "prompt_id": prompt_id,
                    "split_group_id": source.split_group_id,
                    "master_split": master_split,
                    "representation_split": representation_split,
                    "calibration_split": calibration_split,
                    "split_assignment_key": str(assignment["config_key"]),
                    "split_assignment_sha256": split_sha,
                    "conditions": conditions,
                }
            )
    if len(seen_samples) * 8 * 3 != job.expected_tasks:
        raise ValueError("source/split sample scope does not match the complete cache grid")
    output_root = Path(output_dir)
    dataset_path = write_jsonl(output_root / "relation_dataset.jsonl", relation_rows)
    summary = {
        "schema": "mprisk_relation_dataset_from_prefill_summary_v1",
        "model_key": job.model_key,
        "protocol": job.protocol,
        "seed": job.seed,
        "prompt_set_key": job.prompt_set_key,
        "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
        "cache_manifest_sha256": cache_gate["manifest_sha256"],
        "cache_gate_sha256": _json_sha256(cache_gate),
        "split_assignment_key": next(iter(assignments.values()))["config_key"],
        "split_assignment_sha256": split_sha,
        "sample_count": len(seen_samples),
        "row_count": len(relation_rows),
        "representation_split_counts": dict(sorted(split_counts.items())),
        "sample_type_counts": dict(sorted(type_counts.items())),
        "expected_prompt_count": 8,
        "expected_prompt_ids": list(prompt_ids),
        "dataset_sha256": _sha256(dataset_path),
    }
    summary_path = write_json(output_root / "relation_dataset_summary.json", summary)
    return dataset_path, summary_path


def official_test_rows(
    rows: Iterable[dict[str, Any]], *, source_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = list(rows)
    if not all_rows:
        raise ValueError(f"{source_name} is empty")
    split_counts = Counter(str(row.get("representation_split", "")) for row in all_rows)
    included = [row for row in all_rows if row.get("representation_split") == OFFICIAL_TEST]
    if not included:
        raise ValueError(f"{source_name} has no official_test rows")
    forbidden = [
        row
        for row in included
        if row.get("representation_split") in TRAINING_SPLITS | {CALIBRATION}
    ]
    if forbidden:
        raise ValueError("official paper inputs include training/calibration rows")
    sample_ids = [str(row["sample_id"]) for row in included]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("official paper inputs require one row per sample")
    identity = {
        "selection_rule": "representation_split=official_test",
        "source_name": source_name,
        "source_count": len(all_rows),
        "source_split_counts": dict(sorted(split_counts.items())),
        "included_count": len(included),
        "included_sample_ids_sha256": hashlib.sha256(
            json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "split_assignment_key": _one(included, "split_assignment_key"),
        "split_assignment_sha256": _one(included, "split_assignment_sha256"),
    }
    return included, identity
