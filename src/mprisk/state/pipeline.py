"""Package-level S/D/R scoring and State Pattern artifact writers."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.data.manifests import read_jsonl
from mprisk.state.identity import (
    SOURCE_IDENTITY_FIELDS,
    homogeneous_identity,
    require_matching_identity,
)
from mprisk.state.patterns import assign_state, load_thresholds_config
from mprisk.state.spherical import compute_spherical_state, require_exact_sdr_rows
from mprisk.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class SdrScoreResult:
    scores_path: Path
    summary_path: Path
    count: int


@dataclass(frozen=True)
class StatePatternResult:
    patterns_path: Path
    summary_path: Path
    count: int


def compute_sdr_scores(
    *, embedding_manifest_path: str | Path, output_dir: str | Path
) -> SdrScoreResult:
    embedding_path = Path(embedding_manifest_path)
    embedding_rows = read_jsonl(embedding_path)
    source_identity = homogeneous_identity(embedding_rows, fields=SOURCE_IDENTITY_FIELDS)
    embedding_sha256 = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    score_rows = []
    for row in embedding_rows:
        state = compute_spherical_state(row)
        score_rows.append(
            {
                "sample_id": row["sample_id"],
                "sample_type": row["sample_type"],
                "model_key": row["model_key"],
                "protocol": row.get("protocol", ""),
                "prompt_set_key": row.get("prompt_set_key", ""),
                "split_group_id": row.get("split_group_id", ""),
                "master_split": row.get("master_split", ""),
                "representation_split": row.get("representation_split", ""),
                "calibration_split": row.get("calibration_split", ""),
                "split_assignment_key": row.get("split_assignment_key", ""),
                "split_assignment_sha256": row.get("split_assignment_sha256", ""),
                "repr_key": row["repr_key"],
                **source_identity,
                "embedding_manifest_sha256": embedding_sha256,
                **{
                    key: value
                    for key, value in state.items()
                    if key not in {"sample_id", "sample_type"}
                },
            }
        )
    output_root = Path(output_dir)
    scores_path = write_jsonl(output_root / "sdr_scores.jsonl", score_rows)
    summary_path = write_json(
        output_root / "sdr_score_summary.json",
        {
            "embedding_manifest": str(embedding_manifest_path),
            "sdr_scores": str(scores_path),
            "total_samples": len(score_rows),
            **source_identity,
            "embedding_manifest_sha256": embedding_sha256,
        },
    )
    return SdrScoreResult(scores_path, summary_path, len(score_rows))


def assign_state_patterns(
    *,
    sdr_scores_path: str | Path,
    thresholds: dict[str, Any] | str | Path,
    output_dir: str | Path,
) -> StatePatternResult:
    threshold_values = load_thresholds_config(thresholds)
    score_rows = read_jsonl(sdr_scores_path)
    require_exact_sdr_rows(score_rows)
    if threshold_values.identity is None:
        raise ValueError("state pattern assignment requires identity-bound calibration")
    require_matching_identity(score_rows, threshold_values.identity)
    pattern_rows = [
        {
            **row,
            "pattern": assign_state(
                row["S_mean"],
                row["D"],
                row["R"],
                threshold_values,
                delta_i=row["delta_i"],
            ).value,
        }
        for row in score_rows
    ]
    output_root = Path(output_dir)
    patterns_path = write_jsonl(output_root / "state_patterns.jsonl", pattern_rows)
    summary_path = write_json(
        output_root / "state_summary.json",
        {
            "sdr_scores": str(sdr_scores_path),
            "state_patterns": str(patterns_path),
            "thresholds": {
                "kappa": threshold_values.kappa,
                "tau": threshold_values.tau,
                "delta_policy": "per_sample_synchronous_prompt_bootstrap_1.96se",
            },
            "total_samples": len(pattern_rows),
            "sample_type_counts": dict(
                Counter(str(row.get("sample_type", "")) for row in pattern_rows)
            ),
            "pattern_counts": dict(Counter(str(row["pattern"]) for row in pattern_rows)),
            "missing_samples": 0,
        },
    )
    return StatePatternResult(patterns_path, summary_path, len(pattern_rows))
