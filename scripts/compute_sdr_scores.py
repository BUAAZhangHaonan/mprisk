from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.manifests import read_jsonl
from mprisk.state.identity import SOURCE_IDENTITY_FIELDS, homogeneous_identity
from mprisk.state.spherical import compute_spherical_state
from mprisk.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class SdrScoreResult:
    scores_path: Path
    summary_path: Path
    count: int


def compute_sdr_scores(
    *,
    embedding_manifest_path: str | Path,
    output_dir: str | Path,
) -> SdrScoreResult:
    embedding_path = Path(embedding_manifest_path)
    embedding_rows = read_jsonl(embedding_path)
    source_identity = homogeneous_identity(embedding_rows, fields=SOURCE_IDENTITY_FIELDS)
    embedding_sha256 = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    score_rows = [
        _score_row(
            row,
            source_identity=source_identity,
            embedding_manifest_sha256=embedding_sha256,
        )
        for row in embedding_rows
    ]
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
    return SdrScoreResult(scores_path=scores_path, summary_path=summary_path, count=len(score_rows))


def _score_row(
    row: dict[str, Any],
    *,
    source_identity: dict[str, str],
    embedding_manifest_sha256: str,
) -> dict[str, Any]:
    state = compute_spherical_state(row)
    return {
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
        "embedding_manifest_sha256": embedding_manifest_sha256,
        **{key: value for key, value in state.items() if key not in {"sample_id", "sample_type"}},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute S/D/R scores from embedding manifests.")
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_sdr_scores(
        embedding_manifest_path=Path(args.embedding_manifest),
        output_dir=Path(args.output_dir),
    )
    print(f"sdr_scores={result.scores_path}")
    print(f"sdr_score_summary={result.summary_path}")
    print(f"total_samples={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
