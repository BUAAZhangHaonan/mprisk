from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.manifests import read_jsonl
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
    embedding_rows = read_jsonl(embedding_manifest_path)
    score_rows = [_score_row(row) for row in embedding_rows]
    output_root = Path(output_dir)
    scores_path = write_jsonl(output_root / "sdr_scores.jsonl", score_rows)
    summary_path = write_json(
        output_root / "sdr_score_summary.json",
        {
            "embedding_manifest": str(embedding_manifest_path),
            "sdr_scores": str(scores_path),
            "total_samples": len(score_rows),
        },
    )
    return SdrScoreResult(scores_path=scores_path, summary_path=summary_path, count=len(score_rows))


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    state = compute_spherical_state(row)
    return {
        "sample_id": row["sample_id"],
        "sample_type": row["sample_type"],
        "model_key": row["model_key"],
        "protocol": row.get("protocol", ""),
        "prompt_set_key": row.get("prompt_set_key", ""),
        "calibration_split": row.get("calibration_split", ""),
        "repr_key": row["repr_key"],
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
