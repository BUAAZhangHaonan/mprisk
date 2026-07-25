"""CLI wrapper for compute_sdr_scores (canonical impl in mprisk.state.pipeline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.state.pipeline import SdrScoreResult, compute_sdr_scores  # noqa: F401  (re-export)


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
