from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.evaluation.main_results import compare_conflict_vs_aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Conflict and Aligned state differences.")
    parser.add_argument("--sdr-scores", required=True)
    parser.add_argument("--state-patterns", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_conflict_vs_aligned(
        sdr_scores_path=Path(args.sdr_scores),
        state_patterns_path=Path(args.state_patterns),
        output_dir=Path(args.output_dir),
    )
    print(f"conflict_vs_aligned_json={result.json_path}")
    print(f"conflict_vs_aligned_csv={result.csv_path}")
    print(f"total_samples={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
