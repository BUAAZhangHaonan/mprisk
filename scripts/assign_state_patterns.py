"""CLI wrapper for assign_state_patterns (canonical impl in mprisk.state.pipeline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.state.pipeline import StatePatternResult, assign_state_patterns  # noqa: F401  (re-export)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign state patterns from S/D/R scores.")
    parser.add_argument("--sdr-scores", required=True)
    parser.add_argument("--thresholds", required=True, help="JSON string or path to JSON config.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-root",
        default=None,
        help=(
            "Optional informational flag naming the cache family "
            "(e.g. Source vs Target) the SDR scores come from. Not consumed "
            "by assign_state_patterns; accepted for driver-script self-documentation."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = assign_state_patterns(
        sdr_scores_path=Path(args.sdr_scores),
        thresholds=args.thresholds,
        output_dir=Path(args.output_dir),
    )
    print(f"state_patterns={result.patterns_path}")
    print(f"state_summary={result.summary_path}")
    print(f"total_samples={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
