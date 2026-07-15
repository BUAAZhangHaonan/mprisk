from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.manifests import read_jsonl
from mprisk.state.patterns import assign_state, load_thresholds_config
from mprisk.utils.io import write_json, write_jsonl


@dataclass(frozen=True)
class StatePatternResult:
    patterns_path: Path
    summary_path: Path
    count: int


def assign_state_patterns(
    *,
    sdr_scores_path: str | Path,
    thresholds: dict[str, Any] | str | Path,
    output_dir: str | Path,
) -> StatePatternResult:
    threshold_values = load_thresholds_config(thresholds)
    score_rows = read_jsonl(sdr_scores_path)
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
        _summary(
            pattern_rows,
            sdr_scores_path=sdr_scores_path,
            patterns_path=patterns_path,
            thresholds={
                "kappa": threshold_values.kappa,
                "tau": threshold_values.tau,
                "delta_policy": "per_sample_synchronous_prompt_bootstrap_1.96se",
            },
        ),
    )
    return StatePatternResult(
        patterns_path=patterns_path,
        summary_path=summary_path,
        count=len(pattern_rows),
    )


def _summary(
    rows: list[dict[str, Any]],
    *,
    sdr_scores_path: str | Path,
    patterns_path: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    return {
        "sdr_scores": str(sdr_scores_path),
        "state_patterns": str(patterns_path),
        "thresholds": thresholds,
        "total_samples": len(rows),
        "sample_type_counts": dict(Counter(str(row.get("sample_type", "")) for row in rows)),
        "pattern_counts": dict(Counter(str(row.get("pattern", "")) for row in rows)),
        "missing_samples": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign state patterns from S/D/R scores.")
    parser.add_argument("--sdr-scores", required=True)
    parser.add_argument("--thresholds", required=True, help="JSON string or path to JSON config.")
    parser.add_argument("--output-dir", required=True)
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
