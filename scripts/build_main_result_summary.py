from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.evaluation.reporting import build_main_result_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the main result summary Markdown report.")
    parser.add_argument("--conflict-vs-aligned")
    parser.add_argument("--state-to-error")
    parser.add_argument("--repr-comparison")
    parser.add_argument("--efficiency-comparison")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_main_result_summary(
        conflict_vs_aligned_path=_optional_path(args.conflict_vs_aligned),
        state_to_error_path=_optional_path(args.state_to_error),
        repr_comparison_path=_optional_path(args.repr_comparison),
        efficiency_comparison_path=_optional_path(args.efficiency_comparison),
        output_dir=Path(args.output_dir),
    )
    print(f"main_result_summary={result.summary_path}")
    if result.missing_inputs:
        print("missing_inputs=" + ",".join(result.missing_inputs))
    return 0


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


if __name__ == "__main__":
    raise SystemExit(main())
