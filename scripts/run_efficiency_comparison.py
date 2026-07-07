from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.evaluation.efficiency import compare_t0_posthoc_efficiency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare t0 and posthoc timing efficiency.")
    parser.add_argument("--timing-log", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_t0_posthoc_efficiency(
        timing_log_path=Path(args.timing_log),
        output_dir=Path(args.output_dir),
    )
    print(f"efficiency_comparison_json={result.json_path}")
    print(f"efficiency_comparison_csv={result.csv_path}")
    print(f"total_timing_rows={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
