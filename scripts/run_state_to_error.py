from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.evaluation.error_analysis import analyze_state_to_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze state patterns against prediction errors.")
    parser.add_argument("--state-patterns", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_state_to_error(
        state_patterns_path=Path(args.state_patterns),
        predictions_path=Path(args.predictions),
        output_dir=Path(args.output_dir),
    )
    print(f"state_to_error_json={result.json_path}")
    print(f"state_to_error_csv={result.csv_path}")
    print(f"total_samples={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
