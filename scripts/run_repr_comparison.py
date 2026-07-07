from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.evaluation.repr_comparison import compare_representations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and trained representation outputs.")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config mapping repr keys to result paths.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repr_results = _load_config(Path(args.config))
    result = compare_representations(
        repr_results=repr_results,
        output_dir=Path(args.output_dir),
    )
    print(f"repr_comparison_json={result.json_path}")
    print(f"repr_comparison_csv={result.csv_path}")
    print(f"total_reprs={result.count}")
    return 0


def _load_config(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    representations = payload.get("representations", payload)
    if not isinstance(representations, dict):
        raise ValueError(f"{path}: representations must be a JSON object")

    config: dict[str, dict[str, str]] = {}
    for repr_key, pathspec in representations.items():
        if not isinstance(pathspec, dict):
            raise ValueError(f"{path}: {repr_key} pathspec must be a JSON object")
        config[str(repr_key)] = {str(key): str(value) for key, value in pathspec.items()}
    return config


if __name__ == "__main__":
    raise SystemExit(main())
