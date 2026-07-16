from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.ground_truth.description_generation import run_gt_description_generation


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run resumable GT Description generation.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry failed ledger rows; default resume runs pending rows only.",
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/ground_truth/gt_description_generation_pilot_v1.yaml",
    )
    args = parser.parse_args()
    result = asyncio.run(
        run_gt_description_generation(
            repo_root=args.repo_root,
            config_path=args.config,
            retry_failed=args.retry_failed,
        )
    )
    print(json.dumps(result.__dict__, default=str, sort_keys=True))
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
