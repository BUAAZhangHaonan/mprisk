from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.ground_truth.description_generation import verify_gt_description_generation


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify strict GT Description artifacts.")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/ground_truth/gt_description_generation_pilot.yaml",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = verify_gt_description_generation(
        args.repo_root, args.config, require_complete=args.require_complete
    )
    print(json.dumps(result.__dict__, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
