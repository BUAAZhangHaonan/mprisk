from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.generated_archive_freeze import freeze_generated_round1


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Freeze the four generated round-one archives with immutable provenance."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs/data/generated_round1_v1.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze_generated_round1(repo_root=args.repo_root, config_path=args.config)
    print(
        json.dumps(
            {
                "status": "ok",
                "archive_manifest": str(result.archive_manifest_path),
                "gt_eligible": str(result.gt_eligible_path),
                "provenance": str(result.provenance_path),
                "total_count": result.total_count,
                "gt_eligible_count": result.gt_eligible_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
