from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.archetype_canonical_meanings import verify_archetype_canonical_meanings


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify the frozen canonical archetype dictionary."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs/labels/archetype_canonical_meanings_v1.yaml",
    )
    args = parser.parse_args()
    result = verify_archetype_canonical_meanings(
        repo_root=args.repo_root,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "dictionary_count": result.dictionary_count,
                "assignment_count": result.assignment_count,
                "review_count": result.review_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
