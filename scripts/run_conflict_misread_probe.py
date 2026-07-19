from __future__ import annotations

import argparse
import json

from mprisk.evaluation.misread_probe import run_conflict_misread_probe


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the identity-locked Conflict-only unified Misread probe."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_conflict_misread_probe(args.config)
    print(
        json.dumps(
            {
                "run_complete": result["run_complete_path"],
                "run_complete_sha256": result["run_complete_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
