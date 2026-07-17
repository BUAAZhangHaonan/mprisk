from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.cache.prefill_batch import rematerialize_completed_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically rebuild prefill batch manifests from a completed SQLite ledger."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = rematerialize_completed_batch(args.output_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
