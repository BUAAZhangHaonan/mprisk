from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.utils.jsonl_receipt import write_atomic_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind a frozen multi-split manifest to one explicit recovery split."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != args.expected_rows:
        raise ValueError(
            f"Recovery manifest count mismatch: expected {args.expected_rows}, got {len(rows)}"
        )
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Recovery manifest contains duplicate sample IDs")
    rebound = [
        dict(row, source_dataset=args.dataset, split=args.split) for row in rows
    ]
    write_atomic_jsonl(args.output, rebound)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
