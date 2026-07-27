from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mprisk.utils.jsonl_receipt import write_atomic_jsonl
from mprisk.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish a formal cache-closed sample intersection and unmatched-ID report."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-input", type=int, required=True)
    parser.add_argument("--expected-formal", type=int, required=True)
    parser.add_argument("--expected-unmatched", type=int, required=True)
    args = parser.parse_args()
    rows = _read(args.input)
    formal = _read(args.formal_manifest)
    if len(rows) != args.expected_input or len(formal) != args.expected_formal:
        raise ValueError("Input or formal row count does not match the explicit contract")
    by_id = _index(rows)
    formal_ids = [str(row["sample_id"]) for row in formal]
    if len(formal_ids) != len(set(formal_ids)):
        raise ValueError("Formal manifest contains duplicate sample IDs")
    missing = sorted(set(formal_ids) - set(by_id))
    unmatched = sorted(set(by_id) - set(formal_ids))
    if missing or len(unmatched) != args.expected_unmatched:
        raise ValueError(
            f"Intersection contract failed: missing={missing}, unmatched={unmatched}"
        )
    selected = [by_id[sample_id] for sample_id in formal_ids]
    write_atomic_jsonl(args.output, selected)
    write_json(
        args.report,
        {
            "schema_name": "mprisk_formal_sample_intersection_v1",
            "status": "PASS",
            "input_path": str(args.input.resolve()),
            "input_sha256": _sha256(args.input),
            "input_rows": len(rows),
            "formal_manifest_path": str(args.formal_manifest.resolve()),
            "formal_manifest_sha256": _sha256(args.formal_manifest),
            "formal_rows": len(formal),
            "intersection_rows": len(selected),
            "missing_formal_ids": missing,
            "unmatched_ids": unmatched,
            "unmatched_count": len(unmatched),
            "output_path": str(args.output.resolve()),
            "output_sha256": _sha256(args.output),
        },
    )
    return 0


def _read(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(row, dict) and row.get("sample_id") for row in rows):
        raise ValueError(f"Invalid sample rows: {path}")
    return rows


def _index(rows: list[dict]) -> dict[str, dict]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Input contains duplicate sample IDs")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
