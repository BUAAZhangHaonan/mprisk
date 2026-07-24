#!/usr/bin/env python3
"""Build protocol-filtered manifests for delivery_20260716.

Reads the four raw delivery manifests (vt/va × a/c), merges the A and C
halves for each protocol, deduplicates by sample_id (keeping first
occurrence), asserts the protocol field matches the file's protocol on
every row, and writes the merged output to
``{output_dir}/{protocol}_filtered.jsonl``.

Delivery source files are treated read-only. This script only reads from
``--delivery-root`` and only writes under ``--output-dir``.

Usage:
    python scripts/build_delivery_filtered_manifests.py \
        --delivery-root /path/to/delivery_YYYYMMDD \
        --output-dir data/processed/manifests/delivery_YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Protocol -> ordered list of (file_stem, role-label) where role-label is
# purely informational for the summary print. The "_a" file is read first,
# then "_c", matching the historical ordering used by the v2 pipeline.
PROTOCOL_FILES: dict[str, list[tuple[str, str]]] = {
    "vt": [("vt_a_manifest.jsonl", "A"), ("vt_c_manifest.jsonl", "C")],
    "va": [("va_a_manifest.jsonl", "A"), ("va_c_manifest.jsonl", "C")],
}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_protocol(
    protocol: str,
    delivery_root: Path,
    output_dir: Path,
) -> tuple[int, int, int, int]:
    """Merge A+C for one protocol, dedup by sample_id, write output.

    Returns (n_a, n_c, total_written, unique_sample_ids) for the summary.
    """
    proto_upper = protocol.upper()
    files = PROTOCOL_FILES[protocol]

    per_file_counts: list[tuple[str, int]] = []
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    for fname, _role in files:
        path = delivery_root / fname
        if not path.exists():
            raise FileNotFoundError(f"missing delivery manifest: {path}")
        rows = _iter_jsonl(path)
        per_file_counts.append((fname, len(rows)))
        for row in rows:
            row_proto = str(row.get("protocol", "")).strip().upper()
            if row_proto != proto_upper:
                raise ValueError(
                    f"protocol mismatch in {fname}: "
                    f"expected {proto_upper}, got {row_proto!r} "
                    f"(sample_id={row.get('sample_id', '?')})"
                )
            sid = str(row.get("sample_id", ""))
            if sid in seen:
                # First occurrence wins; skip subsequent duplicates.
                continue
            seen.add(sid)
            merged.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{protocol}_filtered.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_a = per_file_counts[0][1]
    n_c = per_file_counts[1][1]
    return n_a, n_c, len(merged), len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--delivery-root",
        type=Path,
        required=True,
        help="Directory containing the four raw delivery manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/manifests/delivery_20260716"),
        help="Directory to write {vt,va}_filtered.jsonl into.",
    )
    args = parser.parse_args()

    delivery_root: Path = args.delivery_root.resolve()
    output_dir: Path = args.output_dir
    # Do NOT mkdir output_dir here — _build_protocol does it, and we want
    # the resolved path printed as the CLI saw it.

    for protocol in ("vt", "va"):
        n_a, n_c, total, unique = _build_protocol(
            protocol, delivery_root, output_dir
        )
        print(
            f"{protocol}: A={n_a} C={n_c} total={total} "
            f"(unique sample_ids={unique})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
