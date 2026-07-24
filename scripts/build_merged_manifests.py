#!/usr/bin/env python3
"""Build merged protocol manifests = (original protocol_manifests)
                                  U (delivery_20260716/{vt,va}_filtered.jsonl)

For each protocol (vt, va), reads the original primary manifest, then the
delivery-filtered manifest from delivery_20260716, and writes a merged
JSONL that:

  * is deduplicated by ``sample_id`` (first occurrence wins on ties, but
    on the second pass over delivery files we let the delivery version
    replace any pre-existing entry to honor "delivery version wins on
    conflict"),
  * preserves the per-row ``protocol`` field; rows whose protocol field
    disagrees with the file's protocol are dropped with a warning to
    stderr,
  * writes ``{output_dir}/{protocol}_merged_primary.jsonl``.

Input files are read-only; this script only writes under ``--output-dir``.

Usage:
    python scripts/build_merged_manifests.py \
        --orig-vt data/processed/manifests/protocol_manifests/vt_primary.jsonl \
        --orig-va data/processed/manifests/protocol_manifests/va_aux.jsonl \
        --delivery-dir data/processed/manifests/delivery_20260716 \
        --output-dir data/processed/manifests/protocol_manifests_merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_row(row: dict, protocol: str) -> dict:
    """Fill in defaults for fields missing from delivery rows so the merged
    manifest satisfies FinalManifestRow schema."""
    sid = row.get('sample_id', '')
    is_gen = bool(row.get('source_is_generated', False))
    if 'source_dataset' not in row:
        row['source_dataset'] = 'delivery_20260716' if is_gen else (sid.split(':', 1)[0] if ':' in sid else 'unknown')
    if 'split_group_id' not in row:
        row['split_group_id'] = sid
    if 'views' not in row:
        row['views'] = {
            'M1': {'modality': 'vision', 'label': '', 'specific_affect': '', 'is_clear': True},
            'M2': {'modality': 'text',   'label': '', 'specific_affect': '', 'is_clear': True},
            'M12': {'modality': 'vision+text', 'label': '', 'specific_affect': '', 'is_clear': True},
        }
    if 'use_in_main' not in row:
        row['use_in_main'] = True
    if 'split' not in row:
        row['split'] = 'train'
    if 'annotation_count' not in row:
        row['annotation_count'] = 0
    if 'annotator_agreement' not in row:
        row['annotator_agreement'] = 0.0
    if 'quality_flags' not in row:
        row['quality_flags'] = []
    if 'dominant_modality' not in row:
        row['dominant_modality'] = 'balanced'
    return row


def _merge_one(
    *,
    protocol: str,
    orig_path: Path,
    delivery_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Merge one protocol; delivery rows override originals on sample_id conflict.

    Returns a small summary dict.
    """
    # Phase 1: collect original rows into an ordered dict keyed by sample_id.
    merged: dict[str, dict[str, Any]] = {}
    orig_protocol_mismatches = 0
    if orig_path.exists():
        for row in _iter_jsonl(orig_path):
            sid = row.get("sample_id")
            if not sid:
                continue
            row_protocol = str(row.get("protocol", "")).lower()
            if row_protocol and row_protocol != protocol:
                orig_protocol_mismatches += 1
                continue
            merged[sid] = row

    # Phase 2: overlay delivery rows. Delivery wins on conflict.
    delivery_count = 0
    delivery_protocol_mismatches = 0
    if delivery_path.exists():
        for row in _iter_jsonl(delivery_path):
            sid = row.get("sample_id")
            if not sid:
                continue
            row_protocol = str(row.get("protocol", "")).lower()
            if row_protocol and row_protocol != protocol:
                delivery_protocol_mismatches += 1
                continue
            merged[sid] = row
            delivery_count += 1

    # Phase 3: emit in deterministic insertion order.
    # Normalize delivery rows to the full FinalManifestRow schema by filling
    # in defaults for fields they don't carry.
    out_rows = [_normalize_row(r, protocol) for r in merged.values()]
    _write_jsonl(output_path, out_rows)

    return {
        "protocol": protocol,
        "orig_path": str(orig_path),
        "delivery_path": str(delivery_path),
        "output_path": str(output_path),
        "n_orig_seen": len(merged) - delivery_count
        if delivery_count <= len(merged)
        else 0,
        "n_delivery_overlaid": delivery_count,
        "n_unique_written": len(out_rows),
        "orig_protocol_mismatches_dropped": orig_protocol_mismatches,
        "delivery_protocol_mismatches_dropped": delivery_protocol_mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge protocol manifests with delivery_20260716 filtered "
            "manifests. Delivery wins on sample_id conflicts; output is "
            "deduplicated."
        ),
    )
    parser.add_argument(
        "--orig-vt",
        type=Path,
        required=True,
        help="Original vt primary manifest JSONL.",
    )
    parser.add_argument(
        "--orig-va",
        type=Path,
        required=True,
        help="Original va aux manifest JSONL.",
    )
    parser.add_argument(
        "--delivery-dir",
        type=Path,
        required=True,
        help="Directory containing vt_filtered.jsonl and va_filtered.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write {vt,va}_merged_primary.jsonl.",
    )
    args = parser.parse_args(argv)

    delivery_dir: Path = args.delivery_dir
    output_dir: Path = args.output_dir

    jobs = [
        (
            "vt",
            args.orig_vt,
            delivery_dir / "vt_filtered.jsonl",
            output_dir / "vt_merged_primary.jsonl",
        ),
        (
            "va",
            args.orig_va,
            delivery_dir / "va_filtered.jsonl",
            output_dir / "va_merged_primary.jsonl",
        ),
    ]

    print(f"[build_merged_manifests] writing merged manifests to {output_dir}")
    for protocol, orig_path, delivery_path, out_path in jobs:
        summary = _merge_one(
            protocol=protocol,
            orig_path=orig_path,
            delivery_path=delivery_path,
            output_path=out_path,
        )
        print(json.dumps(summary, ensure_ascii=False))
        if summary["orig_protocol_mismatches_dropped"]:
            print(
                f"[warn] {protocol}: dropped "
                f"{summary['orig_protocol_mismatches_dropped']} orig rows with "
                f"mismatched protocol field",
                file=sys.stderr,
            )
        if summary["delivery_protocol_mismatches_dropped"]:
            print(
                f"[warn] {protocol}: dropped "
                f"{summary['delivery_protocol_mismatches_dropped']} delivery rows "
                f"with mismatched protocol field",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
