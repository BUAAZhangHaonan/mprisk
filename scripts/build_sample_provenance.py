#!/usr/bin/env python3
"""Build per-sample provenance table from main + delivery cache roots.

For each (model_key, protocol) pair, reads:
  * main cache manifest      outputs/prefill_cache/<model>/<proto>_main_p8_seed20260717/manifest.jsonl
  * delivery cache manifest  outputs/prefill_cache/<model>/<proto>_delivery_p8_seed20260717/manifest.jsonl

A sample_id is considered "complete" in a cache if it has at least
``MIN_ENTRIES_PER_SAMPLE`` (default 24 = 8 prompts x 3 conditions) entries
in that cache's manifest. Incomplete samples are still recorded but flagged
via the ``complete_*`` columns.

Output columns:
    sample_id, model_key, protocol, in_main, in_delivery, source

``source`` is one of:
    MAIN_ONLY       -- sample appears only in the main cache
    DELIVERY_ONLY   -- sample appears only in the delivery cache
    BOTH            -- sample appears in both caches
    CH_SIMS_ORIG    -- non-generated original sample (sample_id does NOT
                       start with ``gen:``); added as a tag column
                       ``generated`` is also emitted (True/False)

The ``generated`` column is the cleaner signal:
    True  -> sample_id starts with ``gen:`` (a delivery-only generated sample
            in the current pipeline; main cache never carries ``gen:`` ids)
    False -> original CH-SIMS v2 sample

Usage:
    python scripts/build_sample_provenance.py \
        --output data/processed/manifests/sample_provenance.csv

Optional:
    --min-entries 24    complete-sample threshold
    --root .            project root (default: cwd)
    --models qwen3_vl_8b internvl3_5_8b qwen2_5_omni_7b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Default models + protocol mapping (mirrors configs/pipeline_runtime/pipeline.yaml).
DEFAULT_MODEL_PROTOCOL: list[tuple[str, str]] = [
    ("qwen3_vl_8b", "vt"),
    ("internvl3_5_8b", "vt"),
    ("qwen2_5_omni_7b", "va"),
]

# Minimum cache entries per sample to consider the sample "complete" in
# that cache. 8 prompts x 3 conditions = 24.
MIN_ENTRIES_PER_SAMPLE_DEFAULT = 24

# Prefix that marks delivery-only generated samples.
GEN_PREFIX = "gen:"


def _scan_cache_manifest(path: Path) -> tuple[set[str], dict[str, int], dict[str, int]]:
    """Scan one cache manifest.

    Returns:
        sample_ids_complete: set of sample_ids with >= MIN_ENTRIES_PER_SAMPLE entries
        sample_id_counts:   {sample_id: total entry count} (all entries)
        sample_id_prompt_counts: {sample_id: distinct prompt_id count}
    """
    counts: dict[str, int] = defaultdict(int)
    prompt_counts: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return set(), dict(counts), {k: len(v) for k, v in prompt_counts.items()}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id")
            if not sid:
                continue
            counts[sid] += 1
            pid = row.get("prompt_id")
            if pid:
                prompt_counts[sid].add(pid)
    return (
        {sid for sid, c in counts.items() if c >= MIN_ENTRIES_PER_SAMPLE_DEFAULT},
        dict(counts),
        {k: len(v) for k, v in prompt_counts.items()},
    )


def _classify_source(in_main: bool, in_delivery: bool) -> str:
    if in_main and in_delivery:
        return "BOTH"
    if in_main:
        return "MAIN_ONLY"
    if in_delivery:
        return "DELIVERY_ONLY"
    # Should not happen because we only iterate over observed ids.
    return "UNKNOWN"


def _classify_generated(sample_id: str) -> bool:
    """A sample is "generated" iff its id starts with ``gen:``."""
    return sample_id.startswith(GEN_PREFIX)


def _model_protocol_pairs(
    cli_models: list[str] | None,
) -> list[tuple[str, str]]:
    """Resolve which (model, protocol) pairs to scan.

    If user passes --models qwen3_vl_8b internvl3_5_8b, we filter the
    default mapping by model_key. If user passes nothing, we use defaults.
    """
    if not cli_models:
        return list(DEFAULT_MODEL_PROTOCOL)
    wanted = set(cli_models)
    return [(m, p) for (m, p) in DEFAULT_MODEL_PROTOCOL if m in wanted]


def build_provenance(
    *,
    project_root: Path,
    model_protocol_pairs: list[tuple[str, str]],
    min_entries: int = MIN_ENTRIES_PER_SAMPLE_DEFAULT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk caches, return (rows, summaries).

    rows: one per (sample_id, model_key, protocol)
    summaries: one per (model_key, protocol) with category counts
    """
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    cache_base = project_root / "outputs" / "prefill_cache"

    for model_key, protocol in model_protocol_pairs:
        main_manifest = (
            cache_base
            / model_key
            / f"{protocol}_main_p8_seed20260717"
            / "manifest.jsonl"
        )
        delivery_manifest = (
            cache_base
            / model_key
            / f"{protocol}_delivery_p8_seed20260717"
            / "manifest.jsonl"
        )

        main_complete, main_counts, main_prompts = _scan_cache_manifest(main_manifest)
        delivery_complete, delivery_counts, delivery_prompts = _scan_cache_manifest(
            delivery_manifest
        )

        all_ids = set(main_counts.keys()) | set(delivery_counts.keys())
        cat_counts: dict[str, int] = defaultdict(int)
        gen_total = 0
        orig_total = 0
        for sid in sorted(all_ids):
            in_main = sid in main_complete
            in_delivery = sid in delivery_complete
            # Fallback: if not "complete" but the id is present, still flag True
            # so we don't lose track of partially-extracted samples. We surface
            # completeness via separate columns.
            in_main_any = sid in main_counts
            in_delivery_any = sid in delivery_counts
            source = _classify_source(in_main_any, in_delivery_any)
            generated = _classify_generated(sid)
            if generated:
                gen_total += 1
            else:
                orig_total += 1
            # The BOTH / MAIN_ONLY / DELIVERY_ONLY label is keyed on
            # presence (any entries), which is the more useful provenance
            # signal. Completeness is reported separately.
            cat_counts[source] += 1
            # Override the per-sample source label so MAIN_ONLY means "only
            # present in main" (not "only complete in main"). This matches
            # the task spec.
            rows.append(
                {
                    "sample_id": sid,
                    "model_key": model_key,
                    "protocol": protocol,
                    "in_main": bool(in_main_any),
                    "in_delivery": bool(in_delivery_any),
                    "complete_in_main": bool(in_main),
                    "complete_in_delivery": bool(in_delivery),
                    "source": source,
                    "generated": bool(generated),
                    "main_entries": int(main_counts.get(sid, 0)),
                    "delivery_entries": int(delivery_counts.get(sid, 0)),
                    "main_prompt_count": int(main_prompts.get(sid, 0)),
                    "delivery_prompt_count": int(delivery_prompts.get(sid, 0)),
                }
            )

        # Map source to the spec's category names. The task spec lists 5
        # categories: MAIN_ONLY, DELIVERY_ONLY, BOTH, CH_SIMS_ORIG,
        # GEN_DELIVERY. CH_SIMS_ORIG and GEN_DELIVERY are *orthogonal* tags
        # (non-generated vs generated) -- we surface them in the summary
        # block but the per-row ``source`` column uses the
        # presence-based labels.
        summaries.append(
            {
                "model_key": model_key,
                "protocol": protocol,
                "main_manifest": str(main_manifest),
                "delivery_manifest": str(delivery_manifest),
                "main_manifest_exists": main_manifest.exists(),
                "delivery_manifest_exists": delivery_manifest.exists(),
                "n_unique_sample_ids": len(all_ids),
                "n_main_only": int(cat_counts.get("MAIN_ONLY", 0)),
                "n_delivery_only": int(cat_counts.get("DELIVERY_ONLY", 0)),
                "n_both": int(cat_counts.get("BOTH", 0)),
                "n_generated": int(gen_total),
                "n_original_ch_sims": int(orig_total),
                "min_entries_per_sample": int(min_entries),
            }
        )

    return rows, summaries


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still create the file with just headers so downstream tools
        # don't crash.
        headers = [
            "sample_id",
            "model_key",
            "protocol",
            "in_main",
            "in_delivery",
            "complete_in_main",
            "complete_in_delivery",
            "source",
            "generated",
            "main_entries",
            "delivery_entries",
            "main_prompt_count",
            "delivery_prompt_count",
        ]
        with output.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
        return
    headers = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-sample provenance table from main + delivery cache "
            "manifests. One row per (sample_id, model_key, protocol)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifests/sample_provenance.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root (default: current working directory).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Subset of model keys to scan. Defaults to "
            "qwen3_vl_8b internvl3_5_8b qwen2_5_omni_7b."
        ),
    )
    parser.add_argument(
        "--min-entries",
        type=int,
        default=MIN_ENTRIES_PER_SAMPLE_DEFAULT,
        help=(
            "Minimum cache entries per sample_id for the sample to be "
            "considered complete in that cache (default 24 = 8x3)."
        ),
    )
    args = parser.parse_args(argv)

    project_root = args.root.resolve()
    pairs = _model_protocol_pairs(args.models)

    print(
        f"[build_sample_provenance] project_root={project_root} "
        f"models={[m for m, _ in pairs]}",
        flush=True,
    )

    rows, summaries = build_provenance(
        project_root=project_root,
        model_protocol_pairs=pairs,
        min_entries=args.min_entries,
    )

    _write_csv(rows, args.output)
    print(f"[build_sample_provenance] wrote {len(rows)} rows -> {args.output}", flush=True)

    print("[build_sample_provenance] per (model, protocol) summary:")
    print(
        f"  {'model_key':22s} {'proto':5s} "
        f"{'main_only':>10s} {'deliv_only':>11s} {'both':>6s} "
        f"{'gen':>6s} {'orig':>6s} {'total':>6s}"
    )
    for s in summaries:
        print(
            f"  {s['model_key']:22s} {s['protocol']:5s} "
            f"{s['n_main_only']:>10d} {s['n_delivery_only']:>11d} "
            f"{s['n_both']:>6d} {s['n_generated']:>6d} "
            f"{s['n_original_ch_sims']:>6d} {s['n_unique_sample_ids']:>6d}"
        )
        if not s["main_manifest_exists"]:
            print(
                f"  [warn] {s['model_key']} {s['protocol']}: main manifest missing: "
                f"{s['main_manifest']}",
                file=sys.stderr,
            )
        if not s["delivery_manifest_exists"]:
            print(
                f"  [warn] {s['model_key']} {s['protocol']}: delivery manifest missing: "
                f"{s['delivery_manifest']}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
