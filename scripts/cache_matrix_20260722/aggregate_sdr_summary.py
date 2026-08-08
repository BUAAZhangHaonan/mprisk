#!/usr/bin/env python3
"""Aggregate SDR state-pattern distributions for cache_matrix_20260722.

For each model, reads a single state_patterns.jsonl (one row per sample) from
either the Source sdr/ tree or the Target sdr_target/ tree, then computes:
  - Overall pct of Consensus / Confusion / Balanced / Dominant
  - pct_dominant within Conflict samples
  - pct_consensus within Aligned samples

Target SDR uses the CH-SIMS v2 natural-domain split (Conflict vs Aligned come
from the model's own misread judgments on that split).

Emits:
  _summary/source_sdr_summary.csv
  _summary/target_sdr_summary.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cache_matrix_20260722"
SOURCE_SDR = BASE / "sdr"
TARGET_SDR = BASE / "sdr_target"
SUMMARY_DIR = BASE / "_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "llava_v1_5_7b", "minicpm_v_2_6", "minicpm_v_4_5",
    "internvl3_5_8b", "phi3_5_vision", "qwen2_5_omni_7b", "qwen2_5_vl_7b",
    "qwen3_5_4b", "qwen3_5_9b", "qwen3_vl_8b",
]

PATTERNS = ("Consensus", "Confusion", "Balanced", "Dominant")


def _find_state_file(model_dir: Path) -> Path | None:
    candidates = list(model_dir.rglob("state_patterns.jsonl"))
    return candidates[0] if candidates else None


def _aggregate_tree(tree_root: Path, out_csv: Path) -> int:
    rows_out: list[dict] = []
    for model in MODELS:
        model_dir = tree_root / model
        state_file = _find_state_file(model_dir) if model_dir.exists() else None
        if state_file is None:
            rows_out.append({
                "model": model, "n_samples": 0,
                "pct_consensus": None, "pct_confusion": None,
                "pct_balanced": None, "pct_dominant": None,
                "conflict_pct_dominant": None, "aligned_pct_consensus": None,
            })
            continue
        rows: list[dict] = []
        for line in state_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        n = len(rows)
        if n == 0:
            rows_out.append({
                "model": model, "n_samples": 0,
                "pct_consensus": None, "pct_confusion": None,
                "pct_balanced": None, "pct_dominant": None,
                "conflict_pct_dominant": None, "aligned_pct_consensus": None,
            })
            continue
        pat_count = {p: 0 for p in PATTERNS}
        conflict_dom = 0
        conflict_n = 0
        aligned_cons = 0
        aligned_n = 0
        for r in rows:
            pat = r.get("state_pattern") or r.get("mode") or r.get("pattern")
            stype = r.get("sample_type")
            if pat in pat_count:
                pat_count[pat] += 1
            if stype == "Conflict":
                conflict_n += 1
                if pat == "Dominant":
                    conflict_dom += 1
            elif stype == "Aligned":
                aligned_n += 1
                if pat == "Consensus":
                    aligned_cons += 1
        rows_out.append({
            "model": model,
            "n_samples": n,
            "pct_consensus": round(100.0 * pat_count["Consensus"] / n, 2),
            "pct_confusion": round(100.0 * pat_count["Confusion"] / n, 2),
            "pct_balanced": round(100.0 * pat_count["Balanced"] / n, 2),
            "pct_dominant": round(100.0 * pat_count["Dominant"] / n, 2),
            "conflict_pct_dominant": round(100.0 * conflict_dom / conflict_n, 2) if conflict_n else None,
            "aligned_pct_consensus": round(100.0 * aligned_cons / aligned_n, 2) if aligned_n else None,
        })

    fieldnames = [
        "model", "n_samples",
        "pct_consensus", "pct_confusion", "pct_balanced", "pct_dominant",
        "conflict_pct_dominant", "aligned_pct_consensus",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    print(f"wrote {out_csv} models={len(rows_out)}")
    return 0


def main() -> int:
    rc1 = _aggregate_tree(SOURCE_SDR, SUMMARY_DIR / "source_sdr_summary.csv")
    rc2 = _aggregate_tree(TARGET_SDR, SUMMARY_DIR / "target_sdr_summary.csv")
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
