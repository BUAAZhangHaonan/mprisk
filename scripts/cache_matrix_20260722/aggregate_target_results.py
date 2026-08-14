#!/usr/bin/env python3
"""Aggregate Target (cross-domain) C/A metrics for cache_matrix_20260722.

Reads:
  outputs/cache_matrix_20260722/runs/ca_tme_{gru,lstm,bilstm}/<model>_seed<seed>/target_metrics.json
Each file exposes val_balanced_accuracy_ac and a nested val_state_separation
dict (target C/A eval over CH-SIMS v2). val_D_gap and val_D_mannwhitney_p are
read from inside val_state_separation; a file missing the nested fields yields
null for those columns.

Emits:
  _summary/target_results.csv          (one row per (encoder, model, seed))
  _summary/target_aggregate_summary.csv (mean +/- std per (encoder, model))
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "outputs/cache_matrix_20260722/runs"
SUMMARY_DIR = ROOT / "outputs/cache_matrix_20260722/_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def main() -> int:
    rows: list[dict] = []
    for encoder in ENCODERS:
        enc_dir = RUNS_DIR / encoder
        if not enc_dir.exists():
            continue
        for run_dir in sorted(enc_dir.iterdir()):
            if not run_dir.is_dir() or "_seed" not in run_dir.name:
                continue
            model, _, seed = run_dir.name.partition("_seed")
            try:
                seed_int = int(seed)
            except ValueError:
                continue
            metrics = _load_json(run_dir / "target_metrics.json")
            separation: dict = {}
            if metrics is not None and isinstance(metrics.get("val_state_separation"), dict):
                separation = metrics["val_state_separation"]
            if metrics is None:
                rows.append({
                    "encoder": encoder, "model": model, "seed": seed_int,
                    "val_balanced_accuracy_ac": None,
                    "val_D_gap": None,
                    "val_D_mannwhitney_p": None,
                    "val_state_separation": None,
                    "metrics_loaded": False,
                })
                continue
            rows.append({
                "encoder": encoder,
                "model": model,
                "seed": seed_int,
                "val_balanced_accuracy_ac": metrics.get("val_balanced_accuracy_ac"),
                "val_D_gap": separation.get("val_D_gap"),
                "val_D_mannwhitney_p": separation.get("val_D_mannwhitney_p"),
                "val_state_separation": metrics.get("val_state_separation"),
                "metrics_loaded": True,
            })

    # Per-cell CSV.
    out_csv = SUMMARY_DIR / "target_results.csv"
    fieldnames = [
        "encoder", "model", "seed",
        "val_balanced_accuracy_ac", "val_D_gap", "val_D_mannwhitney_p",
        "val_state_separation",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"wrote {out_csv} rows={len(rows)}")

    # Aggregate by (encoder, model).
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_cell.setdefault((row["encoder"], row["model"]), []).append(row)

    agg_csv = SUMMARY_DIR / "target_aggregate_summary.csv"
    agg_fields = [
        "encoder", "model", "n_seeds",
        "val_balanced_accuracy_ac_mean", "val_balanced_accuracy_ac_std",
        "val_D_gap_mean", "val_D_gap_std",
        "val_D_mannwhitney_p_mean", "val_D_mannwhitney_p_std",
    ]
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for (encoder, model), cell_rows in sorted(by_cell.items()):
            accs = [r["val_balanced_accuracy_ac"] for r in cell_rows if r["val_balanced_accuracy_ac"] is not None]
            gaps = [r["val_D_gap"] for r in cell_rows if r["val_D_gap"] is not None]
            ps = [r["val_D_mannwhitney_p"] for r in cell_rows if r["val_D_mannwhitney_p"] is not None]
            writer.writerow({
                "encoder": encoder,
                "model": model,
                "n_seeds": len(cell_rows),
                "val_balanced_accuracy_ac_mean": round(statistics.mean(accs), 4) if accs else None,
                "val_balanced_accuracy_ac_std": round(_std(accs), 4) if _std(accs) is not None else None,
                "val_D_gap_mean": round(statistics.mean(gaps), 4) if gaps else None,
                "val_D_gap_std": round(_std(gaps), 4) if _std(gaps) is not None else None,
                "val_D_mannwhitney_p_mean": round(statistics.mean(ps), 6) if ps else None,
                "val_D_mannwhitney_p_std": round(_std(ps), 6) if _std(ps) is not None else None,
            })
    print(f"wrote {agg_csv} cells={len(by_cell)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
