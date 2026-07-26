#!/usr/bin/env python3
"""Aggregate cache_matrix_20260722 metrics across encoders / models / seeds.

Walks outputs/cache_matrix_20260722/runs/{tme_bilstm,tme_lstm,tme_gru,sp_mlp,t_lstm}/
and emits:
  - _summary/main_results.csv  (one row per (encoder, model, seed))
  - _summary/aggregate_summary.csv (mean +/- std per (encoder, model))

TME encoders write metrics.json with best_metrics.{val,test}_mn_{acc,auc}.
SP-MLP / T-LSTM write mn_metrics.json with best_test_mn_acc + val_mn_ap (no AUC).
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "outputs/cache_matrix_20260722/runs"
SUMMARY_DIR = ROOT / "outputs/cache_matrix_20260722/_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ("tme_bilstm", "tme_lstm", "tme_gru", "sp_mlp", "t_lstm")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_tme_row(encoder: str, model: str, seed: str, run_dir: Path) -> dict | None:
    metrics = _load_json(run_dir / "metrics.json")
    if metrics is None:
        return None
    best = metrics.get("best_metrics") or {}
    elapsed = metrics.get("elapsed_seconds") or metrics.get("history", {}).get("elapsed_seconds")
    return {
        "encoder": encoder,
        "model": model,
        "seed": int(seed),
        "protocol": "va" if model in {"qwen2_5_omni_7b", "gemma4_12b_it", "gemma4_12b", "phi4_multimodal"} else "vt",
        "val_mn_acc": best.get("val_mn_acc"),
        "test_mn_acc": best.get("test_mn_acc"),
        "test_mn_auc": best.get("test_mn_auc"),
        "test_mn_ap": best.get("test_mn_ap"),
        "test_mn_f1": best.get("test_mn_f1"),
        "best_epoch": best.get("best_epoch") or metrics.get("best_epoch"),
        "train_time_min": round(elapsed / 60.0, 2) if elapsed else None,
    }


def _extract_two_stage_row(encoder: str, model: str, seed: str, run_dir: Path) -> dict | None:
    metrics = _load_json(run_dir / "mn_metrics.json")
    if metrics is None:
        return None
    return {
        "encoder": encoder,
        "model": model,
        "seed": int(seed),
        "protocol": "va" if model in {"qwen2_5_omni_7b", "gemma4_12b_it", "gemma4_12b", "phi4_multimodal"} else "vt",
        "val_mn_acc": metrics.get("val_mn_acc") or metrics.get("val_balanced_accuracy_mn"),
        "test_mn_acc": metrics.get("best_test_mn_acc") or metrics.get("test_mn_acc"),
        "test_mn_auc": metrics.get("test_mn_auc"),
        "test_mn_ap": metrics.get("test_mn_ap"),
        "test_mn_f1": metrics.get("test_mn_f1") or metrics.get("best_test_mn_f1"),
        "best_epoch": metrics.get("best_epoch"),
        "train_time_min": round(metrics.get("elapsed_seconds", 0) / 60.0, 2) if metrics.get("elapsed_seconds") else None,
    }


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
            if not run_dir.is_dir():
                continue
            name = run_dir.name  # <model>_seed<seed>
            if "_seed" not in name:
                continue
            model, _, seed = name.partition("_seed")
            if not seed:
                continue
            if encoder.startswith("tme_"):
                row = _extract_tme_row(encoder, model, seed, run_dir)
            else:
                row = _extract_two_stage_row(encoder, model, seed, run_dir)
            if row is not None:
                rows.append(row)

    # Per-cell CSV.
    out_csv = SUMMARY_DIR / "main_results.csv"
    fieldnames = [
        "encoder", "model", "seed", "protocol",
        "val_mn_acc", "test_mn_acc", "test_mn_auc", "test_mn_ap", "test_mn_f1",
        "best_epoch", "train_time_min",
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

    agg_csv = SUMMARY_DIR / "aggregate_summary.csv"
    agg_fields = [
        "encoder", "model", "n_seeds",
        "test_mn_acc_mean", "test_mn_acc_std",
        "val_mn_acc_mean", "val_mn_acc_std",
        "test_mn_auc_mean", "test_mn_auc_std",
        "train_time_min_mean",
    ]
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for (encoder, model), cell_rows in sorted(by_cell.items()):
            test_accs = [r["test_mn_acc"] for r in cell_rows if r["test_mn_acc"] is not None]
            val_accs = [r["val_mn_acc"] for r in cell_rows if r["val_mn_acc"] is not None]
            test_aucs = [r["test_mn_auc"] for r in cell_rows if r["test_mn_auc"] is not None]
            times = [r["train_time_min"] for r in cell_rows if r["train_time_min"] is not None]
            writer.writerow({
                "encoder": encoder,
                "model": model,
                "n_seeds": len(cell_rows),
                "test_mn_acc_mean": round(statistics.mean(test_accs), 4) if test_accs else None,
                "test_mn_acc_std": round(_std(test_accs), 4) if _std(test_accs) is not None else None,
                "val_mn_acc_mean": round(statistics.mean(val_accs), 4) if val_accs else None,
                "val_mn_acc_std": round(_std(val_accs), 4) if _std(val_accs) is not None else None,
                "test_mn_auc_mean": round(statistics.mean(test_aucs), 4) if test_aucs else None,
                "test_mn_auc_std": round(_std(test_aucs), 4) if _std(test_aucs) is not None else None,
                "train_time_min_mean": round(statistics.mean(times), 2) if times else None,
            })
    print(f"wrote {agg_csv} cells={len(by_cell)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
