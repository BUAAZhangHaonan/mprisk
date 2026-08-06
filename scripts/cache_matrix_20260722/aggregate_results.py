#!/usr/bin/env python3
"""Aggregate cache_matrix_20260722 metrics across encoders / models / seeds.

Post 2026-07 refactor layout (under outputs/cache_matrix_20260722/runs/):
  - C/A TME  : ca_tme_{gru,lstm,bilstm}/<model>_seed<seed>/train_metrics.json
               (best_val_balanced_accuracy_ac; no test split)
  - M/N TME  : mn_tme_{e2e,frozen}/<model>_seed<seed>/metrics.json
               (best_metrics.{val,test}_mn_{acc,auc,ap,f1})
  - SP-MLP   : sp_mlp/<model>_seed<seed>/mn_metrics.json
  - T-LSTM   : t_lstm/<model>_seed<seed>/mn_metrics.json
               (best_test_mn_acc + val_mn_ap; no AUC)

Emits:
  - _summary/main_results.csv  (one row per (encoder, model, seed))
  - _summary/aggregate_summary.csv (mean +/- std per (encoder, model))
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

# (encoder_label, dir_name, metrics_file, kind)
# kind: 'ca' -> C/A TME train_metrics.json
#       'mn_tme' -> M/N TME metrics.json with best_metrics
#       'two_stage' -> SP-MLP / T-LSTM mn_metrics.json
ENCODERS = [
    ("ca_tme_gru",    "ca_tme_gru",    "train_metrics.json", "ca"),
    ("ca_tme_lstm",   "ca_tme_lstm",   "train_metrics.json", "ca"),
    ("ca_tme_bilstm", "ca_tme_bilstm", "train_metrics.json", "ca"),
    ("mn_tme_e2e",    "mn_tme_e2e",    "metrics.json",       "mn_tme"),
    ("mn_tme_frozen", "mn_tme_frozen", "metrics.json",       "mn_tme"),
    ("sp_mlp",        "sp_mlp",        "mn_metrics.json",    "two_stage"),
    ("t_lstm",        "t_lstm",        "mn_metrics.json",    "two_stage"),
]

VA_MODELS = {"qwen2_5_omni_7b", "gemma4_12b_it", "gemma4_12b", "phi4_multimodal"}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_ca_row(encoder, model, seed, run_dir, metrics):
    """C/A TME row: val_balanced_accuracy_ac is the only primary metric."""
    return {
        "encoder": encoder,
        "model": model,
        "seed": int(seed),
        "protocol": "va" if model in VA_MODELS else "vt",
        "task": "ca",
        "val_balanced_accuracy_ac": metrics.get("best_val_balanced_accuracy_ac"),
        "val_mn_acc": None,
        "test_mn_acc": None,
        "test_mn_auc": None,
        "test_mn_ap": None,
        "test_mn_f1": None,
        "best_epoch": metrics.get("best_epoch"),
        "train_time_min": None,
    }


def _extract_mn_tme_row(encoder, model, seed, run_dir, metrics):
    best = metrics.get("best_metrics") or {}
    return {
        "encoder": encoder,
        "model": model,
        "seed": int(seed),
        "protocol": "va" if model in VA_MODELS else "vt",
        "task": "mn",
        "val_balanced_accuracy_ac": None,
        "val_mn_acc": best.get("val_mn_acc"),
        "test_mn_acc": best.get("test_mn_acc"),
        "test_mn_auc": best.get("test_mn_auc"),
        "test_mn_ap": best.get("test_mn_ap"),
        "test_mn_f1": best.get("test_mn_f1"),
        "best_epoch": best.get("best_epoch") or metrics.get("best_epoch"),
        "train_time_min": None,
    }


def _extract_two_stage_row(encoder, model, seed, run_dir, metrics):
    return {
        "encoder": encoder,
        "model": model,
        "seed": int(seed),
        "protocol": "va" if model in VA_MODELS else "vt",
        "task": "mn",
        "val_balanced_accuracy_ac": None,
        "val_mn_acc": metrics.get("val_mn_acc") or metrics.get("val_balanced_accuracy_mn"),
        "test_mn_acc": metrics.get("best_test_mn_acc") or metrics.get("test_mn_acc"),
        "test_mn_auc": metrics.get("test_mn_auc"),
        "test_mn_ap": metrics.get("test_mn_ap"),
        "test_mn_f1": metrics.get("test_mn_f1") or metrics.get("best_test_mn_f1"),
        "best_epoch": metrics.get("best_epoch"),
        "train_time_min": round(metrics.get("elapsed_seconds", 0) / 60.0, 2) if metrics.get("elapsed_seconds") else None,
    }


def _std(values):
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def main() -> int:
    rows = []

    for encoder, dir_name, mfile, kind in ENCODERS:
        enc_dir = RUNS_DIR / dir_name
        if not enc_dir.exists():
            continue
        for run_dir in sorted(enc_dir.iterdir()):
            if not run_dir.is_dir() or "_seed" not in run_dir.name:
                continue
            model, _, seed = run_dir.name.partition("_seed")
            # Skip suffixed dirs like '.failed_noclip' / '.weak_sdr' — only
            # pure-numeric seeds are canonical runs.
            try:
                seed_int = int(seed)
            except ValueError:
                continue
            metrics = _load_json(run_dir / mfile)
            if metrics is None:
                continue
            if kind == "ca":
                row = _extract_ca_row(encoder, model, seed, run_dir, metrics)
            elif kind == "mn_tme":
                row = _extract_mn_tme_row(encoder, model, seed, run_dir, metrics)
            else:
                row = _extract_two_stage_row(encoder, model, seed, run_dir, metrics)
            rows.append(row)

    # Per-cell CSV.
    out_csv = SUMMARY_DIR / "main_results.csv"
    fieldnames = [
        "encoder", "model", "seed", "protocol", "task",
        "val_balanced_accuracy_ac",
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
    by_cell = {}
    for row in rows:
        by_cell.setdefault((row["encoder"], row["model"]), []).append(row)

    agg_csv = SUMMARY_DIR / "aggregate_summary.csv"
    agg_fields = [
        "encoder", "model", "task", "n_seeds",
        "primary_mean", "primary_std",
        "val_mn_acc_mean", "val_mn_acc_std",
        "test_mn_acc_mean", "test_mn_acc_std",
        "test_mn_auc_mean", "test_mn_auc_std",
    ]
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for (encoder, model), cell_rows in sorted(by_cell.items()):
            task = cell_rows[0]["task"]
            if task == "ca":
                primary = [r["val_balanced_accuracy_ac"] for r in cell_rows if r["val_balanced_accuracy_ac"] is not None]
            else:
                primary = [r["test_mn_acc"] for r in cell_rows if r["test_mn_acc"] is not None]
            val_accs = [r["val_mn_acc"] for r in cell_rows if r["val_mn_acc"] is not None]
            test_accs = [r["test_mn_acc"] for r in cell_rows if r["test_mn_acc"] is not None]
            test_aucs = [r["test_mn_auc"] for r in cell_rows if r["test_mn_auc"] is not None]
            writer.writerow({
                "encoder": encoder,
                "model": model,
                "task": task,
                "n_seeds": len(cell_rows),
                "primary_mean": round(statistics.mean(primary), 4) if primary else None,
                "primary_std": round(_std(primary), 4) if _std(primary) is not None else None,
                "val_mn_acc_mean": round(statistics.mean(val_accs), 4) if val_accs else None,
                "val_mn_acc_std": round(_std(val_accs), 4) if _std(val_accs) is not None else None,
                "test_mn_acc_mean": round(statistics.mean(test_accs), 4) if test_accs else None,
                "test_mn_acc_std": round(_std(test_accs), 4) if _std(test_accs) is not None else None,
                "test_mn_auc_mean": round(statistics.mean(test_aucs), 4) if test_aucs else None,
                "test_mn_auc_std": round(_std(test_aucs), 4) if _std(test_aucs) is not None else None,
            })
    print(f"wrote {agg_csv} cells={len(by_cell)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
