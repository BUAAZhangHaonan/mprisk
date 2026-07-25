#!/usr/bin/env python3
"""Scan all canonical_rerun_v2 results, aggregate val-keyed numbers per experiment."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def best_val_metrics(run_dir: Path) -> dict | None:
    """Return best val-selected metrics for a run, or None if missing."""
    for name in (
        "train_metrics.json",
        "pretrain_metrics.json",
        "mn_metrics.json",
        "metrics.json",
    ):
        p = run_dir / name
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def fmt_acc(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return f"{mean:.4f}±{std:.4f} (n={len(values)})"


def main() -> None:
    root = Path("outputs/canonical_rerun_v2")
    # Final delivery experiments per user's plan
    targets = [
        ("T1_gru_ca_frozen", "T1 TME-PA GRU C/A"),
        ("T5_lstm_ca_frozen", "T5 TME-PA LSTM C/A"),
        ("C1_sp_mlp_v2_ca", "C1 SP-MLP C/A pretrain"),
        ("C2_sp_mlp_v2_mn", "C2 SP-MLP M/N head"),
        ("C3_t_lstm_v2_ca", "C3 T-LSTM C/A pretrain"),
        ("C4_t_lstm_v2_mn", "C4 T-LSTM M/N head"),
        ("C5_tme_v3b_e2e_mn", "C5 TME-E2E M/N"),
        ("T1_ablation_pa_only", "T1 ablation PA-only"),
        ("T1_ablation_pa_d", "T1 ablation PA+D"),
        ("T1_ablation_pa_s", "T1 ablation PA+S"),
        ("T1_ablation_pa_sd", "T1 ablation PA+S+D"),
    ]

    models = ["qwen3_vl_8b", "qwen3_5_4b", "internvl3_5_8b", "qwen2_5_omni_7b"]

    for exp, label in targets:
        exp_dir = root / exp
        if not exp_dir.is_dir():
            print(f"\n## {label} ({exp})\n  MISSING")
            continue

        print(f"\n## {label} ({exp})")

        # group runs by model
        by_model: dict[str, list[dict]] = {m: [] for m in models}
        for run in sorted(exp_dir.iterdir()):
            if not run.is_dir():
                continue
            for m in models:
                if run.name.startswith(m + "_"):
                    mt = best_val_metrics(run)
                    if mt is not None:
                        by_model[m].append(mt)
                    break

        # determine what to print
        is_mn = "M/N" in label or "_mn" in exp or "ablation" in exp
        acc_key = "test_at_best_val_balanced_accuracy_ac"
        f1_key = "test_at_best_val_ac_f1"
        ap_key = "test_at_best_val_ac_ap"

        # Pick metric keys by experiment type
        if exp.startswith("C1_") or exp.startswith("C3_"):
            acc_key = "best_test_ac_acc"
            f1_key = "best_test_ac_f1"
            ap_key = "best_test_ac_ap"
        elif exp.startswith("C2_") or exp.startswith("C4_") or exp == "C5_tme_v3b_e2e_mn":
            acc_key = "best_test_mn_acc"
            f1_key = "best_test_mn_f1"
            ap_key = "best_test_mn_ap"
        elif exp.startswith("T1_ablation"):
            acc_key = "test_at_best_val_balanced_accuracy_ac"
            f1_key = "test_at_best_val_ac_f1"
            ap_key = "test_at_best_val_ac_ap"

        print(f"  {'model':25s} {'test_acc':30s} {'test_f1':30s}")
        for m in models:
            runs = by_model[m]
            if not runs:
                print(f"  {m:25s} --")
                continue
            accs = [r.get(acc_key, r.get("best_test_mn_acc", None)) for r in runs]
            accs = [a for a in accs if isinstance(a, (int, float))]
            f1s = [r.get(f1_key, r.get("best_test_mn_f1", None)) for r in runs]
            f1s = [f for f in f1s if isinstance(f, (int, float))]
            print(f"  {m:25s} {fmt_acc(accs):30s} {fmt_acc(f1s):30s}")

        # also report raw run list for verification
        total_runs = sum(len(v) for v in by_model.values())
        print(f"  total runs: {total_runs}/12 (4 models x 3 seeds)")


if __name__ == "__main__":
    main()
