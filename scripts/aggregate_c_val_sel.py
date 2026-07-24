#!/usr/bin/env python3
"""Aggregate C1-C5 metrics old (test-sel) vs new (val-sel) into mean+-std tables."""
import argparse
import json
import math
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "canonical_rerun_v2"
DEFAULT_OUTPUT = DEFAULT_ROOT / "c_val_sel_aggregate.json"
MODELS = ["qwen3_vl_8b", "qwen3_5_4b"]
SEEDS = [20260717, 20260718, 20260719]

# cfg fields:
#  dir, fname, val/test path into best_metrics for balanced_acc,
#  test_f1/test_ap paths for Macro F1 / AP
METHODS = {
    "C1_SP_MLP": dict(dir="C1_sp_mlp_v2_ca", fname="pretrain_metrics.json",
                      val=("best_metrics", "val_balanced_acc"),
                      test=("best_metrics", "test_balanced_acc"),
                      test_f1=("best_metrics", "test_macro_f1"),
                      test_ap=("best_metrics", "test_ap")),
    "C3_T_LSTM": dict(dir="C3_t_lstm_v2_ca", fname="pretrain_metrics.json",
                      val=("best_metrics", "val_balanced_acc"),
                      test=("best_metrics", "test_balanced_acc"),
                      test_f1=("best_metrics", "test_macro_f1"),
                      test_ap=("best_metrics", "test_ap")),
    "C2_SP_MLP": dict(dir="C2_sp_mlp_v2_mn", fname="mn_metrics.json",
                      val=("best_metrics", "val_balanced_acc"),
                      test=("best_metrics", "test_balanced_acc"),
                      test_f1=("best_metrics", "test_macro_f1"),
                      test_ap=("best_metrics", "test_ap")),
    "C4_T_LSTM": dict(dir="C4_t_lstm_v2_mn", fname="mn_metrics.json",
                      val=("best_metrics", "val_balanced_acc"),
                      test=("best_metrics", "test_balanced_acc"),
                      test_f1=("best_metrics", "test_macro_f1"),
                      test_ap=("best_metrics", "test_ap")),
    "C5_TME_v3B": dict(dir="C5_tme_v3b_e2e_mn", fname="metrics.json",
                       val=("best_metrics", "val_mn_balanced_acc"),
                       test=("best_metrics", "test_mn_balanced_acc"),
                       test_f1=("best_metrics", "test_mn_f1"),
                       test_ap=("best_metrics", "test_mn_ap")),
}


def _dig(d, path):
    cur = d
    for k in path:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    n = len(xs)
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var), n


def fmt_pct(m, s):
    if m is None:
        return "--"
    return f"{m*100:.2f}+-{s*100:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Output root containing per-stage subdirs (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the raw aggregation JSON (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()
    root: Path = args.root
    output: Path = args.output

    results = {}
    for method, cfg in METHODS.items():
        results[method] = {}
        for model in MODELS:
            results[method][model] = {"old": {}, "new": {}}
            for seed in SEEDS:
                run = root / cfg["dir"] / f"{model}_seed{seed}"
                new_path = run / cfg["fname"]
                old_path = run / "old_test_selected" / cfg["fname"]
                for tag, path in (("new", new_path), ("old", old_path)):
                    if not path.exists():
                        continue
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            m = json.load(f)
                    except Exception as e:
                        print(f"[warn] could not read {path}: {e}")
                        continue
                    rec = {
                        "val_bal_acc": _dig(m, cfg["val"]),
                        "test_bal_acc": _dig(m, cfg["test"]),
                        "test_f1": _dig(m, cfg["test_f1"]),
                        "test_ap": _dig(m, cfg["test_ap"]),
                        "selection_metric": _dig(m, ("best_metrics", "selection_metric")),
                        "best_epoch": _dig(m, ("best_metrics", "best_epoch")),
                    }
                    results[method][model][tag][seed] = rec

    print("=" * 100)
    print("C/A methods -- best test_balanced_acc (3 seeds mean+-std)")
    print("=" * 100)
    hdr = f"{'Method':<12} {'Model':<16} {'Old test-sel':<24} {'New val-sel':<24} {'Delta pp':<10}"
    print(hdr)
    for method in ["C1_SP_MLP", "C3_T_LSTM"]:
        for model in MODELS:
            old_test = [results[method][model]["old"][s]["test_bal_acc"]
                        for s in SEEDS if s in results[method][model]["old"]]
            new_test = [results[method][model]["new"][s]["test_bal_acc"]
                        for s in SEEDS if s in results[method][model]["new"]]
            om, os_, _ = mean_std(old_test)
            nm, ns, _ = mean_std(new_test)
            delta = (nm - om) * 100 if (om is not None and nm is not None) else None
            delta_str = f"{delta:+.2f}" if delta is not None else "--"
            print(f"{method:<12} {model:<16} {fmt_pct(om, os_):<24} {fmt_pct(nm, ns):<24} {delta_str:<10}")

    print()
    print("=" * 100)
    print("M/N methods -- test metrics (3 seeds mean+-std)")
    print("=" * 100)
    for metric_key, metric_name in [("test_bal_acc", "Balanced Acc"),
                                    ("test_f1", "Macro F1"),
                                    ("test_ap", "AP")]:
        print()
        print(f"--- {metric_name} ---")
        print(hdr)
        for method in ["C2_SP_MLP", "C4_T_LSTM", "C5_TME_v3B"]:
            for model in MODELS:
                old_vals = [results[method][model]["old"][s].get(metric_key)
                            for s in SEEDS if s in results[method][model]["old"]]
                new_vals = [results[method][model]["new"][s].get(metric_key)
                            for s in SEEDS if s in results[method][model]["new"]]
                om, os_, _ = mean_std(old_vals)
                nm, ns, _ = mean_std(new_vals)
                delta = (nm - om) * 100 if (om is not None and nm is not None) else None
                delta_str = f"{delta:+.2f}" if delta is not None else "--"
                print(f"{method:<12} {model:<16} {fmt_pct(om, os_):<24} {fmt_pct(nm, ns):<24} {delta_str:<10}")

    print()
    print("=" * 100)
    print("Sanity: selection_metric in NEW metrics.json")
    print("=" * 100)
    for method, cfg in METHODS.items():
        for model in MODELS:
            for seed in SEEDS:
                rec = results[method][model]["new"].get(seed, {})
                sm = rec.get("selection_metric", "?")
                ep = rec.get("best_epoch", "?")
                print(f"  {method:<12} {model:<16} seed={seed} selection_metric={sm!s:<22} best_epoch={ep}")
        print()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Raw aggregation saved -> {output}")


if __name__ == "__main__":
    main()
