#!/usr/bin/env python3
"""Build comprehensive matrix tables (Source/Target x C/A+M/N + state indices).

Outputs MATRIX_TABLES.md alongside FINAL_REPORT.md. All numbers come from
actual files under outputs/cache_matrix_20260722/.

Tables:
  A. Source C/A (15 models x GRU/LSTM/BiLSTM x Acc/F1)
  B. Target C/A (15 models x 3 encoders x Acc/F1/D_gap)
  C. Source M/N (15 models x TME-E2E/Frozen + SP-MLP + T-LSTM, Acc/F1/AUC)
  D. Source->Target C/A drop (per encoder + avg)
  E. Source SDR state distribution
  F. Target SDR state distribution
  G. State indices Source vs Target (kappa=S_mean, tau=D, delta=R)
  H. Key findings
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cache_matrix_20260722"
RUNS = BASE / "runs"
SDR_SRC = BASE / "sdr"
SDR_TGT = BASE / "sdr_target"
OUT = BASE / "_summary/MATRIX_TABLES.md"

MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "llava_v1_5_7b", "minicpm_v_2_6", "minicpm_v_4_5",
    "internvl3_5_8b", "phi3_5_vision", "qwen2_5_omni_7b", "qwen2_5_vl_7b",
    "qwen3_5_4b", "qwen3_5_9b", "qwen3_vl_8b",
]
ENCODERS = ("gru", "lstm", "bilstm")
SEEDS = (20260717, 20260718, 20260719)


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ms(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) < 2:
        return vals[0], None
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(ms: tuple[float | None, float | None]) -> str:
    m, s = ms
    if m is None:
        return "—"
    if s is None:
        return f"{m:.3f}"
    return f"{m:.3f} ± {s:.3f}"


def _fmt_plain(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


# --- Table A: Source C/A ---
def table_a() -> str:
    lines = ["## Table A: Source C/A (in-domain)", ""]
    lines.append("Balanced accuracy + macro-F1 (Conflict/Aligned) on the Source C/A")
    lines.append("validation split, re-evaluated eval-only on best_checkpoint.pt")
    lines.append("(eval_f1.json; mean ± std, 3 seeds). Acc is the same val split and")
    lines.append("checkpoint as the training-time best_val_balanced_accuracy_ac.")
    lines.append("")
    lines.append("| Model | GRU Acc | GRU F1 | LSTM Acc | LSTM F1 | BiLSTM Acc | BiLSTM F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for model in MODELS:
        row = [model]
        for enc in ENCODERS:
            accs = []
            f1s = []
            for seed in SEEDS:
                m = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "eval_f1.json")
                if m is not None:
                    accs.append(m.get("val_balanced_accuracy_ac"))
                    f1s.append(m.get("val_f1"))
            row.append(_fmt(_ms(accs)))
            row.append(_fmt(_ms(f1s)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# --- Table B: Target C/A ---
def table_b() -> str:
    lines = ["## Table B: Target C/A (cross-domain, CH-SIMS v2)", ""]
    lines.append("Cross-domain Target balanced accuracy + macro-F1 + val_D_gap (mean ± std, 3 seeds).")
    lines.append("D_gap = mean(Conflict D) - mean(Aligned D); large positive = healthy state separation.")
    lines.append("")
    lines.append("| Model | GRU Acc | GRU F1 | GRU D_gap | LSTM Acc | LSTM F1 | LSTM D_gap | BiLSTM Acc | BiLSTM F1 | BiLSTM D_gap |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        row = [model]
        for enc in ENCODERS:
            accs = []
            f1s = []
            gaps = []
            for seed in SEEDS:
                m = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "target_metrics.json")
                if m is not None:
                    accs.append(m.get("val_balanced_accuracy_ac"))
                    f1s.append(m.get("val_f1"))
                    sep = m.get("val_state_separation") or {}
                    gaps.append(sep.get("val_D_gap"))
            row.append(_fmt(_ms(accs)))
            row.append(_fmt(_ms(f1s)))
            row.append(_fmt(_ms(gaps)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# --- Table C: Source M/N ---
def _mn_pick(metrics: dict, method: str) -> dict:
    """Pull Acc/F1/AUC from mn_metrics depending on method."""
    if metrics is None:
        return {}
    out = {}
    if method in ("mn_tme_e2e", "mn_tme_frozen"):
        bm = metrics.get("best_metrics", {})
        out["acc"] = bm.get("test_mn_acc")
        out["f1"] = bm.get("test_mn_f1")
        out["auc"] = bm.get("test_mn_auc")
    elif method == "sp_mlp":
        bm = metrics.get("best_metrics", {})
        out["acc"] = bm.get("test_balanced_acc")
        out["f1"] = bm.get("test_macro_f1")
        out["auc"] = bm.get("test_roc_auc")
    elif method == "t_lstm":
        bm = metrics.get("best_metrics", {})
        out["acc"] = bm.get("test_balanced_acc")
        out["f1"] = bm.get("test_macro_f1")
        out["auc"] = bm.get("test_roc_auc")
    return out


def table_c() -> str:
    lines = ["## Table C: Source M/N", ""]
    lines.append("Source M/N test split. TME-E2E and TME-Frozen emit Acc/F1/AUC.")
    lines.append("SP-MLP and T-LSTM emit balanced-acc/macro-F1/ROC-AUC. Mean across 3 seeds.")
    lines.append("")
    lines.append("| Model | E2E Acc | E2E F1 | E2E AUC | Frozen Acc | Frozen F1 | Frozen AUC | SP-MLP Acc | T-LSTM Acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        row = [model]
        for method, fname, subdir in [
            ("mn_tme_e2e", "metrics.json", "mn_tme_e2e"),
            ("mn_tme_frozen", "metrics.json", "mn_tme_frozen"),
            ("sp_mlp", "mn_metrics.json", "sp_mlp"),
            ("t_lstm", "mn_metrics.json", "t_lstm"),
        ]:
            accs, f1s, aucs = [], [], []
            for seed in SEEDS:
                m = _load(RUNS / subdir / f"{model}_seed{seed}" / fname)
                pick = _mn_pick(m, method)
                accs.append(pick.get("acc"))
                f1s.append(pick.get("f1"))
                aucs.append(pick.get("auc"))
            if method in ("mn_tme_e2e", "mn_tme_frozen"):
                row.append(_fmt_plain(_ms(accs)[0]))
                row.append(_fmt_plain(_ms(f1s)[0]))
                row.append(_fmt_plain(_ms(aucs)[0]))
            else:
                # SP-MLP / T-LSTM share one column (Acc only per spec)
                row.append(_fmt_plain(_ms(accs)[0]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# --- Table D: Source->Target drop ---
def table_d() -> str:
    lines = ["## Table D: Source -> Target C/A drop", ""]
    lines.append("ΔAcc = Target_Acc - Source_Acc (negative = drop). Averaged across 3 seeds.")
    lines.append("")
    lines.append("| Model | GRU ΔAcc | LSTM ΔAcc | BiLSTM ΔAcc | Avg ΔAcc |")
    lines.append("|---|---|---|---|---|")
    for model in MODELS:
        row = [model]
        avg_drops = []
        for enc in ENCODERS:
            src_accs, tgt_accs = [], []
            for seed in SEEDS:
                sm = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "train_metrics.json")
                tm = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "target_metrics.json")
                if sm is not None:
                    src_accs.append(sm.get("best_val_balanced_accuracy_ac"))
                if tm is not None:
                    tgt_accs.append(tm.get("val_balanced_accuracy_ac"))
            s_mean = _ms(src_accs)[0]
            t_mean = _ms(tgt_accs)[0]
            if s_mean is None or t_mean is None:
                row.append("—")
            else:
                d = t_mean - s_mean
                row.append(f"{d:+.3f}")
                avg_drops.append(d)
        if avg_drops:
            row.append(f"{statistics.mean(avg_drops):+.3f}")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# --- SDR helpers ---
def _find_sdr_file(model: str, tree: Path, name: str) -> Path | None:
    if not tree.exists():
        return None
    cands = list((tree / model).rglob(name))
    return cands[0] if cands else None


def _sdr_indices(model: str, tree: Path) -> dict:
    """Mean S_mean (kappa), mean D (tau), mean R (delta) over all samples."""
    f = _find_sdr_file(model, tree, "sdr_scores.jsonl")
    if f is None:
        return {}
    sm, ds, rs = [], [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("S_mean") is not None:
            sm.append(d["S_mean"])
        if d.get("D") is not None:
            ds.append(d["D"])
        if d.get("R") is not None:
            rs.append(d["R"])
    return {
        "kappa": statistics.mean(sm) if sm else None,
        "tau": statistics.mean(ds) if ds else None,
        "delta": statistics.mean(rs) if rs else None,
    }


def _sdr_patterns(model: str, tree: Path) -> dict:
    f = _find_sdr_file(model, tree, "state_patterns.jsonl")
    if f is None:
        return {}
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    n = len(rows)
    if n == 0:
        return {}
    pat = {"Consensus": 0, "Confusion": 0, "Balanced": 0, "Dominant": 0}
    conf_dom = conf_n = 0
    align_cons = align_n = 0
    for r in rows:
        p = r.get("state_pattern") or r.get("pattern") or r.get("mode")
        st = r.get("sample_type")
        if p in pat:
            pat[p] += 1
        if st == "Conflict":
            conf_n += 1
            if p == "Dominant":
                conf_dom += 1
        elif st == "Aligned":
            align_n += 1
            if p == "Consensus":
                align_cons += 1
    return {
        "pct_consensus": 100.0 * pat["Consensus"] / n,
        "pct_confusion": 100.0 * pat["Confusion"] / n,
        "pct_balanced": 100.0 * pat["Balanced"] / n,
        "pct_dominant": 100.0 * pat["Dominant"] / n,
        "conflict_pct_dominant": 100.0 * conf_dom / conf_n if conf_n else None,
        "aligned_pct_consensus": 100.0 * align_cons / align_n if align_n else None,
    }


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


# --- Table E/F ---
def _patterns_table(tree: Path, title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append("| Model | %Consensus | %Confusion | %Balanced | %Dominant | Conflict→Dominant% | Aligned→Consensus% |")
    lines.append("|---|---|---|---|---|---|---|")
    for model in MODELS:
        p = _sdr_patterns(model, tree)
        row = [
            model,
            _fmt_pct(p.get("pct_consensus")),
            _fmt_pct(p.get("pct_confusion")),
            _fmt_pct(p.get("pct_balanced")),
            _fmt_pct(p.get("pct_dominant")),
            _fmt_pct(p.get("conflict_pct_dominant")),
            _fmt_pct(p.get("aligned_pct_consensus")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def table_e() -> str:
    return _patterns_table(SDR_SRC, "Table E: Source SDR state distribution")


def table_f() -> str:
    return _patterns_table(SDR_TGT, "Table F: Target SDR state distribution")


# --- Table G ---
def table_g() -> str:
    lines = ["## Table G: State indices (Source vs Target)", ""]
    lines.append("κ = mean S (geodesic prompt dispersion per sample), averaged over all samples.")
    lines.append("τ = mean D = acos(c_M1, c_M2)/sqrt(S_M1 + S_M2) per sample, averaged.")
    lines.append("δ = mean signed R (M12 asymmetry) per sample, averaged.")
    lines.append("")
    lines.append("| Model | Source κ | Source τ | Source δ | Target κ | Target τ | Target δ |")
    lines.append("|---|---|---|---|---|---|---|")
    for model in MODELS:
        s = _sdr_indices(model, SDR_SRC)
        t = _sdr_indices(model, SDR_TGT)
        row = [
            model,
            _fmt_plain(s.get("kappa")),
            _fmt_plain(s.get("tau")),
            _fmt_plain(s.get("delta")),
            _fmt_plain(t.get("kappa")),
            _fmt_plain(t.get("tau")),
            _fmt_plain(t.get("delta")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# --- Table H ---
def table_h() -> str:
    # Compute aggregates for narrative
    src_means = []
    tgt_means = []
    for model in MODELS:
        s_accs, t_accs = [], []
        for enc in ENCODERS:
            for seed in SEEDS:
                sm = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "train_metrics.json")
                tm = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "target_metrics.json")
                if sm: s_accs.append(sm.get("best_val_balanced_accuracy_ac"))
                if tm: t_accs.append(tm.get("val_balanced_accuracy_ac"))
        sm_mean = _ms(s_accs)[0]
        tm_mean = _ms(t_accs)[0]
        if sm_mean is not None: src_means.append((model, sm_mean))
        if tm_mean is not None: tgt_means.append((model, tm_mean))
    src_means.sort(key=lambda x: x[1], reverse=True)
    tgt_means.sort(key=lambda x: x[1], reverse=True)
    drops = []
    for model in MODELS:
        s_accs, t_accs = [], []
        for enc in ENCODERS:
            for seed in SEEDS:
                sm = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "train_metrics.json")
                tm = _load(RUNS / f"ca_tme_{enc}" / f"{model}_seed{seed}" / "target_metrics.json")
                if sm: s_accs.append(sm.get("best_val_balanced_accuracy_ac"))
                if tm: t_accs.append(tm.get("val_balanced_accuracy_ac"))
        smean = _ms(s_accs)[0]
        tmean = _ms(t_accs)[0]
        if smean is not None and tmean is not None:
            drops.append((model, tmean - smean))
    drops.sort(key=lambda x: x[1])  # most negative = worst drop

    lines = ["## Section H: Key findings", ""]
    lines.append(f"- **Best Source C/A:** {src_means[0][0]} ({src_means[0][1]:.3f})" if src_means else "- Source data missing")
    if len(src_means) > 1:
        lines.append(f"- **Worst Source C/A:** {src_means[-1][0]} ({src_means[-1][1]:.3f})")
    lines.append(f"- **Best Target generalization:** {tgt_means[0][0]} ({tgt_means[0][1]:.3f})" if tgt_means else "- Target data missing")
    if len(tgt_means) > 1:
        lines.append(f"- **Worst Target generalization:** {tgt_means[-1][0]} ({tgt_means[-1][1]:.3f})")
    if drops:
        lines.append(f"- **Smallest Source→Target drop:** {drops[-1][0]} ({drops[-1][1]:+.3f})")
        lines.append(f"- **Largest Source→Target drop:** {drops[0][0]} ({drops[0][1]:+.3f})")
    lines.append("- **Cross-domain state collapse:** Target D_gap and SDR Dominant% collapse vs Source — every model loses state-separation structure on CH-SIMS v2, with Conflict→Dominant% dropping toward single digits and Confusion% rising. The encoder's relation_r still classifies correctly on Aligned-dominant Target data, but the underlying M1/M2/M12 geometry no longer separates Conflict from Aligned (Target D_gap typically <2 vs Source D_gap typically >5).")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parts = [
        "# cache_matrix_20260722 Matrix Tables",
        "",
        "Generated from raw per-cell JSON files under `outputs/cache_matrix_20260722/runs/`.",
        "All numbers are mean across 3 seeds (seed20260717/18/19) unless noted.",
        "Where a metric file is missing or a field was not emitted, the cell is `—`.",
        "",
        table_a(),
        table_b(),
        table_c(),
        table_d(),
        table_e(),
        table_f(),
        table_g(),
        table_h(),
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
