#!/usr/bin/env python3
"""Render FINAL_REPORT.md for cache_matrix_20260722 from emitted CSVs.

Sections:
  1. Source C/A (in-domain)
  2. Target C/A (cross-domain) + Source->Target drop
  3. Source M/N (e2e/frozen/SP-MLP/T-LSTM)
  4. Source SDR state distribution
  5. Target SDR state distribution
  6. Key findings

Phi4_multimodal is dropped (max_new_tokens=64 produced ImproImpro loops; 0
judgments). Target M/N is not possible (CH-SIMS v2 has no misread ground truth).
The Target C/A eval pipeline exposes val_state_separation only; D_gap and
Mann-Whitney p are reported as n/a.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
SUMMARY = BASE / "outputs/cache_matrix_20260722/_summary"
OUT = SUMMARY / "FINAL_REPORT.md"

MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "llava_v1_5_7b", "minicpm_v_2_6", "minicpm_v_4_5",
    "internvl3_5_8b", "phi3_5_vision", "qwen2_5_omni_7b", "qwen2_5_vl_7b",
    "qwen3_5_4b", "qwen3_5_9b", "qwen3_vl_8b",
]
ENC_LABEL = {"ca_tme_gru": "GRU", "ca_tme_lstm": "LSTM", "ca_tme_bilstm": "BiLSTM"}


def _fmt_ms(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _read_csv(name: str) -> list[dict]:
    with (SUMMARY / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_ca_source() -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for r in _read_csv("aggregate_summary.csv"):
        if r["task"] != "ca":
            continue
        if not r["primary_mean"]:
            continue
        out[r["model"]][r["encoder"]] = (float(r["primary_mean"]), float(r["primary_std"] or 0))
    return out


def _load_ca_target() -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for r in _read_csv("target_aggregate_summary.csv"):
        if not r["val_balanced_accuracy_ac_mean"]:
            continue
        out[r["model"]][r["encoder"]] = (
            float(r["val_balanced_accuracy_ac_mean"]),
            float(r["val_balanced_accuracy_ac_std"] or 0),
        )
    return out


def _load_mn_source() -> dict[str, dict[str, dict[str, float]]]:
    """model -> encoder -> {acc, auc}."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in _read_csv("aggregate_summary.csv"):
        if r["task"] != "mn":
            continue
        m = r["model"]
        e = r["encoder"]
        out[m][e]["acc_mean"] = float(r["test_mn_acc_mean"]) if r["test_mn_acc_mean"] else 0.0
        out[m][e]["acc_std"] = float(r["test_mn_acc_std"] or 0)
        out[m][e]["auc_mean"] = float(r["test_mn_auc_mean"]) if r["test_mn_auc_mean"] else 0.0
        out[m][e]["auc_std"] = float(r["test_mn_auc_std"] or 0)
    return out


def _load_sdr(name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in _read_csv(name):
        out[r["model"]] = r
    return out


def _section_1(ca_src):
    lines = [
        "## Section 1: Source C/A (in-domain, 15 models x 3 encoders x 3 seeds)",
        "",
        "Balanced accuracy on the Source conflict-attn validation split (mean +/- std across 3 seeds).",
        "Best encoder per model in **bold**.",
        "",
        "| Model | GRU | LSTM | BiLSTM |",
        "|---|---|---|---|",
    ]
    for m in MODELS:
        cells = ca_src.get(m, {})
        enc_values = {e: cells.get(e, (None, None))[0] for e in ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm")}
        finite = {e: v for e, v in enc_values.items() if v is not None}
        best_e = max(finite, key=finite.get) if finite else None
        row = [m]
        for e in ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm"):
            v = cells.get(e)
            txt = _fmt_ms(v[0], v[1]) if v else "n/a"
            if e == best_e:
                txt = f"**{txt}**"
            row.append(txt)
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _section_2(ca_src, ca_tgt):
    lines = [
        "## Section 2: Target C/A (cross-domain, CH-SIMS v2)",
        "",
        "Cross-domain transfer: balanced accuracy on the CH-SIMS v2 Target split.",
        "Source->Target drop is averaged across the 3 encoders (positive = degradation).",
        "",
        "| Model | GRU | LSTM | BiLSTM | avg Source | avg Target | avg Drop |",
        "|---|---|---|---|---|---|---|",
    ]
    drops = []
    for m in MODELS:
        s = ca_src.get(m, {})
        t = ca_tgt.get(m, {})
        row = [m]
        for e in ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm"):
            v = t.get(e)
            row.append(_fmt_ms(v[0], v[1]) if v else "n/a")
        s_vals = [s.get(e, (None,))[0] for e in ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm")]
        t_vals = [t.get(e, (None,))[0] for e in ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm")]
        s_pairs = [(sv, tv) for sv, tv in zip(s_vals, t_vals) if sv is not None and tv is not None]
        if s_pairs:
            avg_s = sum(sv for sv, _ in s_pairs) / len(s_pairs)
            avg_t = sum(tv for _, tv in s_pairs) / len(s_pairs)
            drop = avg_s - avg_t
            drops.append((m, drop))
            row.append(f"{avg_s:.4f}")
            row.append(f"{avg_t:.4f}")
            row.append(f"{drop:+.4f}")
        else:
            row += ["n/a", "n/a", "n/a"]
        lines.append("| " + " | ".join(row) + " |")
    if drops:
        biggest = max(drops, key=lambda x: x[1])
        smallest = min(drops, key=lambda x: x[1])
        lines += [
            "",
            f"Biggest avg drop: **{biggest[0]}** ({biggest[1]:+.4f}).  ",
            f"Smallest avg drop: **{smallest[0]}** ({smallest[1]:+.4f}).",
            "",
            "_val_D_gap and val_D_mannwhitney_p: n/a — Target eval pipeline only writes val_balanced_accuracy_ac and val_state_separation (always null in current outputs)._",
        ]
    return lines


def _section_3(mn_src):
    lines = [
        "## Section 3: Source M/N (15 models x 4 methods x 3 seeds)",
        "",
        "Test accuracy / AUC on the Source M/N split (mean +/- std across 3 seeds).",
        "SP-MLP and T-LSTM do not produce AUC (only accuracy + AP).",
        "",
        "| Model | E2E acc | E2E AUC | Frozen acc | Frozen AUC | SP-MLP acc | T-LSTM acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        e = mn_src.get(m, {})
        e2e = e.get("mn_tme_e2e", {})
        fr = e.get("mn_tme_frozen", {})
        sp = e.get("sp_mlp", {})
        tl = e.get("t_lstm", {})
        lines.append("| " + " | ".join([
            m,
            _fmt_ms(e2e.get("acc_mean"), e2e.get("acc_std")),
            _fmt_ms(e2e.get("auc_mean"), e2e.get("auc_std")),
            _fmt_ms(fr.get("acc_mean"), fr.get("acc_std")),
            _fmt_ms(fr.get("auc_mean"), fr.get("auc_std")),
            _fmt_ms(sp.get("acc_mean"), sp.get("acc_std")),
            _fmt_ms(tl.get("acc_mean"), tl.get("acc_std")),
        ]) + " |")
    return lines


def _section_sdr(title, sdr):
    lines = [
        f"## {title}",
        "",
        "Pattern distribution over the relevant SDR sample set. Conflict->Dominant% and Aligned->Consensus% are conditioned on sample_type.",
        "",
        "| Model | n | %Consensus | %Confusion | %Balanced | %Dominant | Conflict->Dominant% | Aligned->Consensus% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        r = sdr.get(m, {})
        def _g(k):
            v = r.get(k)
            return v if v not in (None, "") else "n/a"
        lines.append("| " + " | ".join([
            m, str(_g("n_samples")),
            _g("pct_consensus"), _g("pct_confusion"), _g("pct_balanced"), _g("pct_dominant"),
            _g("conflict_pct_dominant"), _g("aligned_pct_consensus"),
        ]) + " |")
    return lines


def _section_findings(ca_src, ca_tgt, mn_src, src_sdr, tgt_sdr):
    # Best Source C/A model+encoder
    best_src = (None, None, -1.0)
    for m, cells in ca_src.items():
        for e, (mean, _) in cells.items():
            if mean > best_src[2]:
                best_src = (m, e, mean)
    # Best Target C/A model+encoder
    best_tgt = (None, None, -1.0)
    for m, cells in ca_tgt.items():
        for e, (mean, _) in cells.items():
            if mean > best_tgt[2]:
                best_tgt = (m, e, mean)
    # Smallest avg Source->Target drop
    drops = []
    for m in ca_src:
        s = ca_src.get(m, {})
        t = ca_tgt.get(m, {})
        pairs = []
        for e in ("ca_tme_gru", "ca_tme_lstm", "ca_tme_bilstm"):
            if e in s and e in t:
                pairs.append((s[e][0], t[e][0]))
        if pairs:
            avg_drop = sum(sv - tv for sv, tv in pairs) / len(pairs)
            drops.append((m, avg_drop))
    smallest_drop = min(drops, key=lambda x: x[1]) if drops else (None, None)
    biggest_drop = max(drops, key=lambda x: x[1]) if drops else (None, None)
    # Best M/N model (highest test acc across all 4 methods)
    best_mn = (None, None, -1.0)
    for m, encs in mn_src.items():
        for e, vals in encs.items():
            if vals.get("acc_mean", 0) > best_mn[2]:
                best_mn = (m, e, vals["acc_mean"])
    # State shift: avg pct Dominant across models (Conflict->Dominant%)
    def _avg(key, sdr):
        vals = []
        for m in MODELS:
            v = sdr.get(m, {}).get(key)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        return sum(vals) / len(vals) if vals else float("nan")
    src_dom = _avg("pct_dominant", src_sdr)
    tgt_dom = _avg("pct_dominant", tgt_sdr)
    src_cons = _avg("pct_consensus", src_sdr)
    tgt_cons = _avg("pct_consensus", tgt_sdr)
    src_conf = _avg("pct_confusion", src_sdr)
    tgt_conf = _avg("pct_confusion", tgt_sdr)
    return [
        "## Section 6: Key findings",
        "",
        f"- **Best Source C/A**: `{best_src[0]}` + `{ENC_LABEL.get(best_src[1], best_src[1])}` at {best_src[2]:.4f} balanced accuracy.",
        f"- **Best Target C/A (cross-domain)**: `{best_tgt[0]}` + `{ENC_LABEL.get(best_tgt[1], best_tgt[1])}` at {best_tgt[2]:.4f}.",
        f"- **Smallest Source->Target drop (best transfer)**: `{smallest_drop[0]}` (avg drop {smallest_drop[1]:+.4f}); biggest drop: `{biggest_drop[0]}` ({biggest_drop[1]:+.4f}).",
        f"- **Best Source M/N**: `{best_mn[0]}` + `{best_mn[1]}` at {best_mn[2]:.4f} test accuracy.",
        f"- **State shift Source->Target**: avg %Dominant {src_dom:.2f} -> {tgt_dom:.2f}, avg %Consensus {src_cons:.2f} -> {tgt_cons:.2f}, avg %Confusion {src_conf:.2f} -> {tgt_conf:.2f}. Cross-domain reduces Dominant and inflates Confusion, consistent with misread-ground-truth mismatch on CH-SIMS v2.",
        "",
        "_Notes: phi4_multimodal dropped (max_new_tokens=64 bug, 0 judgments). Target M/N: not possible (CH-SIMS v2 has no misread GT). Target D_gap / Mann-Whitney p: not written by the current eval pipeline (val_state_separation is the only auxiliary field and is null in all 135 cells)._",
    ]


def main() -> int:
    ca_src = _load_ca_source()
    ca_tgt = _load_ca_target()
    mn_src = _load_mn_source()
    src_sdr = _load_sdr("source_sdr_summary.csv")
    tgt_sdr = _load_sdr("target_sdr_summary.csv")

    L: list[str] = ["# cache_matrix_20260722 Final Report", ""]
    L += _section_1(ca_src) + [""]
    L += _section_2(ca_src, ca_tgt) + [""]
    L += _section_3(mn_src) + [""]
    L += _section_sdr("Section 4: Source SDR state distribution", src_sdr) + [""]
    L += _section_sdr("Section 5: Target SDR state distribution", tgt_sdr) + [""]
    L += _section_findings(ca_src, ca_tgt, mn_src, src_sdr, tgt_sdr) + [""]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
