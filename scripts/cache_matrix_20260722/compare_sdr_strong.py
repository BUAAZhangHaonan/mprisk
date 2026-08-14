#!/usr/bin/env python3
"""Compare canonical vs sdr_strong runs for the 7 weak-separation models.

Reads per model:
  - train metrics:  runs/ca_tme_gru{,_sdr_strong}/<model>_seed20260717/train_metrics.json
  - tau:            thresholds{,_sdr_strong}/<model>/thresholds.json
  - Source patterns: sdr{,_strong}/<model>/**/state_patterns.jsonl
  - Target patterns: sdr_target{,_strong_target}/<model>/**/state_patterns.jsonl

Emits a markdown table (canonical -> strong per column) to
  outputs/cache_matrix_20260722/_summary/SDR_STRONG_COMPARE.md

Verdict rule: strong Conflict->Dominant% at least doubles OR exceeds 50%
(geometric separation improved) AND val_balanced_accuracy_ac drops no more
than 2 points -> effective. Otherwise as reported (val drop / no gain).
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cache_matrix_20260722"

MODELS = [
    "gemma4_12b", "phi3_5_vision", "qwen3_5_9b", "internvl3_5_8b",
    "qwen3_5_4b", "glm4_6v_flash", "minicpm_v_4_5",
]
SEED = 20260717

CANON = {
    "run": BASE / "runs/ca_tme_gru",
    "thr": BASE / "thresholds",
    "sdr": BASE / "sdr",
    "tgt": BASE / "sdr_target",
}
STRONG = {
    "run": BASE / "runs/ca_tme_gru_sdr_strong",
    "thr": BASE / "thresholds_sdr_strong",
    "sdr": BASE / "sdr_strong",
    "tgt": BASE / "sdr_strong_target",
}


def _val_ac(run_root: Path, model: str) -> float | None:
    f = run_root / f"{model}_seed{SEED}" / "train_metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("best_val_balanced_accuracy_ac")
    except json.JSONDecodeError:
        return None


def _tau(thr_root: Path, model: str) -> float | None:
    f = thr_root / model / "thresholds.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("tau")
    except json.JSONDecodeError:
        return None


def _pattern_rows(tree: Path, model: str) -> list[dict]:
    d = tree / model
    if not d.exists():
        return []
    f = next(d.rglob("state_patterns.jsonl"), None)
    if f is None:
        return []
    rows: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _conflict_stats(rows: list[dict]) -> dict:
    """Pct Dominant / Confusion within Conflict samples + Conflict D median."""
    conf = [r for r in rows if r.get("sample_type") == "Conflict"]
    n = len(conf)
    if n == 0:
        return {"n": 0, "dom": None, "cfs": None, "d_med": None}
    dom = sum(1 for r in conf if r.get("pattern") == "Dominant")
    cfs = sum(1 for r in conf if r.get("pattern") == "Confusion")
    ds = [r["D"] for r in conf if isinstance(r.get("D"), (int, float))]
    return {
        "n": n,
        "dom": 100.0 * dom / n,
        "cfs": 100.0 * cfs / n,
        "d_med": statistics.median(ds) if ds else None,
    }


def _pair(c, s, fmt) -> str:
    def one(v):
        return fmt(v) if v is not None else "n/a"
    return f"{one(c)} -> {one(s)}"


def _pct(v) -> str:
    return f"{v:.1f}%"


def _f2(v) -> str:
    return f"{v:.2f}"


def main() -> int:
    summary_dir = BASE / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "SDR_STRONG_COMPARE.md"

    lines = [
        "# SDR strong vs canonical (7 weak-separation models)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"GRU, seed {SEED} | cells: canonical -> sdr_strong",
        "",
        "sdr_strong = sdr_aux_weight 2.0 + state supervision "
        "(d_supervision_weight 0.5, angular 0.2, d_aux 2/class, selection filters).",
        "",
        "| model | val_ac (C/A) | Dom% | Cfs% | tau | T-Dom% | T-D med | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes: list[str] = []

    for m in MODELS:
        ac_c, ac_s = _val_ac(CANON["run"], m), _val_ac(STRONG["run"], m)
        tau_c, tau_s = _tau(CANON["thr"], m), _tau(STRONG["thr"], m)
        src_c = _conflict_stats(_pattern_rows(CANON["sdr"], m))
        src_s = _conflict_stats(_pattern_rows(STRONG["sdr"], m))
        tgt_c = _conflict_stats(_pattern_rows(CANON["tgt"], m))
        tgt_s = _conflict_stats(_pattern_rows(STRONG["tgt"], m))

        # Verdict.
        if ac_s is None or src_s["dom"] is None:
            verdict = "INCOMPLETE"
        else:
            dom_up = (src_s["dom"] >= 2.0 * src_c["dom"]) if src_c["dom"] else (src_s["dom"] > 0)
            dom_up = dom_up or (src_s["dom"] > 50.0)
            ac_drop = None if ac_c is None else 100.0 * (ac_c - ac_s)
            if dom_up and (ac_drop is None or ac_drop <= 2.0):
                verdict = "effective"
            elif not dom_up:
                verdict = "no separation gain"
            else:
                verdict = f"val_ac -{ac_drop:.1f}pt"
            if ac_c is not None and ac_drop is not None and ac_drop > 2.0:
                notes.append(f"- {m}: val_ac {ac_c:.4f} -> {ac_s:.4f} (-{ac_drop:.1f}pt)")
        if verdict == "INCOMPLETE":
            notes.append(f"- {m}: strong artifacts missing (still running or failed)")

        lines.append(
            "| {m} | {ac} | {dom} | {cfs} | {tau} | {tdom} | {tdmed} | {v} |".format(
                m=m,
                ac=_pair(ac_c, ac_s, _f2),
                dom=_pair(src_c["dom"], src_s["dom"], _pct),
                cfs=_pair(src_c["cfs"], src_s["cfs"], _pct),
                tau=_pair(tau_c, tau_s, _f2),
                tdom=_pair(tgt_c["dom"], tgt_s["dom"], _pct),
                tdmed=_pair(tgt_c["d_med"], tgt_s["d_med"], _f2),
                v=verdict,
            )
        )

    lines += [
        "",
        "Columns: val_ac = best val balanced accuracy (C/A); Dom% = Dominant within Conflict "
        "(Source); Cfs% = Confusion within Conflict (Source); tau = calibrated Aligned threshold; "
        "T-Dom% = Dominant within Conflict (Target CH-SIMS v2); T-D med = median D of Target Conflict samples.",
        "",
        "Verdict rule: Dom% doubles or >50% AND val_ac drop <= 2pt = effective.",
        "",
        "Notes:",
        *(notes if notes else ["- none"]),
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
