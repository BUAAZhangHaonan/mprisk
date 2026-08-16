#!/usr/bin/env python3
"""Build TARGET_STATE_BREAKDOWN.md and STATE_INDICES_BY_GROUP.md.

TARGET_STATE_BREAKDOWN.md mirrors SOURCE_STATE_BREAKDOWN.md (see
build_source_state_breakdown.py) on the Target (ch_sims_v2 natural-domain)
SDR runs under outputs/cache_matrix_20260722/sdr_target/: per model, the
within-group percentage of each state pattern for Conflict and Aligned
samples.

STATE_INDICES_BY_GROUP.md holds one table per domain (Source, Target) with
median S_mean (dispersion), median D (d_score), and median signed R per
(model, sample_type) group, plus the Conflict-minus-Aligned median D gap.

All models ran protocol VT except gemma4_12b and qwen2_5_omni_7b (VA).
All numbers come from actual state_patterns.jsonl files under
outputs/cache_matrix_20260722/.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cache_matrix_20260722"
OUT_TARGET = BASE / "_summary/TARGET_STATE_BREAKDOWN.md"
OUT_INDICES = BASE / "_summary/STATE_INDICES_BY_GROUP.md"

DOMAINS = [("Source", BASE / "sdr"), ("Target", BASE / "sdr_target")]

RUN_STEM = "main_p8_seed20260717"
REPR = "tme_proxy_anchor_v1"
PATTERNS = ("Consensus", "Confusion", "Balanced", "Dominant")
SAMPLE_TYPES = ("Conflict", "Aligned")

MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "llava_v1_5_7b", "minicpm_v_2_6", "minicpm_v_4_5",
    "internvl3_5_8b", "phi3_5_vision", "qwen2_5_omni_7b", "qwen2_5_vl_7b",
    "qwen3_5_4b", "qwen3_5_9b", "qwen3_vl_8b",
]
VA_MODELS = {"gemma4_12b", "qwen2_5_omni_7b"}


def _patterns_path(model: str, domain_dir: Path) -> Path:
    proto = "VA" if model in VA_MODELS else "VT"
    run = f"{proto.lower()}_{RUN_STEM}"
    return (domain_dir / model / "outputs" / "states" / model / proto / run
            / REPR / "state_patterns.jsonl")


def _load_rows(model: str, domain_dir: Path) -> list[dict]:
    path = _patterns_path(model, domain_dir)
    if not path.exists():
        raise FileNotFoundError(f"missing state_patterns.jsonl for {model}: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}" if total else "—"


def _num(value: float, fmt: str) -> str:
    return format(value, fmt) if isinstance(value, (int, float)) else "—"


def _breakdown_row(model: str, domain_dir: Path) -> list[str]:
    rows = _load_rows(model, domain_dir)
    counts = {st: {p: 0 for p in PATTERNS} for st in SAMPLE_TYPES}
    for r in rows:
        st = r.get("sample_type")
        p = r.get("state_pattern") or r.get("pattern") or r.get("mode")
        if st in counts and p in counts[st]:
            counts[st][p] += 1
    n = len(rows)
    group_n = {st: sum(counts[st].values()) for st in SAMPLE_TYPES}
    cells = [model, str(n), str(group_n["Conflict"]), str(group_n["Aligned"])]
    for st in SAMPLE_TYPES:
        for p in PATTERNS:
            cells.append(_pct(counts[st][p], group_n[st]))
    return cells


def _indices_row(model: str, domain_dir: Path) -> list[str]:
    rows = _load_rows(model, domain_dir)
    vals: dict[str, dict[str, list[float]]] = {
        st: {"S": [], "D": [], "R": []} for st in SAMPLE_TYPES
    }
    for r in rows:
        st = r.get("sample_type")
        if st not in vals:
            continue
        for key, col in (("S", "S_mean"), ("D", "D"), ("R", "R")):
            raw = r.get(col)
            if isinstance(raw, (int, float)):
                vals[st][key].append(float(raw))
    cells = [model]
    med = {}
    for st in SAMPLE_TYPES:
        group = vals[st]
        med[st] = {k: statistics.median(group[k]) if group[k] else None
                   for k in ("S", "D", "R")}
        cells.append(str(len(vals[st]["S"])))
        cells.append(_num(med[st]["S"], ".4f"))
        cells.append(_num(med[st]["D"], ".1f"))
        cells.append(_num(med[st]["R"], ".3f"))
    if med["Conflict"]["D"] is not None and med["Aligned"]["D"] is not None:
        cells.append(f"{med['Conflict']['D'] - med['Aligned']['D']:+.1f}")
    else:
        cells.append("—")
    return cells


def _md_table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def build_target_breakdown() -> None:
    target_dir = dict(DOMAINS)["Target"]
    header = (["Model", "N", "Conflict N", "Aligned N"]
              + [f"Conflict->{p}%" for p in PATTERNS]
              + [f"Aligned->{p}%" for p in PATTERNS])
    body = [_breakdown_row(m, target_dir) for m in MODELS]
    OUT_TARGET.parent.mkdir(parents=True, exist_ok=True)
    OUT_TARGET.write_text("\n".join(_md_table(header, body)) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TARGET} ({len(MODELS)} models)")


def build_indices_by_group() -> None:
    header = (["Model", "C N", "C S_med", "C D_med", "C R_med",
               "A N", "A S_med", "A D_med", "A R_med", "D_gap (C-A)"])
    lines: list[str] = []
    for label, domain_dir in DOMAINS:
        lines.append(f"## {label}\n")
        body = [_indices_row(m, domain_dir) for m in MODELS]
        lines.extend(_md_table(header, body))
        lines.append("")
    OUT_INDICES.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDICES.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_INDICES} ({len(DOMAINS)} domains x {len(MODELS)} models)")


def main() -> int:
    build_target_breakdown()
    build_indices_by_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
