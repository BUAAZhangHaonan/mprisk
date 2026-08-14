#!/usr/bin/env python3
"""Build SOURCE_STATE_BREAKDOWN.md: Source SDR state patterns by sample type.

For each of the 15 models (phi4_multimodal excluded), reads the Source SDR
state_patterns.jsonl under outputs/cache_matrix_20260722/sdr/ and groups rows
by sample_type (Conflict/Aligned) x state pattern, emitting the within-group
percentage of each pattern. All models ran protocol VT except gemma4_12b and
qwen2_5_omni_7b, which ran VA.

All numbers come from actual files under outputs/cache_matrix_20260722/.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cache_matrix_20260722"
SDR_SRC = BASE / "sdr"
OUT = BASE / "_summary/SOURCE_STATE_BREAKDOWN.md"

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


def _patterns_path(model: str) -> Path:
    proto = "VA" if model in VA_MODELS else "VT"
    run = f"{proto.lower()}_{RUN_STEM}"
    return (SDR_SRC / model / "outputs" / "states" / model / proto / run
            / REPR / "state_patterns.jsonl")


def _load_rows(model: str) -> list[dict]:
    path = _patterns_path(model)
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


def _breakdown_row(model: str) -> list[str]:
    rows = _load_rows(model)
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


def main() -> int:
    header = (["Model", "N", "Conflict N", "Aligned N"]
              + [f"Conflict->{p}%" for p in PATTERNS]
              + [f"Aligned->{p}%" for p in PATTERNS])
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for model in MODELS:
        lines.append("| " + " | ".join(_breakdown_row(model)) + " |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(MODELS)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
