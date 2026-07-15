"""Pending-safe artifact-backed LaTeX exports for Tables I-III."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

TABLE_SCHEMA = "mprisk_paper_table_map_v2"
PENDING = "Pending"


def export_bundle_tables(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if config.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"table config schema must be {TABLE_SCHEMA}")
    specs = config.get("tables")
    if not isinstance(specs, Mapping) or len(specs) != 3:
        raise ValueError("paper table map must contain exactly Tables I-III")
    exported: dict[str, Any] = {}
    for key, spec in specs.items():
        rows = _load_rows(Path(str(spec["input"])), key=str(key))
        output = Path(str(spec["output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        columns = [str(column) for column in spec["columns"]]
        output.write_text(_latex_table(str(spec["title"]), columns, rows), encoding="utf-8")
        exported[str(key)] = {"output": str(output), "row_count": len(rows)}
    return {"schema": "mprisk_paper_table_export_v1", "tables": exported}


def _load_rows(path: Path, *, key: str) -> list[dict[str, str]]:
    if path.is_file() and path.stat().st_size:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if rows:
            return rows
    if key == "tab01_cross_backbone_results":
        return [
            {
                "Model / Protocol": model,
                "Diagnostic Acc. (Aligned)": PENDING,
                "Diagnostic Acc. (Conflict)": PENDING,
                "Dominant Modality Signature": PENDING,
            }
            for model in ("Qwen2.5-Omni-7B / VA", "Qwen3-VL-8B / VT", "InternVL3.5-8B / VT")
        ]
    if key == "tab02_conflict_misread_baselines":
        return [
            {
                "Representation": representation,
                "Accuracy": PENDING,
                "Macro-F1": PENDING,
                "AUPRC": PENDING,
                "Latency": PENDING,
            }
            for representation in ("Single-Point", "Trajectory MLP", "TME")
        ]
    return [
        {"Model": model, "Setting": "Conflict-only", "Misattribution": PENDING, "Quality": PENDING}
        for model in ("Qwen2.5-Omni-7B", "Qwen3-VL-8B", "InternVL3.5-8B")
    ]


def _latex_table(title: str, columns: list[str], rows: list[dict[str, str]]) -> str:
    alignment = "l" + "c" * (len(columns) - 1)
    lines = [
        "% Artifact-backed table; Pending marks unavailable registered evidence.",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_escape(column) for column in columns) + r" \\",
        "\\midrule",
    ]
    lines.extend(
        " & ".join(_escape(str(row.get(column, PENDING))) for column in columns) + r" \\"
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", f"% {title}", ""])
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
