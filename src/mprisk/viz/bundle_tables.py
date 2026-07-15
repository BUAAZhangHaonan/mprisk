"""Pending-safe artifact-backed LaTeX exports for Tables I-III."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

TABLE_SCHEMA = "mprisk_paper_table_map_v2"
TABLE_INPUT_SCHEMA = "mprisk_paper_table_input_v1"
PENDING = "Pending"
READY = "Ready"
TABLE_COLUMNS = {
    "tab01_cross_backbone_results": (
        "Model / Protocol",
        "Diagnostic Acc. (Aligned)",
        "Diagnostic Acc. (Conflict)",
        "Dominant Modality Signature",
    ),
    "tab02_conflict_misread_baselines": (
        "Method",
        "Accuracy",
        "Macro-F1",
        "AUPRC",
        "Latency",
    ),
    "tab03_downstream_quality": (
        "Model",
        "Setting",
        "Affective Misattribution",
        "Avg. Response Quality",
    ),
}


def export_bundle_tables(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if config.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"table config schema must be {TABLE_SCHEMA}")
    specs = config.get("tables")
    if not isinstance(specs, Mapping) or len(specs) != 3:
        raise ValueError("paper table map must contain exactly Tables I-III")
    exported: dict[str, Any] = {}
    for key, spec in specs.items():
        if key not in TABLE_COLUMNS:
            raise ValueError(f"unknown registered paper table: {key}")
        columns = tuple(str(column) for column in spec["columns"])
        if columns != TABLE_COLUMNS[key]:
            raise ValueError(f"table {key} columns do not match the locked schema")
        input_path = Path(str(spec["input"]))
        rows, status, provenance = _load_rows(input_path, key=str(key), columns=columns)
        output = Path(str(spec["output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_latex_table(str(spec["title"]), list(columns), rows), encoding="utf-8")
        exported[str(key)] = {
            "status": status,
            "input": str(input_path),
            "input_sha256": _sha256(input_path) if input_path.is_file() else None,
            "provenance": provenance,
            "output": str(output),
            "output_sha256": _sha256(output),
            "row_count": len(rows),
        }
    return {"schema": "mprisk_paper_table_export_v1", "tables": exported}


def _load_rows(
    path: Path, *, key: str, columns: tuple[str, ...]
) -> tuple[list[dict[str, str]], str, dict[str, Any] | None]:
    if path.is_file():
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        if not sidecar.is_file():
            raise ValueError(f"table input requires provenance sidecar: {sidecar}")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_provenance(provenance, key=key, columns=columns, input_path=path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError(f"table {key} CSV header does not match the locked schema")
            rows = [dict(row) for row in reader]
        if len(rows) != provenance["row_count"]:
            raise ValueError(f"table {key} row count does not match provenance")
        if provenance["status"] == READY:
            if not rows:
                raise ValueError(f"Ready table {key} requires real rows")
            if any(
                any(not row[column] or row[column] == PENDING for column in columns)
                for row in rows
            ):
                raise ValueError(f"Ready table {key} contains empty or Pending values")
            return rows, READY, provenance
        if rows:
            raise ValueError(f"Pending table {key} must not contain data rows")
        return _pending_rows(key), PENDING, provenance
    return _pending_rows(key), PENDING, None


def _pending_rows(key: str) -> list[dict[str, str]]:
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
                "Method": representation,
                "Accuracy": PENDING,
                "Macro-F1": PENDING,
                "AUPRC": PENDING,
                "Latency": PENDING,
            }
            for representation in ("Single-Point", "Trajectory MLP", "TME")
        ]
    return [
        {
            "Model": model,
            "Setting": "Conflict-only",
            "Affective Misattribution": PENDING,
            "Avg. Response Quality": PENDING,
        }
        for model in ("Qwen2.5-Omni-7B", "Qwen3-VL-8B", "InternVL3.5-8B")
    ]


def _validate_provenance(
    provenance: Any,
    *,
    key: str,
    columns: tuple[str, ...],
    input_path: Path,
) -> None:
    if not isinstance(provenance, dict) or provenance.get("schema") != TABLE_INPUT_SCHEMA:
        raise ValueError(f"table {key} provenance schema must be {TABLE_INPUT_SCHEMA}")
    if provenance.get("table_key") != key or provenance.get("status") not in {READY, PENDING}:
        raise ValueError(f"table {key} provenance identity/status mismatch")
    if tuple(provenance.get("columns") or ()) != columns:
        raise ValueError(f"table {key} provenance columns do not match the locked schema")
    if provenance.get("input_sha256") != _sha256(input_path):
        raise ValueError(f"table {key} input checksum mismatch")
    if not isinstance(provenance.get("row_count"), int) or provenance["row_count"] < 0:
        raise ValueError(f"table {key} provenance row_count is invalid")
    command = provenance.get("generated_command")
    sources = provenance.get("sources")
    if not isinstance(command, list) or not command or any(not str(part) for part in command):
        raise ValueError(f"table {key} provenance requires generated_command argv")
    if provenance["status"] == READY and (not isinstance(sources, list) or not sources):
        raise ValueError(f"Ready table {key} provenance requires hashed source artifacts")
    for source in sources or []:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not _is_sha256(source.get("sha256"))
        ):
            raise ValueError(f"table {key} provenance source hash is invalid")
        source_path = Path(source["path"])
        if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
            raise ValueError(f"table {key} provenance source checksum mismatch: {source_path}")


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
        " & ".join(_escape(str(row[column])) for column in columns) + r" \\"
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", f"% {title}", ""])
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
