"""Aggregate artifact-backed bundle readiness into RUN_STATUS.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mprisk.viz.figure_inputs import PENDING, PROVENANCE_SCHEMA, provenance_path

RUN_RECORDS_SCHEMA = "mprisk_run_records_v1"


def build_run_status(
    config_path: str | Path,
    *,
    output_path: str | Path,
    records_path: str | Path | None = None,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    records = _load_records(records_path)
    groups = (("Main figures", config.get("figures", {})), ("Appendix", config.get("appendix", {})))
    lines = [
        "# RUN STATUS",
        "",
        "Status is derived only from declared figure inputs and machine-readable runtime records.",
        "",
        *_command_lines(records["commands"]),
        *_gpu_lines(records["gpus"]),
        *_cache_lines(records["caches"]),
        *_split_lines(records["splits"]),
        *_experiment_lines(records["experiments"]),
        *_visual_qa_lines(records["visual_qa"]),
    ]
    for heading, specs in groups:
        lines.extend(
            (
                f"## {heading}",
                "",
                "| Artifact | Status | Input | Output |",
                "|---|---|---|---|",
            )
        )
        for key, spec in specs.items():
            input_path = Path(str(spec["input"]))
            output = Path(str(spec["output"]))
            status = _figure_input_status(input_path)
            pdf_status = "Openable" if output.is_file() and output.stat().st_size else PENDING
            lines.append(
                f"| {key} | {status} | `{input_path}` | `{output}` ({pdf_status}) |"
            )
        lines.append("")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _load_records(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {
            "commands": [],
            "gpus": [],
            "caches": [],
            "experiments": [],
            "splits": [],
            "visual_qa": [],
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RUN_RECORDS_SCHEMA:
        raise ValueError(f"run records schema must be {RUN_RECORDS_SCHEMA}")
    result: dict[str, list[dict[str, Any]]] = {}
    for key in ("commands", "gpus", "caches", "splits", "experiments", "visual_qa"):
        rows = payload.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"run records {key} must be a list of objects")
        result[key] = rows
    return result


def _gpu_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## GPU Snapshot",
        "",
        "| GPU | Name | Memory Used/Total MiB | Utilization | Recorded |",
        "|---:|---|---:|---:|---|",
    ]
    if not rows:
        lines.extend(("| - | None recorded | - | - | - |", ""))
        return lines
    for row in rows:
        lines.append(
            f"| {row.get('physical_index', '')} | {row.get('name', '')} | "
            f"{row.get('memory_used_mib', '')}/{row.get('memory_total_mib', '')} | "
            f"{row.get('utilization_percent', '')}% | {row.get('recorded_at', '')} |"
        )
    lines.append("")
    return lines


def _split_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Representation Split",
        "",
        "| Split | Seed | Calibration fraction | Train | Validation | Calibration | Test | "
        "Manifest SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if not rows:
        lines.extend(("| None recorded | - | - | - | - | - | - | - |", ""))
        return lines
    for row in rows:
        lines.append(
            f"| {row.get('split_key', '')} | {row.get('seed', '')} | "
            f"{row.get('calibration_fraction', '')} | {row.get('relation_train', '')} | "
            f"{row.get('relation_val', '')} | {row.get('aligned_calibration', '')} | "
            f"{row.get('official_test', '')} | `{row.get('manifest_sha256', '')}` |"
        )
        lines.append(f"| Rule | - | `{row.get('ranking_rule', '')}` | - | - | - | - | - |")
    lines.append("")
    return lines


def _command_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Commands", "", "| ID | Status | Command | PID | GPU |", "|---|---|---|---:|---|"]
    if not rows:
        lines.extend(("| None recorded | Pending | - | - | - |", ""))
        return lines
    for row in rows:
        argv = row.get("argv")
        if not isinstance(argv, list) or any(not isinstance(part, str) for part in argv):
            raise ValueError("command argv must be a string list")
        gpu = row.get("gpu")
        gpu_text = "CPU"
        if isinstance(gpu, dict):
            index = gpu.get("physical_index")
            peak = gpu.get("peak_memory_mib")
            gpu_text = f"GPU {index}; peak {peak} MiB"
        lines.append(
            f"| {row.get('command_id', '')} | {row.get('status', '')} | "
            f"`{' '.join(argv)}` | {row.get('pid', '')} | {gpu_text} |"
        )
    lines.append("")
    return lines


def _cache_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Cache Status",
        "",
        "| Cache | Status | Complete | Failed | Missing |",
        "|---|---|---:|---:|---:|",
    ]
    if not rows:
        lines.extend(("| None recorded | Pending | - | - | - |", ""))
        return lines
    for row in rows:
        lines.append(
            f"| {row.get('cache_key', '')} | {row.get('status', '')} | "
            f"{row.get('complete', '')} | {row.get('failed', '')} | "
            f"{row.get('missing', '')} |"
        )
    lines.append("")
    return lines


def _experiment_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Experiments",
        "",
        "| Experiment | Status | Command | Reason |",
        "|---|---|---|---|",
    ]
    if not rows:
        lines.extend(("| None recorded | Pending | - | - |", ""))
        return lines
    for row in rows:
        lines.append(
            f"| {row.get('experiment_key', '')} | {row.get('status', '')} | "
            f"{row.get('command_id', '')} | {row.get('reason', '')} |"
        )
    lines.append("")
    return lines


def _visual_qa_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## PDF Visual QA",
        "",
        "| QA | Status | PDFs | PNGs | Embedded-font PDFs | Forbidden matches | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    if not rows:
        lines.extend(("| None recorded | Pending | - | - | - | - | - |", ""))
        return lines
    for row in rows:
        lines.append(
            f"| {row.get('qa_key', '')} | {row.get('status', '')} | "
            f"{row.get('pdf_count', '')} | {row.get('rendered_png_count', '')} | "
            f"{row.get('embedded_font_pdf_count', '')} | "
            f"{row.get('forbidden_match_count', '')} | {row.get('notes', '')} |"
        )
    lines.append("")
    return lines


def _figure_input_status(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return PENDING
    if path.suffix.casefold() == ".csv":
        sidecar = provenance_path(path)
        if not sidecar.is_file():
            return PENDING
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("schema") != PROVENANCE_SCHEMA:
            return PENDING
        return str(payload.get("status", PENDING))
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("status"):
            return str(payload["status"])
    return "Ready"
