"""Artifact-only vector PDF exports for the final ten-figure bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from mprisk.viz.figure_inputs import (  # noqa: E402
    PENDING_INPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    provenance_path,
)

FIGURE_SCHEMA = "mprisk_bundle_figure_map_v1"
STATUS_READY = "Ready"
STATUS_PENDING = "Pending"
LOCKED_TERMS = {
    "conflict": "Conflict",
    "aligned": "Aligned",
    "misread": "Misread",
    "non_misread": "Non-misread",
    "vision_lean": "V lean",
    "text_audio_lean": "T/A lean",
}
FORBIDDEN_PDF_TEXT = (
    "illustrative",
    "placeholder",
    "[xx]",
    "wrong-answer",
    "state consistency",
    "divergence",
    "arbitration",
)


def export_bundle_figures(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if config.get("schema") != FIGURE_SCHEMA:
        raise ValueError(f"figure config schema must be {FIGURE_SCHEMA}")
    figures = _export_group(config.get("figures"), expected_count=10)
    appendix = _export_group(config.get("appendix", {}), expected_count=None)
    return {
        "schema": "mprisk_bundle_figure_export_v1",
        "config": str(config_file),
        "figures": figures,
        "appendix": appendix,
    }


def _export_group(
    specs: object,
    *,
    expected_count: int | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(specs, Mapping):
        raise ValueError("figure group must be a mapping")
    if expected_count is not None and len(specs) != expected_count:
        raise ValueError(f"main figure map must contain exactly {expected_count} figures")
    exported: dict[str, dict[str, Any]] = {}
    for key, raw_spec in specs.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"figure {key} specification must be a mapping")
        title = _required_text(raw_spec, "title")
        input_path = Path(_required_text(raw_spec, "input"))
        output_path = Path(_required_text(raw_spec, "output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status, rows, provenance = _load_figure_input(str(key), input_path)
        if status == STATUS_READY:
            _render_artifact(
                key=str(key),
                title=title,
                rows=rows,
                provenance=provenance,
                output_path=output_path,
            )
        else:
            _render_pending(title=title, output_path=output_path)
        _validate_pdf_open(output_path)
        _validate_pdf_text(output_path)
        exported[str(key)] = {
            "status": status,
            "input": str(input_path),
            "output": str(output_path),
            "sha256": _sha256(output_path),
        }
    return exported


def _render_pending(*, title: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=14)
    axis.text(0.5, 0.42, STATUS_PENDING, ha="center", va="center", fontsize=18)
    figure.savefig(output_path, format="pdf", metadata={"Title": title, "Subject": STATUS_PENDING})
    plt.close(figure)


def _render_artifact(
    *,
    key: str,
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError(f"Ready figure input must contain real rows: {key}")
    if key == "fig04_sdr_distributions":
        _render_sdr_distributions(title, rows, provenance, output_path)
    elif key == "fig05_four_state_stacks":
        _render_four_state_stacks(title, rows, provenance, output_path)
    elif key == "fig06_stable_d_signed_r":
        _render_d_signed_r(title, rows, provenance, output_path)
    elif key == "fig07_misread_bias":
        _render_misread_bias(title, rows, output_path)
    elif key == "fig08_representation_comparison":
        _render_representation_comparison(title, rows, output_path)
    else:
        _render_evidence_table(title, rows, output_path)


def _render_sdr_distributions(
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    _require_columns(rows, {"sample_type", "S", "D", "R", "metric", "value"})
    _validate_fig04_masks(rows, provenance)
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)
    groups = ("Aligned", "Conflict")
    colors = {"Aligned": "#2a9d8f", "Conflict": "#d1495b"}
    for axis, metric in zip(axes, ("S", "D", "abs_R"), strict=True):
        values = [
            [
                float(row["value"])
                for row in rows
                if row["sample_type"] == group and row["metric"] == metric
            ]
            for group in groups
        ]
        if any(not group_values for group_values in values):
            raise ValueError("Fig. 4 requires both Aligned and Conflict rows")
        boxes = axis.boxplot(values, tick_labels=groups, patch_artist=True)
        for patch, group in zip(boxes["boxes"], groups, strict=True):
            patch.set_facecolor(colors[group])
        axis.set_title(metric)
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_four_state_stacks(
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    _require_columns(rows, {"model", "sample_type", "pattern", "count", "total", "proportion"})
    masks = provenance.get("sample_masks") or {}
    if masks.get("patterns") != "all_samples":
        raise ValueError("Fig. 5 requires the all-samples pattern mask")
    if provenance.get("source_sample_count") != provenance.get("included_sample_count"):
        raise ValueError("Fig. 5 all-samples provenance count mismatch")
    _render_stacked_rows(title, rows, output_path, category="pattern", value="proportion")


def _render_d_signed_r(
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    _require_columns(
        rows,
        {"S", "D", "R", "stable", "direction_emphasized", "sample_type"},
    )
    _validate_fig06_masks(rows, provenance)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for sample_type, color in (("Aligned", "#2a9d8f"), ("Conflict", "#d1495b")):
        for emphasized, marker, alpha in ((False, "o", 0.28), (True, "D", 0.9)):
            selected = [
                row
                for row in rows
                if row["sample_type"] == sample_type
                and _as_bool(row["direction_emphasized"]) is emphasized
            ]
            if selected:
                axis.scatter(
                    [float(row["D"]) for row in selected],
                    [float(row["R"]) for row in selected],
                    label=f"{sample_type}{' directional' if emphasized else ''}",
                    color=color,
                    marker=marker,
                    alpha=alpha,
                )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.text(0.99, 0.96, "V lean", transform=axis.transAxes, ha="right", va="top")
    axis.text(0.99, 0.04, "T/A lean", transform=axis.transAxes, ha="right", va="bottom")
    axis.set(xlabel="D", ylabel="signed R", title=title)
    axis.legend()
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_misread_bias(title: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    _require_columns(rows, {"panel", "category", "value", "status"})
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), constrained_layout=True)
    top = [row for row in rows if row["panel"] == "misread"]
    if not top or any(row["status"] != STATUS_READY for row in top):
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, f"Misread: {STATUS_PENDING}", ha="center", va="center")
    else:
        axes[0].bar([row["category"] for row in top], [float(row["value"]) for row in top])
        axes[0].set_title("Misread")
    bottom = [row for row in rows if row["panel"] == "bias"]
    if not bottom:
        raise ValueError("Fig. 7 requires real lower-panel bias rows")
    axes[1].bar([row["category"] for row in bottom], [float(row["value"]) for row in bottom])
    axes[1].set_title("V lean vs T/A lean")
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_representation_comparison(
    title: str, rows: list[dict[str, Any]], output_path: Path
) -> None:
    _require_columns(rows, {"panel", "representation", "metric", "value", "status"})
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    ac_rows = [row for row in rows if row["panel"] == "ac"]
    axes[0].bar(
        [row["representation"] for row in ac_rows],
        [float(row["value"]) for row in ac_rows],
    )
    axes[0].set_title("Conflict/Aligned")
    misread_rows = [row for row in rows if row["panel"] == "misread"]
    if not misread_rows or any(row["status"] != STATUS_READY for row in misread_rows):
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, f"Misread: {STATUS_PENDING}", ha="center", va="center")
    else:
        axes[1].bar(
            [row["representation"] for row in misread_rows],
            [float(row["value"]) for row in misread_rows],
        )
        axes[1].set_title("Misread")
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_evidence_table(title: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.axis("off")
    columns = list(rows[0])
    table_rows = [[str(row.get(column, "")) for column in columns] for row in rows[:12]]
    axis.table(cellText=table_rows, colLabels=columns, loc="center")
    axis.set_title(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_stacked_rows(
    title: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    category: str,
    value: str,
) -> None:
    labels = sorted({f"{row['model']} | {row['sample_type']}" for row in rows})
    categories = sorted({str(row[category]) for row in rows})
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    bottoms = [0.0] * len(labels)
    for item in categories:
        values = [
            sum(
                float(row[value])
                for row in rows
                if f"{row['model']} | {row['sample_type']}" == label and row[category] == item
            )
            for label in labels
        ]
        axis.barh(labels, values, left=bottoms, label=item)
        bottoms = [left + current for left, current in zip(bottoms, values, strict=True)]
    axis.set_title(title)
    axis.legend()
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_figure_input(
    figure_key: str,
    input_path: Path,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not input_path.is_file() or input_path.stat().st_size == 0:
        return STATUS_PENDING, [], {}
    suffix = input_path.suffix.casefold()
    if suffix == ".csv":
        sidecar = provenance_path(input_path)
        if not sidecar.is_file():
            raise ValueError(f"Ready CSV figure input requires provenance sidecar: {sidecar}")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_provenance(figure_key, provenance)
        status = str(provenance.get("status"))
        return status, _read_csv(input_path) if status == STATUS_READY else [], provenance
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON figure inputs must use a provenance envelope")
        if payload.get("schema") == PENDING_INPUT_SCHEMA:
            if payload.get("figure_key") != figure_key or payload.get("status") != STATUS_PENDING:
                raise ValueError("Pending JSON figure input identity/status mismatch")
            return STATUS_PENDING, [], payload
        _validate_provenance(figure_key, payload)
        rows = payload.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Ready JSON figure input rows must be a list of objects")
        return str(payload["status"]), rows, payload
    raise ValueError(f"figure input must be CSV or JSON: {input_path}")


def _validate_provenance(figure_key: str, provenance: dict[str, Any]) -> None:
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"figure provenance schema must be {PROVENANCE_SCHEMA}")
    if provenance.get("figure_key") != figure_key:
        raise ValueError("figure provenance key mismatch")
    if provenance.get("status") not in {STATUS_READY, STATUS_PENDING}:
        raise ValueError("figure provenance status must be Ready or Pending")
    if provenance.get("status") == STATUS_PENDING:
        return
    command = provenance.get("generated_command")
    sources = provenance.get("sources")
    if not isinstance(command, list) or not command:
        raise ValueError("Ready figure provenance requires generated_command argv")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Ready figure provenance requires source hashes")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not _is_sha256(source.get("sha256"))
        ):
            raise ValueError("figure provenance source path/sha256 is invalid")


def _validate_fig04_masks(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    masks = provenance.get("sample_masks") or {}
    if masks != {
        "S": "all_samples",
        "D": "S<=kappa",
        "abs_R": "S<=kappa and D>tau",
    }:
        raise ValueError("Fig. 4 sample masks do not match the locked contract")
    thresholds = provenance.get("thresholds") or {}
    kappa = float(thresholds["kappa"])
    tau = float(thresholds["tau"])
    for row in rows:
        metric = row["metric"]
        if metric not in {"S", "D", "abs_R"}:
            raise ValueError("Fig. 4 metric must be S, D, or abs_R")
        if metric == "D" and float(row["S"]) > kappa:
            raise ValueError("Fig. 4 D row violates stable mask")
        if metric == "abs_R" and (
            float(row["S"]) > kappa or float(row["D"]) <= tau
        ):
            raise ValueError("Fig. 4 abs_R row violates stable directional mask")
        expected = (
            float(row["S"])
            if metric == "S"
            else float(row["D"])
            if metric == "D"
            else abs(float(row["R"]))
        )
        if not abs(float(row["value"]) - expected) <= 1e-9:
            raise ValueError("Fig. 4 metric value does not match source S/D/R")


def _validate_fig06_masks(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    masks = provenance.get("sample_masks") or {}
    if masks != {
        "points": "S<=kappa",
        "direction_emphasis": "S<=kappa and D>tau",
    }:
        raise ValueError("Fig. 6 sample masks do not match the locked contract")
    thresholds = provenance.get("thresholds") or {}
    kappa = float(thresholds["kappa"])
    tau = float(thresholds["tau"])
    for row in rows:
        if float(row["S"]) > kappa or not _as_bool(row["stable"]):
            raise ValueError("Fig. 6 stable mask violation")
        if _as_bool(row["direction_emphasized"]) != (float(row["D"]) > tau):
            raise ValueError("Fig. 6 direction emphasis mask violation")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).casefold() == "true":
        return True
    if str(value).casefold() == "false":
        return False
    raise ValueError(f"expected true/false value, got {value!r}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.casefold())


def _require_columns(rows: list[dict[str, Any]], columns: set[str]) -> None:
    missing = columns - set(rows[0])
    if missing:
        raise ValueError(f"figure input is missing columns: {', '.join(sorted(missing))}")


def _required_text(spec: Mapping[str, Any], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"figure field {field} must be non-empty text")
    return value


def _validate_pdf_open(path: Path) -> None:
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"PDF validation failed for {path}: {completed.stderr.strip()}")


def _validate_pdf_text(path: Path) -> None:
    completed = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"PDF text extraction failed for {path}: {completed.stderr.strip()}")
    normalized = completed.stdout.casefold()
    matches = [term for term in FORBIDDEN_PDF_TEXT if term in normalized]
    if matches:
        raise ValueError(f"PDF contains forbidden text: {', '.join(matches)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
