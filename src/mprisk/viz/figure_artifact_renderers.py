"""Renderers for figures backed by real rows + provenance."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .figure_constants import MODEL_SPECS, STATUS_READY, UMAP_CONFIG
from .figure_input_loaders import (
    _validate_fig04_masks,
    _validate_fig06_masks,
    _validate_state_provenance,
)
from .figure_pending_helpers import _pending_axis
from .figure_validators import _as_bool, _require_columns


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
        _render_misread_bias(title, rows, provenance, output_path)
    elif key == "fig08_representation_comparison":
        _render_representation_comparison(title, rows, provenance, output_path)
    else:
        _render_evidence_table(title, rows, output_path)


def _render_sdr_distributions(
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    _require_columns(rows, {"model", "sample_type", "S", "D", "R", "metric", "value"})
    _validate_state_provenance(rows, provenance)
    _validate_fig04_masks(rows, provenance)
    figure, axes = plt.subplots(3, 3, figsize=(10.2, 7.2), constrained_layout=True)
    groups = ("Aligned", "Conflict")
    colors = {"Aligned": "#2a9d8f", "Conflict": "#d1495b"}
    for model_index, (model_key, model_label) in enumerate(MODEL_SPECS):
        for metric_index, metric in enumerate(("S", "D", "abs_R")):
            axis = axes[model_index, metric_index]
            values = [
                [
                    float(row["value"])
                    for row in rows
                    if row["model"] == model_key
                    and row["sample_type"] == group
                    and row["metric"] == metric
                ]
                for group in groups
            ]
            if any(not group_values for group_values in values):
                raise ValueError(f"Fig. 4 requires both classes for {model_key}/{metric}")
            boxes = axis.boxplot(values, tick_labels=groups, patch_artist=True)
            for patch, group in zip(boxes["boxes"], groups, strict=True):
                patch.set_facecolor(colors[group])
            axis.set_title(f"{model_label} | {metric}", fontsize=9)
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
    _validate_state_provenance(rows, provenance)
    masks = provenance.get("sample_masks") or {}
    if masks.get("patterns") != "representation_split=official_test":
        raise ValueError("Fig. 5 requires the official-test pattern mask")
    patterns = ("Consensus", "Balanced", "Dominant", "Confusion")
    colors = ("#315a96", "#f4b183", "#c95359", "#c8c8c8")
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    for axis, (model_key, model_label) in zip(axes, MODEL_SPECS, strict=True):
        bottoms = [0.0, 0.0]
        for pattern, color in zip(patterns, colors, strict=True):
            values = [
                sum(
                    float(row["proportion"])
                    for row in rows
                    if row["model"] == model_key
                    and row["sample_type"] == sample_type
                    and row["pattern"] == pattern
                )
                for sample_type in ("Aligned", "Conflict")
            ]
            axis.bar(("Aligned", "Conflict"), values, bottom=bottoms, label=pattern, color=color)
            bottoms = [left + current for left, current in zip(bottoms, values, strict=True)]
        if any(abs(total - 1.0) > 1e-6 for total in bottoms):
            raise ValueError(f"Fig. 5 proportions must sum to one for {model_key}")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(model_label)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


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
    _validate_state_provenance(rows, provenance)
    _validate_fig06_masks(rows, provenance)
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    for axis, (model_key, model_label) in zip(axes, MODEL_SPECS, strict=True):
        for sample_type, color in (("Aligned", "#2a9d8f"), ("Conflict", "#d1495b")):
            for emphasized, marker, alpha in ((False, "o", 0.28), (True, "D", 0.9)):
                selected = [
                    row
                    for row in rows
                    if row["model"] == model_key
                    and row["sample_type"] == sample_type
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
                        s=14,
                    )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.text(0.99, 0.96, "V lean", transform=axis.transAxes, ha="right", va="top")
        axis.text(0.99, 0.04, "T/A lean", transform=axis.transAxes, ha="right", va="bottom")
        axis.set(xlabel="D", ylabel="signed R", title=model_label)
    axes[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_misread_bias(
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    _require_columns(
        rows,
        {"panel", "model", "sample_type", "S", "D", "R", "direction_emphasized", "status"},
    )
    _validate_state_provenance(rows, provenance)
    if provenance.get("sample_masks") != {
        "misread": "Pending Misread annotations",
        "bias": (
            "representation_split=official_test and sample_type=Conflict and S<=kappa"
        ),
        "direction_emphasis": "D>tau",
    }:
        raise ValueError("Fig. 7 sample masks do not match the locked contract")
    thresholds_by_model = provenance["thresholds_by_model"]
    for row in rows:
        thresholds = thresholds_by_model[row["model"]]
        if (
            row["panel"] != "bias"
            or row["sample_type"] != "Conflict"
            or float(row["S"]) > float(thresholds["kappa"])
            or _as_bool(row["direction_emphasized"])
            != (float(row["D"]) > float(thresholds["tau"]))
        ):
            raise ValueError("Fig. 7 row violates official-test stable Conflict bias mask")
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), constrained_layout=True)
    for column, (model_key, model_label) in enumerate(MODEL_SPECS):
        _pending_axis(
            axes[0, column], f"{model_label} | Misread", "Pending Misread annotations"
        )
        bottom = [
            row
            for row in rows
            if row["panel"] == "bias"
            and row["model"] == model_key
            and row["sample_type"] == "Conflict"
            and row["status"] == STATUS_READY
        ]
        if not bottom:
            raise ValueError(f"Fig. 7 requires real stable Conflict bias rows for {model_key}")
        axis = axes[1, column]
        for emphasized, marker, alpha in ((False, "o", 0.3), (True, "D", 0.9)):
            selected = [
                row for row in bottom if _as_bool(row["direction_emphasized"]) is emphasized
            ]
            if selected:
                axis.scatter(
                    [float(row["D"]) for row in selected],
                    [float(row["R"]) for row in selected],
                    marker=marker,
                    alpha=alpha,
                    s=14,
                    color="#d1495b",
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{model_label} | stable Conflict D-signed R", fontsize=8)
        axis.set_xlabel("D")
        axis.set_ylabel("signed R")
        axis.text(0.98, 0.93, "V lean", ha="right", transform=axis.transAxes)
        axis.text(0.98, 0.07, "T/A lean", ha="right", transform=axis.transAxes)
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_representation_comparison(
    title: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_path: Path,
) -> None:
    _require_columns(
        rows,
        {
            "panel",
            "representation",
            "model",
            "protocol",
            "seed",
            "sample_id",
            "sample_type",
            "representation_split",
            "feature",
            "status",
        },
    )
    ac_rows = [row for row in rows if row["panel"] == "ac"]
    if not ac_rows or any(
        row["representation_split"] != "official_test"
        or row["status"] != STATUS_READY
        or row["model"] != "qwen3_vl_8b"
        or row["protocol"] != "VT"
        or row["seed"] != "20260717"
        for row in ac_rows
    ):
        raise ValueError(
            "Fig. 8 requires Ready qwen3_vl_8b/VT/seed20260717 official_test features"
        )
    sample_sets: dict[str, set[tuple[str, str]]] = {}
    for representation in ("Single-Point", "Trajectory MLP", "TME"):
        selected = [row for row in ac_rows if row["representation"] == representation]
        sample_keys = [(row["sample_id"], row["sample_type"]) for row in selected]
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError(f"Fig. 8 {representation} contains duplicate sample rows")
        sample_sets[representation] = set(sample_keys)
    if len({frozenset(samples) for samples in sample_sets.values()}) != 1:
        raise ValueError("Fig. 8 representations require exact held-out sample correspondence")
    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("Fig. 8 requires pinned umap-learn; PCA fallback is forbidden") from exc
    import numpy as np

    umap_version = importlib.metadata.version("umap-learn")
    expected_umap = {"package": "umap-learn", "version": umap_version, **UMAP_CONFIG}
    if provenance.get("representation_split") != "official_test":
        raise ValueError("Fig. 8 provenance must lock representation_split=official_test")
    if provenance.get("representative_backbone") != {
        "model": "qwen3_vl_8b",
        "protocol": "VT",
        "seed": "20260717",
    }:
        raise ValueError("Fig. 8 provenance must lock the registered representative backbone")
    if provenance.get("umap") != expected_umap:
        raise ValueError("Fig. 8 provenance must lock the installed UMAP version and parameters")
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), constrained_layout=True)
    for column, representation in enumerate(("Single-Point", "Trajectory MLP", "TME")):
        selected = [
            row
            for row in ac_rows
            if row["representation"] == representation
        ]
        if len(selected) <= UMAP_CONFIG["n_neighbors"]:
            raise ValueError("Fig. 8 UMAP requires more samples than fixed n_neighbors")
        decoded = [json.loads(str(row["feature"])) for row in selected]
        if any(not isinstance(feature, list) or not feature for feature in decoded):
            raise ValueError(f"Fig. 8 {representation} features must be non-empty vectors")
        dimensions = {len(feature) for feature in decoded}
        if len(dimensions) != 1:
            raise ValueError(f"Fig. 8 {representation} features must have one fixed dimension")
        features = np.asarray(decoded, dtype=float)
        if features.ndim != 2 or features.shape[0] != len(selected):
            raise ValueError(f"Fig. 8 {representation} features must have one fixed dimension")
        if not np.isfinite(features).all():
            raise ValueError(f"Fig. 8 {representation} features must be finite")
        if {row["sample_type"] for row in selected} != {"Aligned", "Conflict"}:
            raise ValueError(f"Fig. 8 {representation} requires both Aligned and Conflict")
        projection = UMAP(**UMAP_CONFIG).fit_transform(features)
        for sample_type, color in (("Aligned", "#2a9d8f"), ("Conflict", "#d1495b")):
            indexes = [i for i, row in enumerate(selected) if row["sample_type"] == sample_type]
            axes[0, column].scatter(
                projection[indexes, 0], projection[indexes, 1], color=color, label=sample_type, s=14
            )
        axes[0, column].set_title(f"{representation} | UMAP")
        axes[0, column].legend(fontsize=7)
        _pending_axis(
            axes[1, column],
            f"{representation} | Misread AUPRC",
            "Pending Misread annotations",
        )
    figure.suptitle(title)
    figure.text(
        0.5,
        0.01,
        f"umap-learn {umap_version}; n_neighbors={UMAP_CONFIG['n_neighbors']}; "
        f"min_dist={UMAP_CONFIG['min_dist']}; metric={UMAP_CONFIG['metric']}; "
        f"random_state={UMAP_CONFIG['random_state']}",
        ha="center",
        fontsize=7,
    )
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
