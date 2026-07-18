"""Template-v2 paper figures backed only by registered real experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from mprisk.viz.bundle_figures import (  # noqa: E402
    MODEL_SPECS,
    UMAP_CONFIG,
    _load_figure_input,
    _validate_fig04_masks,
    _validate_fig06_masks,
)
from mprisk.viz.figure_inputs import provenance_path  # noqa: E402

SCHEMA = "mprisk_template_v2_export_v1"
PENDING = "Pending Misread annotations"
BLUE = "#6F8DB7"
BLUE_DARK = "#58749C"
RED = "#D27068"
RED_DARK = "#B75752"
GRID = "#CFCFCF"
STATE_COLORS = {
    "Consensus": "#2F5597",
    "Balanced": "#F2B38C",
    "Dominant": "#C95D61",
    "Confusion": "#D2D2D2",
}
MODEL_MARKERS = {
    "qwen2_5_omni_7b": "o",
    "qwen3_vl_8b": "s",
    "internvl3_5_8b": "^",
}
FIGURES = {
    "fig04_sdr_distributions": "fig04_state_indices_template_v2",
    "fig05_four_state_stacks": "fig05_state_patterns_template_v2",
    "fig06_stable_d_signed_r": "fig06_geometry_template_v2",
    "fig07_misread_bias": "fig07_misread_associations_template_v2",
    "fig08_representation_comparison": "fig08_representation_quality_template_v2",
}


def export_template_v2_figures(
    *,
    source_root: str | Path = "outputs/paper_exports/figures",
    input_root: str | Path = "outputs/paper_exports/figures/template_v2",
    output_root: str | Path = "paper/figures/generated/template_v2",
    generated_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize real plotting inputs and export the five template-v2 figures."""
    source_dir = Path(source_root)
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    if source_dir.resolve() == input_dir.resolve():
        raise ValueError("template-v2 inputs must not overwrite the registered source inputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = list(generated_command or [sys.executable, "scripts/export_template_v2_figures.py"])
    result: dict[str, dict[str, Any]] = {}
    for figure_key, output_stem in FIGURES.items():
        source_csv = source_dir / f"{figure_key}.csv"
        status, source_rows, source_provenance = _load_figure_input(figure_key, source_csv)
        if status != "Ready" or not source_rows:
            raise ValueError(f"template-v2 source must be a non-empty Ready input: {source_csv}")
        rows = _materialize_rows(figure_key, source_rows, source_provenance)
        snapshot = input_dir / source_csv.name
        _write_csv(snapshot, rows)
        snapshot_provenance = _snapshot_provenance(
            figure_key=figure_key,
            source_csv=source_csv,
            source_provenance=source_provenance,
            generated_command=command,
            row_count=len(rows),
        )
        provenance_path(snapshot).write_text(
            json.dumps(snapshot_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pdf_path = output_dir / f"{output_stem}.pdf"
        png_path = output_dir / f"{output_stem}.png"
        _render(figure_key, rows, snapshot_provenance, pdf_path, png_path)
        if not pdf_path.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError(f"failed to produce an openable PDF: {pdf_path}")
        result[figure_key] = {
            "input": str(snapshot),
            "provenance": str(provenance_path(snapshot)),
            "pdf": str(pdf_path),
            "png": str(png_path),
            "input_sha256": _sha256(snapshot),
            "pdf_sha256": _sha256(pdf_path),
            "png_sha256": _sha256(png_path),
        }
    manifest = {
        "schema": SCHEMA,
        "source_root": str(source_dir),
        "input_root": str(input_dir),
        "output_root": str(output_dir),
        "figures": result,
        "pending_panels": {
            "fig05": "Conflict State Pattern by Misread outcome",
            "fig07": "State-to-Misread associations",
            "fig08": "Conflict-only Misread projections and supervision-budget curves",
        },
    }
    manifest_path = input_dir / "template_v2_export.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _materialize_rows(
    figure_key: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    if figure_key == "fig04_sdr_distributions":
        _validate_fig04_masks(rows, provenance)
        return rows
    if figure_key == "fig05_four_state_stacks":
        if provenance.get("sample_masks") != {"patterns": "representation_split=official_test"}:
            raise ValueError("Fig. 5 requires the registered official-test State Pattern mask")
        return rows
    if figure_key == "fig06_stable_d_signed_r":
        _validate_fig06_masks(rows, provenance)
        return _with_normalized_split(rows, provenance)
    if figure_key == "fig07_misread_bias":
        masks = provenance.get("sample_masks")
        if not isinstance(masks, dict) or masks.get("misread") != PENDING:
            raise ValueError("Fig. 7 must preserve Pending Misread annotations")
        if {row.get("panel") for row in rows} != {"bias"}:
            raise ValueError("Fig. 7 may only contain registered real bias rows")
        return _with_normalized_split(rows, provenance)
    if figure_key == "fig08_representation_comparison":
        masks = provenance.get("sample_masks")
        if not isinstance(masks, dict) or masks.get("misread") != PENDING:
            raise ValueError("Fig. 8 must preserve Pending Misread annotations")
        if not str(masks.get("conflict_retention", "")).startswith("Pending"):
            raise ValueError("Fig. 8 must preserve Pending Conflict-retention results")
        return _project_fig08_rows(rows, provenance)
    raise ValueError(f"unsupported template-v2 figure: {figure_key}")


def _with_normalized_split(
    rows: list[dict[str, Any]], provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    thresholds = provenance.get("thresholds_by_model")
    if not isinstance(thresholds, dict):
        raise ValueError("state geometry requires per-model calibration thresholds")
    materialized: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        tau = float(thresholds[model]["tau"])
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"invalid tau for {model}")
        item = dict(row)
        item["D_over_tau"] = float(row["D"]) / tau
        materialized.append(item)
    return materialized


def _project_fig08_rows(
    rows: list[dict[str, Any]], provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("template-v2 Fig. 8 requires pinned umap-learn") from exc
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    installed = importlib.metadata.version("umap-learn")
    expected = {"package": "umap-learn", "version": installed, **UMAP_CONFIG}
    if provenance.get("umap") != expected:
        raise ValueError("Fig. 8 source provenance does not match the installed locked UMAP")
    if {row.get("panel") for row in rows} != {"ac"}:
        raise ValueError("template-v2 Fig. 8 accepts only real C/A representation rows")
    materialized: list[dict[str, Any]] = []
    for representation in ("Single-Point", "Trajectory MLP", "TME"):
        selected = [row for row in rows if row["representation"] == representation]
        keys = [(row["sample_id"], row["sample_type"]) for row in selected]
        if len(keys) != len(set(keys)) or len(selected) <= UMAP_CONFIG["n_neighbors"]:
            raise ValueError(f"invalid held-out sample identity set for {representation}")
        features = np.asarray([json.loads(row["feature"]) for row in selected], dtype=float)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError(f"invalid real representation features for {representation}")
        labels = np.asarray([1 if row["sample_type"] == "Conflict" else 0 for row in selected])
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"Fig. 8 requires both C/A classes for {representation}")
        projection = UMAP(**UMAP_CONFIG).fit_transform(features)
        silhouette = float(silhouette_score(features, labels, metric="cosine"))
        neighbors = NearestNeighbors(n_neighbors=6, metric="cosine").fit(features)
        neighbor_ids = neighbors.kneighbors(return_distance=False)[:, :5]
        purity = float(np.mean(labels[neighbor_ids] == labels[:, None]))
        for row, point in zip(selected, projection, strict=True):
            materialized.append(
                {
                    "panel": "ac_umap",
                    "representation": representation,
                    "model": row["model"],
                    "protocol": row["protocol"],
                    "seed": row["seed"],
                    "sample_id": row["sample_id"],
                    "sample_type": row["sample_type"],
                    "representation_split": row["representation_split"],
                    "umap_x": float(point[0]),
                    "umap_y": float(point[1]),
                    "silhouette_original": silhouette,
                    "knn5_purity_original": purity,
                    "status": "Ready",
                }
            )
    return materialized


def _snapshot_provenance(
    *,
    figure_key: str,
    source_csv: Path,
    source_provenance: dict[str, Any],
    generated_command: list[str],
    row_count: int,
) -> dict[str, Any]:
    source_sidecar = provenance_path(source_csv)
    snapshot = dict(source_provenance)
    snapshot["generated_command"] = generated_command
    snapshot["sources"] = [
        {"path": str(source_csv.resolve()), "sha256": _sha256(source_csv)},
        {"path": str(source_sidecar.resolve()), "sha256": _sha256(source_sidecar)},
    ]
    snapshot["template_v2"] = {
        "schema": SCHEMA,
        "figure_key": figure_key,
        "layout_source": "taffc_fig_templates visual grammar only",
        "synthetic_data_used": False,
        "row_count": row_count,
        "pending_misread_panels": figure_key
        in {
            "fig05_four_state_stacks",
            "fig07_misread_bias",
            "fig08_representation_comparison",
        },
    }
    return snapshot


def _render(
    figure_key: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    pdf_path: Path,
    png_path: Path,
) -> None:
    _set_style()
    if figure_key == "fig04_sdr_distributions":
        figure = _render_fig04(rows)
    elif figure_key == "fig05_four_state_stacks":
        figure = _render_fig05(rows)
    elif figure_key == "fig06_stable_d_signed_r":
        figure = _render_fig06(rows)
    elif figure_key == "fig07_misread_bias":
        figure = _render_fig07(rows, provenance)
    elif figure_key == "fig08_representation_comparison":
        figure = _render_fig08(rows)
    else:
        raise ValueError(figure_key)
    figure.savefig(pdf_path, format="pdf", facecolor="white")
    figure.savefig(png_path, format="png", dpi=100, facecolor="white")
    plt.close(figure)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _render_fig04(rows: list[dict[str, Any]]) -> Any:
    rng = np.random.default_rng(20260718)
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 4. Pre-generation state indices in Conflict and Aligned samples",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )
    grid = figure.add_gridspec(1, 3, left=0.07, right=0.975, bottom=0.18, top=0.82, wspace=0.30)
    specs = (
        ("A", "S", r"State Dispersion ($S$)", r"$S$ (higher = less prompt-stable)"),
        ("B", "D", r"Modality Split ($\mathcal{D}$)", r"$\mathcal{D}$ (higher = stronger split)"),
        ("C", "abs_R", r"Joint Lean Magnitude ($|\mathcal{R}|$)", r"$|\mathcal{R}|$"),
    )
    for index, (letter, metric, heading, ylabel) in enumerate(specs):
        axis = figure.add_subplot(grid[0, index])
        groups = [
            np.asarray(
                [
                    float(row["value"])
                    for row in rows
                    if row["metric"] == metric and row["sample_type"] == kind
                ]
            )
            for kind in ("Aligned", "Conflict")
        ]
        if any(group.size == 0 for group in groups):
            raise ValueError(f"Fig. 4 lacks both C/A groups for {metric}")
        violins = axis.violinplot(groups, [1, 2], widths=0.72, showextrema=False, bw_method=0.38)
        for body, face, edge in zip(
            violins["bodies"], (BLUE, RED), (BLUE_DARK, RED_DARK), strict=True
        ):
            body.set_facecolor(face)
            body.set_edgecolor(edge)
            body.set_alpha(0.20)
        for position, group, color in zip((1, 2), groups, (BLUE_DARK, RED_DARK), strict=True):
            take = rng.choice(group.size, min(230, group.size), replace=False)
            axis.scatter(
                position + rng.normal(0, 0.055, take.size),
                group[take],
                s=11,
                color=color,
                alpha=0.62,
                linewidths=0,
            )
        boxes = axis.boxplot(
            groups,
            positions=[1, 2],
            widths=0.23,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 7,
            },
            medianprops={"color": "black"},
        )
        for patch, face in zip(boxes["boxes"], (BLUE_DARK, RED_DARK), strict=True):
            patch.set_facecolor(face)
            patch.set_alpha(0.78)
        difference = float(groups[1].mean() - groups[0].mean())
        low, high = _bootstrap_mean_difference(groups[0], groups[1], seed=20260718 + index)
        p_value = float(mannwhitneyu(groups[0], groups[1], alternative="two-sided").pvalue)
        y_min = min(float(group.min()) for group in groups)
        y_max = max(float(group.max()) for group in groups)
        span = max(y_max - y_min, 1e-9)
        axis.set_ylim(max(0.0, y_min - 0.08 * span), y_max + 0.30 * span)
        bracket = y_max + 0.10 * span
        axis.plot(
            [1.12, 1.12, 1.88, 1.88],
            [bracket, bracket + 0.03 * span, bracket + 0.03 * span, bracket],
            color="black",
        )
        axis.text(
            1.5, bracket + 0.045 * span, _significance(p_value), ha="center", fontweight="bold"
        )
        axis.text(1.5, bracket + 0.12 * span, rf"$\Delta={difference:.3g}$", ha="center")
        axis.text(
            0.98,
            0.025,
            f"95% CI [{low:.3g}, {high:.3g}]",
            transform=axis.transAxes,
            ha="right",
            fontsize=11,
            fontstyle="italic",
        )
        axis.text(
            0.5,
            1.03,
            f"Aligned n={groups[0].size:,}; Conflict n={groups[1].size:,}",
            transform=axis.transAxes,
            ha="center",
            fontsize=11,
            fontstyle="italic",
        )
        axis.text(-0.14, 1.13, letter, transform=axis.transAxes, fontsize=24, fontweight="bold")
        axis.set_title(heading, pad=48, fontweight="bold")
        axis.set_xticks([1, 2], ["Aligned", "Conflict"])
        axis.set_ylabel(ylabel)
        axis.yaxis.grid(True, linestyle=(0, (2, 3)), color=GRID, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        handles=[
            Patch(facecolor=BLUE_DARK, label="Aligned"),
            Patch(facecolor=RED_DARK, label="Conflict"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        frameon=False,
        ncol=2,
    )
    figure.text(
        0.5,
        0.035,
        "Official-test observations; eligibility masks follow the registered protocol.",
        ha="center",
        fontsize=12,
        fontstyle="italic",
    )
    return figure


def _render_fig05(rows: list[dict[str, Any]]) -> Any:
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 5. State Pattern distributions across input relations",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )
    grid = figure.add_gridspec(1, 2, left=0.08, right=0.97, bottom=0.18, top=0.82, wspace=0.28)
    axis = figure.add_subplot(grid[0, 0])
    sample_types = ("Aligned", "Conflict")
    patterns = ("Consensus", "Balanced", "Dominant", "Confusion")
    totals = {
        sample_type: sum(int(row["count"]) for row in rows if row["sample_type"] == sample_type)
        for sample_type in sample_types
    }
    bottoms = np.zeros(2)
    x = np.asarray([0.0, 1.05])
    for pattern in patterns:
        counts = np.asarray(
            [
                sum(
                    int(row["count"])
                    for row in rows
                    if row["sample_type"] == sample_type and row["pattern"] == pattern
                )
                for sample_type in sample_types
            ]
        )
        proportions = counts / np.asarray([totals[item] for item in sample_types])
        axis.bar(
            x,
            proportions,
            0.7,
            bottom=bottoms,
            color=STATE_COLORS[pattern],
            edgecolor="#333333",
            label=pattern,
        )
        for column, (count, proportion) in enumerate(zip(counts, proportions, strict=True)):
            if proportion >= 0.035:
                color = "white" if pattern in {"Consensus", "Dominant"} else "black"
                axis.text(
                    x[column],
                    bottoms[column] + proportion / 2,
                    f"{proportion:.0%}\n({count:,})",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=13,
                )
        bottoms += proportions
    axis.set_ylim(0, 1)
    axis.set_xticks(
        x, [f"Aligned\n(n={totals['Aligned']:,})", f"Conflict\n(n={totals['Conflict']:,})"]
    )
    axis.set_ylabel("Proportion of model-sample observations (%)")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.grid(True, linestyle=(0, (2, 3)), color=GRID)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        "State Pattern composition in\nAligned and Conflict samples", fontweight="bold", pad=25
    )
    axis.text(-0.16, 1.10, "(a)", transform=axis.transAxes, fontsize=22, fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    pending = figure.add_subplot(grid[0, 1])
    _pending_panel(
        pending,
        "Misread composition within each\nState Pattern (Conflict only)",
        xlabel="State Pattern",
        ylabel="Proportion of Conflict samples (%)",
        xticks=(0, 1, 2, 3),
        xticklabels=("Confusion", "Consensus", "Balanced", "Dominant"),
    )
    pending.text(-0.16, 1.10, "(b)", transform=pending.transAxes, fontsize=22, fontweight="bold")
    return figure


def _render_fig06(rows: list[dict[str, Any]]) -> Any:
    conflict = [row for row in rows if row["sample_type"] == "Conflict"]
    if not conflict:
        raise ValueError("Fig. 6 requires stable Conflict observations")
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 6. Geometric interpretation of stable Conflict states",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )
    axis = figure.add_axes([0.085, 0.16, 0.74, 0.72])
    cmap = LinearSegmentedColormap.from_list("lean", ["#2B5A9B", "#F8F8F8", "#C72F3A"])
    norm = Normalize(-1.0, 1.0)
    for model, label in MODEL_SPECS:
        selected = [row for row in conflict if row["model"] == model]
        axis.scatter(
            [float(row["D_over_tau"]) for row in selected],
            [float(row["R"]) for row in selected],
            c=[float(row["R"]) for row in selected],
            cmap=cmap,
            norm=norm,
            marker=MODEL_MARKERS[model],
            s=28,
            alpha=0.58,
            linewidths=0.25,
            edgecolors="#333333",
            label=label,
        )
    axis.axvline(1.0, color="#777777", linestyle=(0, (5, 4)), linewidth=1.5)
    axis.axhline(0.0, color="#777777", linestyle=(0, (2, 3)), linewidth=1.2)
    axis.set_xlabel(r"Normalized Modality Split $\mathcal{D}/\tau$")
    axis.set_ylabel(r"Signed Joint Lean $\mathcal{R}$")
    axis.set_xlim(left=0)
    axis.set_ylim(-1.05, 1.05)
    axis.grid(color=GRID, linestyle=(0, (2, 3)), alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(0.04, 0.93, "Consensus", transform=axis.transAxes, fontweight="bold", fontsize=18)
    axis.text(
        0.75, 0.93, "Dominant (V lean)", transform=axis.transAxes, fontweight="bold", fontsize=17
    )
    axis.text(
        0.75, 0.05, "Dominant (T/A lean)", transform=axis.transAxes, fontweight="bold", fontsize=17
    )
    axis.text(
        0.55,
        0.50,
        "Stable split geometry",
        transform=axis.transAxes,
        fontsize=16,
        fontstyle="italic",
    )
    axis.text(
        0.98,
        0.02,
        "Confusion filtered out",
        transform=axis.transAxes,
        ha="right",
        color="#777777",
        fontstyle="italic",
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=3, frameon=False)
    color_axis = figure.add_axes([0.855, 0.26, 0.022, 0.56])
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.solids.set_rasterized(False)
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(["T/A lean", "Neutral", "V lean"])
    figure.text(
        0.5,
        0.018,
        r"$\mathcal{D}/\tau=1$ is the calibrated split threshold; no fitted trend is imposed.",
        ha="center",
        fontsize=13,
        fontstyle="italic",
    )
    return figure


def _render_fig07(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> Any:
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 7. State-Misread associations and model-specific modality bias",
        fontsize=23,
        fontweight="bold",
        y=0.978,
    )
    grid = figure.add_gridspec(
        2, 3, left=0.05, right=0.93, bottom=0.12, top=0.88, wspace=0.30, hspace=0.42
    )
    headings = (
        "State Dispersion vs Misread Rate",
        "Modality Split vs Misread Rate",
        "State Pattern vs Misread Rate",
    )
    for column, heading in enumerate(headings):
        axis = figure.add_subplot(grid[0, column])
        xticklabels = ("Confusion", "Consensus", "Balanced", "Dominant") if column == 2 else None
        _pending_panel(
            axis,
            heading,
            xlabel="State Pattern" if column == 2 else "Normalized state coordinate",
            ylabel="Misread Rate (%)",
            xticks=(0, 1, 2, 3) if column == 2 else (0, 0.5, 1, 1.5, 2),
            xticklabels=xticklabels,
        )
        axis.text(
            -0.16,
            1.10,
            f"({chr(97 + column)})",
            transform=axis.transAxes,
            fontsize=17,
            fontweight="bold",
        )
    cmap = LinearSegmentedColormap.from_list("lean", ["#2455A4", "#F8F8F8", "#E52B2B"])
    norm = Normalize(-1, 1)
    thresholds = provenance["thresholds_by_model"]
    for column, (model, label) in enumerate(MODEL_SPECS):
        axis = figure.add_subplot(grid[1, column])
        selected = [row for row in rows if row["model"] == model and row["status"] == "Ready"]
        if not selected:
            raise ValueError(f"Fig. 7 lacks real modality-bias rows for {model}")
        axis.scatter(
            [float(row["D_over_tau"]) for row in selected],
            [float(row["R"]) for row in selected],
            c=[float(row["R"]) for row in selected],
            cmap=cmap,
            norm=norm,
            s=22,
            alpha=0.65,
            linewidths=0.2,
            edgecolors="#333333",
        )
        axis.axvline(1.0, color="#888888", linestyle=(0, (5, 4)), linewidth=1.1)
        axis.axhline(0, color="#888888", linestyle=(0, (2, 3)), linewidth=1.0)
        protocol = "V-A" if model == "qwen2_5_omni_7b" else "V-T"
        tau = float(thresholds[model]["tau"])
        v_count = sum(float(row["D"]) > tau and float(row["R"]) > 0 for row in selected)
        other_count = sum(float(row["D"]) > tau and float(row["R"]) < 0 for row in selected)
        other = "Audio" if protocol == "V-A" else "Text"
        axis.text(
            0.03,
            0.96,
            f"Directional: V={v_count}, {other}={other_count}",
            transform=axis.transAxes,
            va="top",
            fontsize=10,
        )
        axis.text(
            0.96,
            0.91,
            "V lean",
            transform=axis.transAxes,
            ha="right",
            color="#C72F3A",
            fontstyle="italic",
        )
        axis.text(
            0.96,
            0.07,
            f"{other} lean",
            transform=axis.transAxes,
            ha="right",
            color="#2455A4",
            fontstyle="italic",
        )
        axis.set_title(f"{label} ({protocol})", fontweight="bold")
        axis.set_xlabel(r"Normalized Modality Split $\mathcal{D}/\tau$")
        axis.set_ylabel(r"Signed Joint Lean $\mathcal{R}$" if column == 0 else "")
        axis.set_ylim(-1.05, 1.05)
        axis.grid(color=GRID, linestyle=(0, (2, 3)), alpha=0.55)
        axis.text(
            -0.16,
            1.08,
            f"({chr(100 + column)})",
            transform=axis.transAxes,
            fontsize=17,
            fontweight="bold",
        )
    color_axis = figure.add_axes([0.945, 0.17, 0.014, 0.25])
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.solids.set_rasterized(False)
    color_axis.set_title(r"$\mathcal{R}$", fontsize=11, pad=5)
    figure.text(
        0.5,
        0.035,
        "Bottom row: official-test stable Conflict observations. "
        "Misread-dependent panels remain pending.",
        ha="center",
        fontsize=12,
        fontstyle="italic",
    )
    return figure


def _render_fig08(rows: list[dict[str, Any]]) -> Any:
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 8. Frozen representation quality and pending Misread sensitivity",
        fontsize=23,
        fontweight="bold",
        y=0.978,
    )
    grid = figure.add_gridspec(
        2, 3, left=0.05, right=0.965, bottom=0.12, top=0.88, wspace=0.28, hspace=0.62
    )
    for column, representation in enumerate(("Single-Point", "Trajectory MLP", "TME")):
        selected = [row for row in rows if row["representation"] == representation]
        axis = figure.add_subplot(grid[0, column])
        for sample_type, color in (("Aligned", "#154BFF"), ("Conflict", "#FF2020")):
            subset = [row for row in selected if row["sample_type"] == sample_type]
            axis.scatter(
                [float(row["umap_x"]) for row in subset],
                [float(row["umap_y"]) for row in subset],
                s=16,
                color=color,
                alpha=0.75,
                linewidths=0.2,
                edgecolors="#333333",
                label=sample_type,
            )
        silhouette = float(selected[0]["silhouette_original"])
        purity = float(selected[0]["knn5_purity_original"])
        axis.text(
            0.97,
            0.95,
            f"Original-space\nSilhouette = {silhouette:.3f}\n5-NN purity = {purity:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        axis.set_title(representation, fontweight="bold")
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")
        axis.grid(color=GRID, linestyle=(0, (2, 3)), alpha=0.55)
        axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True, fontsize=10
        )
        axis.text(
            -0.16,
            1.10,
            f"({chr(97 + column)})",
            transform=axis.transAxes,
            fontsize=17,
            fontweight="bold",
        )
        pending = figure.add_subplot(grid[1, column])
        _pending_panel(
            pending,
            f"{representation}\nConflict supervision sensitivity",
            xlabel="Conflict supervision retained (%)",
            ylabel="Misread AUPRC",
            xticks=(10, 25, 50, 100),
            xticklabels=("10", "25", "50", "100"),
            title_size=15,
        )
        pending.text(
            -0.16,
            1.08,
            f"({chr(100 + column)})",
            transform=pending.transAxes,
            fontsize=17,
            fontweight="bold",
        )
    figure.text(
        0.5,
        0.035,
        "Top row: real C/A official-test representations; no label-dependent coordinate "
        "shifts. Bottom row awaits controlled Conflict-only Misread probes.",
        ha="center",
        fontsize=11.5,
        fontstyle="italic",
    )
    return figure


def _pending_panel(
    axis: Any,
    title: str,
    *,
    xlabel: str,
    ylabel: str,
    xticks: Sequence[float],
    xticklabels: Sequence[str] | None = None,
    title_size: float = 18,
) -> None:
    axis.set_title(title, fontweight="bold", pad=12, fontsize=title_size)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(xticks, xticklabels if xticklabels is not None else None)
    axis.set_ylim(0, 100 if "Rate" in ylabel or "Proportion" in ylabel else 1)
    axis.grid(color=GRID, linestyle=(0, (2, 3)), alpha=0.65)
    axis.text(
        0.5,
        0.5,
        PENDING,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#777777",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F7F7F7", "edgecolor": "#AAAAAA"},
    )
    axis.spines[["top", "right"]].set_visible(False)


def _bootstrap_mean_difference(
    aligned: np.ndarray, conflict: np.ndarray, *, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(2000, dtype=float)
    for index in range(samples.size):
        samples[index] = (
            rng.choice(conflict, conflict.size).mean() - rng.choice(aligned, aligned.size).mean()
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _significance(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty template-v2 input: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("template-v2 rows must have one stable field order")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_output_paths(root: str | Path) -> Iterable[Path]:
    """Return the original Fig. 4-8 PDFs whose hashes must remain unchanged."""
    base = Path(root)
    return (
        base / "fig04_sdr_distributions.pdf",
        base / "fig05_four_state_stacks.pdf",
        base / "fig06_stable_d_signed_r.pdf",
        base / "fig07_misread_bias.pdf",
        base / "fig08_representation_comparison.pdf",
    )
