"""Artifact-only vector PDF exports for the final ten-figure bundle."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from mprisk.viz.figure_inputs import (  # noqa: E402
    CONCEPTUAL_INPUT_SCHEMA,
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
CONCEPTUAL_KEYS = {
    "fig01_problem_protocol",
    "fig02_representation_pipeline",
    "fig03_spherical_sdr",
    "figB1_representation_details",
}
MODEL_SPECS = (
    ("qwen2_5_omni_7b", "Qwen2.5-Omni-7B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
    ("internvl3_5_8b", "InternVL3.5-8B"),
)
MODEL_LABELS = tuple(label for _, label in MODEL_SPECS)
UMAP_CONFIG = {
    "random_state": 20260716,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine",
}
FULL_MODEL_LABELS = (
    "Gemma-3-4B",
    "Gemma-3-12B",
    "GLM-4.6V-Flash",
    "InternVL3.5-8B",
    "LLaVA-v1.5-7B",
    "LLaVA-OneVision-7B",
    "MiniCPM-V-2.6",
    "MiniCPM-V-4.5",
    "Phi-3.5-Vision",
    "Qwen2.5-VL-7B",
    "Qwen3-VL-8B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Gemma-4-12B",
    "Phi-4-Multimodal",
    "Qwen2.5-Omni-7B",
)


def export_bundle_figures(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if config.get("schema") != FIGURE_SCHEMA:
        raise ValueError(f"figure config schema must be {FIGURE_SCHEMA}")
    figures = _export_group(config.get("figures"), expected_count=10)
    appendix = _export_group(config.get("appendix", {}), expected_count=14)
    excluded = config.get("optional_excluded")
    if not isinstance(excluded, Mapping) or set(excluded) != {
        "figD2_j_lens",
        "figE3_self_correction",
    }:
        raise ValueError("figure map must explicitly exclude optional D2 and E3")
    return {
        "schema": "mprisk_bundle_figure_export_v1",
        "config": str(config_file),
        "figures": figures,
        "appendix": appendix,
        "optional_excluded": dict(excluded),
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
        if str(key) in CONCEPTUAL_KEYS:
            if status != STATUS_READY:
                raise ValueError(f"conceptual figure input must be Ready: {input_path}")
            _render_locked_layout(key=str(key), title=title, output_path=output_path)
        elif status == STATUS_READY:
            _render_artifact(
                key=str(key),
                title=title,
                rows=rows,
                provenance=provenance,
                output_path=output_path,
            )
        else:
            _render_locked_layout(key=str(key), title=title, output_path=output_path)
        _validate_pdf_open(output_path)
        _validate_pdf_text(output_path)
        exported[str(key)] = {
            "status": status,
            "input": str(input_path),
            "output": str(output_path),
            "sha256": _sha256(output_path),
        }
    return exported


def _render_locked_layout(*, key: str, title: str, output_path: Path) -> None:
    if key == "fig01_problem_protocol":
        _render_flow(
            title,
            [
                "Complete multimodal input",
                r"Pre-generation state at $t_0$",
                "Diagnostic affect description",
                "Misread\nPending annotations",
            ],
            output_path,
        )
        return
    if key == "fig02_representation_pipeline":
        _render_framework(title, output_path)
        return
    if key == "fig03_spherical_sdr":
        _render_sdr_method(title, output_path)
        return
    if key == "figB1_representation_details":
        _render_representation_details(title, output_path)
        return
    if key in {"fig04_sdr_distributions", "fig05_four_state_stacks", "fig06_stable_d_signed_r"}:
        _render_model_facets(key, title, output_path)
        return
    if key in {"fig07_misread_bias", "fig08_representation_comparison"}:
        _render_two_by_three(key, title, output_path)
        return
    if key == "fig09_conflict_case":
        _render_cards(
            title,
            ("Conflict input + GT", "Baseline response", "State-guided response"),
            output_path,
        )
        return
    if key == "fig10_four_pattern_cases":
        _render_cards(title, ("Confusion", "Consensus", "Balanced", "Dominant"), output_path)
        return
    _render_appendix_layout(key, title, output_path)


def _pending_axis(
    axis: Any,
    heading: str,
    message: str = STATUS_PENDING,
    *,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: tuple[float, ...],
    yticks: tuple[float, ...],
    xticklabels: tuple[str, ...] | None = None,
    legend_labels: tuple[str, ...] = (),
    legend_colors: tuple[str, ...] = (),
    legend_style: str = "line",
) -> None:
    axis.set_title(heading, fontsize=9)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xticks(xticks)
    axis.set_yticks(yticks)
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    axis.grid(True, color="#d7dce0", linewidth=0.6, alpha=0.75)
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, transform=axis.transAxes)
    if legend_labels:
        colors = legend_colors or tuple("#607d8b" for _ in legend_labels)
        if len(colors) != len(legend_labels):
            raise ValueError("pending legend labels and colors must have equal length")
        handles = [
            Line2D(
                [],
                [],
                color=color,
                marker="s" if legend_style == "patch" else None,
                linestyle="None" if legend_style == "patch" else "-",
                linewidth=1.8,
                markersize=7,
                label=label,
            )
            for label, color in zip(legend_labels, colors, strict=True)
        ]
        axis.legend(handles=handles, fontsize=6.5, loc="best", frameon=True)
    for spine in axis.spines.values():
        spine.set_color("#9aa0a6")


def _pending_card(axis: Any, heading: str, message: str = STATUS_PENDING) -> None:
    """Render a final card slot for non-coordinate case-study panels."""
    axis.set_title(heading, fontsize=9)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(())
    axis.set_yticks(())
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=8)
    for spine in axis.spines.values():
        spine.set_color("#9aa0a6")


def _add_pending_dr_framework(axis: Any) -> None:
    """Show the final D--R decision frame without inventing a calibrated tau."""
    axis.axhline(0.0, color="#7d8790", linewidth=0.8)
    axis.plot(
        (0.55, 0.55),
        (0.0, 1.0),
        color="#7d8790",
        linestyle="--",
        linewidth=0.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.57,
        0.95,
        r"$D=\tau$ threshold position Pending",
        ha="left",
        va="top",
        fontsize=6.5,
        transform=axis.transAxes,
    )
    axis.text(0.97, 0.84, "V lean", ha="right", fontsize=7, transform=axis.transAxes)
    axis.text(0.97, 0.10, "T/A lean", ha="right", fontsize=7, transform=axis.transAxes)


def _render_flow(title: str, labels: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 3.2), constrained_layout=True)
    axis.axis("off")
    for index, label in enumerate(labels):
        x = 0.13 + index * (0.74 / (len(labels) - 1))
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=.5", "fc": "white", "ec": "#3b6f8f"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(x + 0.19, 0.5),
                xytext=(x + 0.08, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#3b6f8f"},
            )
    axis.set_title(title, fontsize=14)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_framework(title: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 5.0), constrained_layout=True)
    axis.axis("off")
    boxes = (
        (0.09, "P=8 prompts\nM1: V | M2: T/A | M12: joint"),
        (0.31, "Full-layer trajectories\n3 x L x H at t0"),
        (0.53, "Shared TME\nlayer L2 + GRU -> unit z"),
        (0.74, "ordered u -> linear r\nProxy Anchor (A/C)"),
        (0.92, "S, D, signed R\nState Pattern"),
    )
    for x, label in boxes:
        axis.text(
            x,
            0.62,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=.45", "fc": "white", "ec": "#276678"},
        )
    for left, right in zip(boxes, boxes[1:], strict=False):
        axis.annotate(
            "",
            xy=(right[0] - 0.08, 0.62),
            xytext=(left[0] + 0.08, 0.62),
            arrowprops={"arrowstyle": "->"},
        )
    axis.text(
        0.64,
        0.20,
        "Offline Conflict/Aligned supervision only",
        ha="center",
        bbox={"fc": "white", "ec": "#8b5e3c", "linestyle": "--"},
    )
    axis.text(
        0.88,
        0.22,
        "Conflict-only Misread probe\nPending",
        ha="center",
        bbox={"fc": "white", "ec": "#8b5e3c", "linestyle": "--"},
    )
    axis.set_title(title, fontsize=14)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_sdr_method(title: str, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    axes[0].axis("off")
    axes[0].set_title("Spherical geometry")
    axes[0].text(0.03, 0.82, r"$d_g(a,b)=\arccos(\mathrm{clip}(a^Tb,-1,1))$", fontsize=10)
    axes[0].text(0.03, 0.65, r"$\mu_c=\mathrm{norm}(\sum_p z_{c,p})$", fontsize=10)
    axes[0].text(
        0.03,
        0.48,
        r"$s_c=P^{-1}\sum_p d_g^2(z_{c,p},\mu_c)$; $S=(s_1+s_2+s_{12})/3$",
        fontsize=9,
    )
    axes[0].text(0.03, 0.30, r"$D=d_g(\mu_1,\mu_2)/(\sqrt{s_1+s_2}+\epsilon)$", fontsize=9)
    axes[0].text(
        0.03,
        0.13,
        r"$R=[d_g(\mu_{12},\mu_2)-d_g(\mu_{12},\mu_1)]/"
        r"[d_g(\mu_1,\mu_2)+\epsilon]$",
        fontsize=8,
    )
    axes[0].text(0.03, 0.02, r"$R>0$: V lean; $R<0$: T/A lean", fontsize=9)
    axes[1].axis("off")
    axes[1].set_title("Hierarchical decision")
    axes[1].text(0.5, 0.82, "S > kappa?  Confusion", ha="center")
    axes[1].text(0.5, 0.60, "D <= tau?  Consensus", ha="center")
    axes[1].text(0.5, 0.38, "|R| <= delta_i?  Balanced", ha="center")
    axes[1].text(0.5, 0.16, "otherwise  Dominant", ha="center")
    figure.suptitle(title, fontsize=14)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_representation_details(title: str, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    details = (
        ("Single-Point", "M1/M2/M12 final points\n3H concat -> Linear"),
        ("Trajectory MLP", "3 x L x H\nLinear + GELU -> hidden128"),
        ("TME", "layer L2 -> GRU -> z\nordered u -> linear r\nProxy Anchor"),
    )
    for axis, (heading, body) in zip(axes, details, strict=True):
        axis.axis("off")
        axis.set_title(heading)
        axis.text(
            0.5,
            0.5,
            body,
            ha="center",
            va="center",
            bbox={"fc": "white", "ec": "#3b6f8f", "pad": 10},
        )
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_model_facets(key: str, title: str, output_path: Path) -> None:
    if key == "fig04_sdr_distributions":
        figure, axes = plt.subplots(3, 3, figsize=(10.2, 7.2), constrained_layout=True)
        metric_specs = (
            ("State Dispersion (S)", (0.0, 1.6), (0.0, 0.4, 0.8, 1.2, 1.6)),
            ("Modality Split (D)", (0.0, 2.0), (0.0, 0.5, 1.0, 1.5, 2.0)),
            ("Absolute Joint Lean (|R|)", (0.0, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0)),
        )
        for row, model in enumerate(MODEL_LABELS):
            for column, (metric, ylim, yticks) in enumerate(metric_specs):
                _pending_axis(
                    axes[row, column],
                    f"{model} | {metric}",
                    xlabel="Sample class",
                    ylabel=metric,
                    xlim=(-0.5, 1.5),
                    ylim=ylim,
                    xticks=(0.0, 1.0),
                    yticks=yticks,
                    xticklabels=("Aligned", "Conflict"),
                )
    elif key == "fig05_four_state_stacks":
        figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
        for axis, model in zip(axes, MODEL_LABELS, strict=True):
            _pending_axis(
                axis,
                model,
                xlabel="Sample class",
                ylabel="State Pattern proportion (%)",
                xlim=(-0.5, 1.5),
                ylim=(0.0, 100.0),
                xticks=(0.0, 1.0),
                yticks=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0),
                xticklabels=("Aligned", "Conflict"),
                legend_labels=("Confusion", "Consensus", "Balanced", "Dominant"),
                legend_colors=("#9e9e9e", "#2f5597", "#f4b183", "#c55a5a"),
                legend_style="patch",
            )
    elif key == "fig06_stable_d_signed_r":
        figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
        for axis, model in zip(axes, MODEL_LABELS, strict=True):
            _pending_axis(
                axis,
                model,
                xlabel="Modality Split (D)",
                ylabel="signed Joint Lean (R)",
                xlim=(0.0, 2.0),
                ylim=(-1.0, 1.0),
                xticks=(0.0, 0.5, 1.0, 1.5, 2.0),
                yticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
                legend_labels=("Aligned", "Conflict", r"direction: $D>\tau$"),
                legend_colors=("#2a9d8f", "#d1495b", "#4d4d4d"),
            )
            _add_pending_dr_framework(axis)
    else:
        raise ValueError(f"unsupported pending model facet: {key}")
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_two_by_three(key: str, title: str, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), constrained_layout=True)
    headings = (
        MODEL_LABELS if key == "fig07_misread_bias" else ("Single-Point", "Trajectory MLP", "TME")
    )
    for column, heading in enumerate(headings):
        if key == "fig07_misread_bias":
            _pending_axis(
                axes[0, column],
                f"{heading} | State-to-Misread",
                "Pending Misread annotations",
                xlabel="State-indicator quantile",
                ylabel="Misread rate (%)",
                xlim=(0.5, 5.5),
                ylim=(0.0, 100.0),
                xticks=(1.0, 2.0, 3.0, 4.0, 5.0),
                yticks=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0),
                legend_labels=("State Dispersion", "Modality Split", "Absolute Joint Lean"),
                legend_colors=("#5b8ff9", "#61d9a8", "#f6bd16"),
            )
            _pending_axis(
                axes[1, column],
                f"{heading} | stable Conflict D-signed R",
                xlabel="Modality Split (D)",
                ylabel="signed Joint Lean (R)",
                xlim=(0.0, 2.0),
                ylim=(-1.0, 1.0),
                xticks=(0.0, 0.5, 1.0, 1.5, 2.0),
                yticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
                legend_labels=("Conflict", r"direction: $D>\tau$"),
                legend_colors=("#d1495b", "#4d4d4d"),
            )
            _add_pending_dr_framework(axes[1, column])
        else:
            _pending_axis(
                axes[0, column],
                f"{heading} | UMAP",
                xlabel="UMAP-1",
                ylabel="UMAP-2",
                xlim=(-5.0, 5.0),
                ylim=(-5.0, 5.0),
                xticks=(-5.0, -2.5, 0.0, 2.5, 5.0),
                yticks=(-5.0, -2.5, 0.0, 2.5, 5.0),
                legend_labels=("Aligned", "Conflict"),
                legend_colors=("#2a9d8f", "#d1495b"),
            )
            _pending_axis(
                axes[1, column],
                f"{heading} | Misread AUPRC",
                "Pending Misread annotations",
                xlabel="Conflict samples retained (%)",
                ylabel="AUPRC",
                xlim=(5.0, 105.0),
                ylim=(0.0, 1.0),
                xticks=(10.0, 25.0, 50.0, 100.0),
                yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
            )
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_cards(title: str, labels: tuple[str, ...], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(labels), figsize=(11.0, 3.5), constrained_layout=True)
    for axis, label in zip(axes, labels, strict=True):
        _pending_card(axis, label)
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def _render_appendix_layout(key: str, title: str, output_path: Path) -> None:
    cards = {
        "figA1_case_types": ("Conflict", "Aligned", "Ambiguous"),
        "figA2_misread_cases": ("Misread", "Non-misread"),
        "figE2_pattern_cases": ("Confusion", "Consensus", "Balanced", "Dominant"),
    }
    if key == "figC5_model_patterns":
        figure, axis = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
        _pending_axis(
            axis,
            "16 models | 100% pattern stacks",
            "3 registered models Pending; 13 models Pending",
            xlabel="Model",
            ylabel="State Pattern proportion (%)",
            xlim=(-0.5, 15.5),
            ylim=(0.0, 100.0),
            xticks=tuple(float(index) for index in range(16)),
            yticks=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0),
            xticklabels=FULL_MODEL_LABELS,
            legend_labels=("Confusion", "Consensus", "Balanced", "Dominant"),
            legend_colors=("#9e9e9e", "#2f5597", "#f4b183", "#c55a5a"),
            legend_style="patch",
        )
        axis.tick_params(axis="x", labelrotation=65, labelsize=6)
    elif key in cards:
        headings = cards[key]
        figure, axes = plt.subplots(1, len(headings), figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading in zip(axes, headings, strict=True):
            message = (
                "Pending Misread annotations"
                if key == "figA2_misread_cases"
                else STATUS_PENDING
            )
            _pending_card(axis, heading, message)
    elif key == "figB2_prompt_stability_latency":
        figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading in zip(axes, MODEL_LABELS, strict=True):
            _pending_axis(
                axis,
                heading,
                xlabel="Equivalent prompts (P)",
                ylabel="Normalized value",
                xlim=(0.0, 17.0),
                ylim=(0.0, 1.0),
                xticks=(1.0, 2.0, 4.0, 8.0, 16.0),
                yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
                legend_labels=("State stability", "Latency"),
                legend_colors=("#2f5597", "#c55a5a"),
            )
    elif key == "figB3_delta_bootstrap_geometry":
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
        _pending_axis(
            axes[0],
            r"$\delta_i$ bootstrap",
            xlabel="Bootstrap resamples",
            ylabel=r"Prompt uncertainty $\delta_i$",
            xlim=(50.0, 2050.0),
            ylim=(0.0, 1.0),
            xticks=(100.0, 500.0, 1000.0, 2000.0),
            yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
            legend_labels=MODEL_LABELS,
            legend_colors=("#2f5597", "#c55a5a", "#f4a261"),
        )
        _pending_axis(
            axes[1],
            "Spherical State Pattern geometry",
            xlabel="Modality Split (D)",
            ylabel="signed Joint Lean (R)",
            xlim=(0.0, 2.0),
            ylim=(-1.0, 1.0),
            xticks=(0.0, 0.5, 1.0, 1.5, 2.0),
            yticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
        )
        _add_pending_dr_framework(axes[1])
    elif key == "figC1_ac_roc_pr":
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading, xlabel, ylabel in (
            (axes[0], "A/C ROC", "False-positive rate", "True-positive rate"),
            (axes[1], "A/C PR", "Recall", "Precision"),
        ):
            _pending_axis(
                axis,
                heading,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                xticks=(0.0, 0.25, 0.5, 0.75, 1.0),
                yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
                legend_labels=("Single-Point", "Trajectory MLP", "TME"),
                legend_colors=("#6c757d", "#f4a261", "#2f5597"),
            )
    elif key == "figC2_conflict_retention":
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading, ylabel, labels in (
            (axes[0], "Nested budget", "Retained Conflict samples (%)", MODEL_LABELS),
            (axes[1], "A/C metrics", "A/C classification score", ("Accuracy", "Macro-F1", "AUPRC")),
        ):
            _pending_axis(
                axis,
                heading,
                xlabel="Conflict budget (%)",
                ylabel=ylabel,
                xlim=(5.0, 105.0),
                ylim=(0.0, 100.0) if heading == "Nested budget" else (0.0, 1.0),
                xticks=(10.0, 25.0, 50.0, 100.0),
                yticks=(0.0, 25.0, 50.0, 75.0, 100.0)
                if heading == "Nested budget"
                else (0.0, 0.25, 0.5, 0.75, 1.0),
                legend_labels=labels,
                legend_colors=("#2f5597", "#c55a5a", "#f4a261"),
            )
    elif key == "figC3_seed_robustness":
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading, ylabel in (
            (axes[0], "Three-seed correlation", r"Spearman $\rho$"),
            (axes[1], "State Pattern agreement", "Agreement (%)"),
        ):
            _pending_axis(
                axis,
                heading,
                xlabel="Prompt-seed pair",
                ylabel=ylabel,
                xlim=(-0.5, 2.5),
                ylim=(-1.0, 1.0) if heading == "Three-seed correlation" else (0.0, 100.0),
                xticks=(0.0, 1.0, 2.0),
                yticks=(-1.0, -0.5, 0.0, 0.5, 1.0)
                if heading == "Three-seed correlation"
                else (0.0, 25.0, 50.0, 75.0, 100.0),
                xticklabels=("1-2", "1-3", "2-3"),
                legend_labels=MODEL_LABELS,
                legend_colors=("#2f5597", "#c55a5a", "#f4a261"),
            )
    elif key == "figC4_threshold_sensitivity":
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading, ylabel, labels, colors, style in (
            (
                axes[0],
                r"$\kappa/\tau/\delta_i$ sensitivity",
                "State Pattern agreement (%)",
                (r"$\kappa$", r"$\tau$", r"$\delta_i$"),
                ("#2f5597", "#c55a5a", "#f4a261"),
                "line",
            ),
            (
                axes[1],
                "State Pattern stack",
                "State Pattern proportion (%)",
                ("Confusion", "Consensus", "Balanced", "Dominant"),
                ("#9e9e9e", "#2f5597", "#f4b183", "#c55a5a"),
                "patch",
            ),
        ):
            _pending_axis(
                axis,
                heading,
                xlabel="Threshold multiplier",
                ylabel=ylabel,
                xlim=(0.75, 1.25),
                ylim=(0.0, 100.0),
                xticks=(0.8, 0.9, 1.0, 1.1, 1.2),
                yticks=(0.0, 25.0, 50.0, 75.0, 100.0),
                legend_labels=labels,
                legend_colors=colors,
                legend_style=style,
            )
    elif key == "figD1_misread_pr":
        figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        _pending_axis(
            axis,
            "Conflict-only Misread PR",
            "Pending Misread annotations",
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xticks=(0.0, 0.25, 0.5, 0.75, 1.0),
            yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
            legend_labels=("Single-Point", "Trajectory MLP", "TME", "State-Indices Readout"),
            legend_colors=("#6c757d", "#f4a261", "#2f5597", "#61a5c2"),
        )
    elif key == "figD3_latency_breakdown":
        figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        _pending_axis(
            axis,
            "Latency components",
            xlabel="Pipeline component",
            ylabel="Latency (s)",
            xlim=(-0.5, 3.5),
            ylim=(0.0, 60.0),
            xticks=(0.0, 1.0, 2.0, 3.0),
            yticks=(0.0, 15.0, 30.0, 45.0, 60.0),
            xticklabels=("Cache", "State", "Diagnostic", "Total"),
            legend_labels=MODEL_LABELS,
            legend_colors=("#2f5597", "#c55a5a", "#f4a261"),
        )
    elif key == "figE1_human_quality":
        figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), constrained_layout=True)
        for axis, heading in zip(axes, ("Relevance", "Helpfulness", "Safety"), strict=True):
            _pending_axis(
                axis,
                heading,
                xlabel="Response method",
                ylabel="Mean human rating",
                xlim=(-0.5, 1.5),
                ylim=(1.0, 5.0),
                xticks=(0.0, 1.0),
                yticks=(1.0, 2.0, 3.0, 4.0, 5.0),
                xticklabels=("Baseline", "Ours"),
                legend_labels=MODEL_LABELS,
                legend_colors=("#2f5597", "#c55a5a", "#f4a261"),
            )
    else:
        figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        _pending_card(axis, title)
    figure.suptitle(title)
    figure.savefig(output_path, format="pdf")
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
    figure, axes = plt.subplots(3, 3, figsize=(10.8, 7.5), constrained_layout=True)
    groups = ("Aligned", "Conflict")
    colors = {"Aligned": "#2a9d8f", "Conflict": "#d1495b"}
    metric_labels = {
        "S": "State Dispersion (S)",
        "D": "Modality Split (D)",
        "abs_R": "Absolute Joint Lean (|R|)",
    }
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
            axis.set_title(f"{model_label}\n{metric_labels[metric]}", fontsize=8.5)
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
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_title(model_label)
    axes[0].set_ylabel("Samples (%)")
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
        text_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8}
        axis.text(
            0.99,
            0.96,
            "V lean",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox=text_box,
        )
        axis.text(
            0.99,
            0.04,
            "T/A lean",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            bbox=text_box,
        )
        axis.set(
            xlabel="Modality Split (D)",
            ylabel="signed Joint Lean (R)",
            title=model_label,
        )
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
    pending_specs = (
        {
            "heading": "State Dispersion (S) vs Misread",
            "xlabel": "State-indicator quantile",
            "xlim": (0.0, 1.0),
            "xticks": (0.0, 0.25, 0.5, 0.75, 1.0),
            "xticklabels": None,
        },
        {
            "heading": "Modality Split (D) vs Misread",
            "xlabel": "State-indicator quantile",
            "xlim": (0.0, 1.0),
            "xticks": (0.0, 0.25, 0.5, 0.75, 1.0),
            "xticklabels": None,
        },
        {
            "heading": "State Pattern vs Misread",
            "xlabel": "State Pattern",
            "xlim": (-0.5, 3.5),
            "xticks": (0.0, 1.0, 2.0, 3.0),
            "xticklabels": ("Confusion", "Consensus", "Balanced", "Dominant"),
        },
    )
    for column, (model_key, model_label) in enumerate(MODEL_SPECS):
        pending = pending_specs[column]
        _pending_axis(
            axes[0, column],
            pending["heading"],
            "Pending Misread annotations",
            xlabel=pending["xlabel"],
            ylabel="Misread rate (%)",
            xlim=pending["xlim"],
            ylim=(0.0, 100.0),
            xticks=pending["xticks"],
            yticks=(0.0, 25.0, 50.0, 75.0, 100.0),
            xticklabels=pending["xticklabels"],
        )
        if column == 2:
            axes[0, column].tick_params(axis="x", labelsize=7)
            plt.setp(
                axes[0, column].get_xticklabels(),
                rotation=20,
                ha="right",
                rotation_mode="anchor",
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
        axis.set_xlabel("Modality Split (D)")
        axis.set_ylabel("signed Joint Lean (R)")
        text_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8}
        axis.text(
            0.98,
            0.93,
            "V lean",
            ha="right",
            transform=axis.transAxes,
            bbox=text_box,
        )
        axis.text(
            0.98,
            0.07,
            "T/A lean",
            ha="right",
            transform=axis.transAxes,
            bbox=text_box,
        )
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
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 7.2))
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.12,
        top=0.90,
        wspace=0.28,
        hspace=0.38,
    )
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
        axes[0, column].set_xlabel("UMAP-1")
        axes[0, column].set_ylabel("UMAP-2")
        axes[0, column].legend(fontsize=7)
        _pending_axis(
            axes[1, column],
            f"{representation} | Misread AUPRC",
            "Pending Misread annotations",
            xlabel="Conflict samples retained (%)",
            ylabel="AUPRC",
            xlim=(0.0, 100.0),
            ylim=(0.0, 1.0),
            xticks=(0.0, 25.0, 50.0, 75.0, 100.0),
            yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        )
    figure.suptitle(title)
    figure.text(
        0.5,
        0.025,
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
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_figure_input(
    figure_key: str,
    input_path: Path,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not input_path.is_file() or input_path.stat().st_size == 0:
        if figure_key in CONCEPTUAL_KEYS:
            raise ValueError(f"conceptual figure input is missing or empty: {input_path}")
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
        if payload.get("schema") == CONCEPTUAL_INPUT_SCHEMA:
            if (
                figure_key not in CONCEPTUAL_KEYS
                or payload.get("figure_key") != figure_key
                or payload.get("status") != STATUS_READY
                or payload.get("sources") != []
                or payload.get("sample_masks") != {"data_dependency": "none"}
            ):
                raise ValueError("conceptual figure input identity/status mismatch")
            return STATUS_READY, [], payload
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
        source_path = Path(source["path"])
        if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
            raise ValueError(f"figure provenance source checksum mismatch: {source_path}")


def _validate_fig04_masks(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    masks = provenance.get("sample_masks") or {}
    if masks != {
        "S": "representation_split=official_test",
        "D": "representation_split=official_test and S<=kappa",
        "abs_R": "representation_split=official_test and S<=kappa and D>tau",
    }:
        raise ValueError("Fig. 4 sample masks do not match the locked contract")
    thresholds_by_model = provenance.get("thresholds_by_model") or {}
    for row in rows:
        thresholds = thresholds_by_model.get(row["model"])
        if not isinstance(thresholds, dict):
            raise ValueError("Fig. 4 is missing per-model calibration thresholds")
        kappa = float(thresholds["kappa"])
        tau = float(thresholds["tau"])
        metric = row["metric"]
        if metric not in {"S", "D", "abs_R"}:
            raise ValueError("Fig. 4 metric must be S, D, or abs_R")
        if metric == "D" and float(row["S"]) > kappa:
            raise ValueError("Fig. 4 D row violates stable mask")
        if metric == "abs_R" and (float(row["S"]) > kappa or float(row["D"]) <= tau):
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
    thresholds_by_model = provenance.get("thresholds_by_model") or {}
    for row in rows:
        thresholds = thresholds_by_model.get(row["model"])
        if not isinstance(thresholds, dict):
            raise ValueError("Fig. 6 is missing per-model calibration thresholds")
        kappa = float(thresholds["kappa"])
        tau = float(thresholds["tau"])
        if float(row["S"]) > kappa or not _as_bool(row["stable"]):
            raise ValueError("Fig. 6 stable mask violation")
        if _as_bool(row["direction_emphasized"]) != (float(row["D"]) > tau):
            raise ValueError("Fig. 6 direction emphasis mask violation")


def _validate_state_provenance(
    rows: list[dict[str, Any]], provenance: dict[str, Any]
) -> None:
    if provenance.get("representation_split") != "official_test":
        raise ValueError("paper state figures require representation_split=official_test")
    source_count = provenance.get("source_sample_count")
    official_count = provenance.get("official_test_sample_count")
    excluded_count = provenance.get("excluded_non_official_test_count")
    counts = (source_count, official_count, excluded_count)
    if not all(isinstance(value, int) and value >= 0 for value in counts):
        raise ValueError(
            "paper state provenance requires non-negative source/included/excluded counts"
        )
    if source_count != official_count + excluded_count:
        raise ValueError("paper state provenance source count does not reconcile")
    models = {str(row["model"]) for row in rows}
    if models != {key for key, _ in MODEL_SPECS}:
        raise ValueError("paper state figures require all three registered model facets")
    thresholds_by_model = provenance.get("thresholds_by_model")
    if not isinstance(thresholds_by_model, dict) or set(thresholds_by_model) != models:
        raise ValueError("paper state provenance requires per-model calibration thresholds")
    split_identities = provenance.get("split_identities")
    calibration_identities = provenance.get("calibration_identities")
    if not isinstance(split_identities, list) or {
        str(item.get("model")) for item in split_identities if isinstance(item, dict)
    } != models:
        raise ValueError("paper state provenance requires one split identity per model")
    if any(
        item.get("representation_split") != "official_test"
        or not _is_sha256(item.get("split_assignment_sha256"))
        for item in split_identities
    ):
        raise ValueError("paper state split identity is invalid")
    if not isinstance(calibration_identities, list) or {
        str(item.get("model")) for item in calibration_identities if isinstance(item, dict)
    } != models:
        raise ValueError("paper state provenance requires one calibration identity per model")
    for identity in calibration_identities:
        if identity.get("model_key") != identity.get("model"):
            raise ValueError("paper state calibration model identity mismatch")
        if any(not str(identity.get(field, "")) for field in (
            "protocol",
            "prompt_set_key",
            "repr_key",
            "prompt_set_artifact_sha256",
            "encoder_checkpoint_sha256",
            "split_assignment_sha256",
            "embedding_manifest_sha256",
        )):
            raise ValueError("paper state calibration identity is incomplete")


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
