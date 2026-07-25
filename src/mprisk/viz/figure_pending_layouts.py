"""Pending-layout renderers for figures without real data yet."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .figure_constants import (
    FULL_MODEL_LABELS,
    MODEL_LABELS,
    STATUS_PENDING,
)
from .figure_pending_helpers import (
    _add_pending_dr_framework,
    _pending_axis,
    _pending_card,
)


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
