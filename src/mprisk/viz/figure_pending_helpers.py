"""Pending-state visual primitives shared across figure layouts."""

from __future__ import annotations

from typing import Any

from matplotlib.lines import Line2D  # noqa: E402

from .figure_constants import STATUS_PENDING


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
