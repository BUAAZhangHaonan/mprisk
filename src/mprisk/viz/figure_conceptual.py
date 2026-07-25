"""Conceptual schematic figures (no real data, no provenance)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


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
