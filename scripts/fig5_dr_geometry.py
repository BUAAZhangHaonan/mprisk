"""Fig.5-style D-R geometry plot with modality bias annotations.

Stable samples only (S ≤ kappa) per paper definition. Conflict vs Aligned
shown as separate marker colors. Threshold lines and V-lean / T-lean regions
annotated.

Adapted from /home/team/lvshuyang/Multimodal-SplitBrain_v3/scripts/paper_figures/common/fig_D_R_dual_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from mprisk_viz.plotting import load_state_patterns

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def make_fig5_dr(
    *,
    state_patterns_path: str | Path,
    thresholds_path: str | Path,
    output_path: str | Path,
    split_filter: str = "official_test",
):
    df = load_state_patterns(state_patterns_path)
    with open(thresholds_path) as f:
        th = json.load(f)
    kappa = float(th["kappa"])
    tau = float(th["tau"])

    sub = df[df["split"] == split_filter].copy()
    print(f"[fig5] {len(sub)} samples on {split_filter}", flush=True)
    print(f"[fig5] kappa={kappa:.4f}, tau={tau:.4f}", flush=True)

    # Per the paper, D-R geometry is for STABLE samples (S ≤ kappa).
    # delta_i is per-sample; use the median delta for plotting the horizontal lines.
    stable = sub[sub["S"] <= kappa].copy()
    delta = float(stable["delta_i"].median())
    print(f"[fig5] stable samples (S<=kappa): {len(stable)} / {len(sub)}", flush=True)
    print(f"[fig5] median delta_i (for plotting ±delta lines): {delta:.4f}", flush=True)

    # Plot bounds
    d_max = float(stable["D"].quantile(0.99)) if len(stable) > 0 else 1.0
    d_plot = stable[stable["D"] <= d_max].copy()

    # Build figure
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)

    # Background KDE for density visualization
    if len(d_plot) > 10:
        try:
            sns.kdeplot(
                data=d_plot, x="D", y="R", fill=False, thresh=0.05, levels=8,
                color="lightgray", linewidths=0.6, alpha=0.45, ax=ax,
            )
        except Exception:
            pass

    # Scatter by sample_type (Conflict on top)
    conflict_color = "#D7191C"
    aligned_color = "#1F77B4"
    cmap = LinearSegmentedColormap.from_list("v_lean_t_lean", ["#2455A4", "#F8F8F8", "#C72F3A"], N=256)
    norm = Normalize(-1.5, 1.5)

    # Aligned first (background)
    a_sub = d_plot[d_plot["sample_type"] == "Aligned"]
    if len(a_sub) > 0:
        ax.scatter(
            a_sub["D"], a_sub["R"],
            c=aligned_color, s=22, alpha=0.55,
            edgecolors="white", linewidths=0.3,
            label=f"Aligned (n={len(a_sub)})", zorder=3,
        )

    # Conflict on top
    c_sub = d_plot[d_plot["sample_type"] == "Conflict"]
    if len(c_sub) > 0:
        ax.scatter(
            c_sub["D"], c_sub["R"],
            c=conflict_color, s=42, alpha=0.85, marker="^",
            edgecolors="white", linewidths=0.4,
            label=f"Conflict (n={len(c_sub)})", zorder=4,
        )

    # Threshold lines
    ax.axvline(tau, color="#444444", linestyle=(0, (5, 4)), linewidth=1.4, alpha=0.85, zorder=2)
    ax.axhline(+delta, color="#A33A40", linestyle=(0, (1, 2)), linewidth=1.4, alpha=0.75, zorder=2)
    ax.axhline(-delta, color="#2E58A6", linestyle=(0, (1, 2)), linewidth=1.4, alpha=0.75, zorder=2)
    ax.axhline(0, color="#888888", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

    # Threshold annotations
    x_text = tau + 0.4
    y_top = max(d_plot["R"].max() if len(d_plot) else 1.0, delta + 0.3)
    ax.text(x_text, y_top * 0.96, rf"$\tau={tau:.2f}$",
            color="#444444", fontsize=13, va="top")
    ax.text(d_max * 0.65, delta + 0.05, rf"$+\delta={delta:.2f}$",
            color="#A33A40", fontsize=12)
    ax.text(d_max * 0.65, -delta - 0.15, rf"$-\delta=-{delta:.2f}$",
            color="#2E58A6", fontsize=12)

    # Region labels (modality bias annotations)
    ax.text(d_max * 0.10, y_top * 0.85, "Consensus\n(low D, |R| small)",
            fontsize=11, color="#555555", ha="center", style="italic")
    ax.text(d_max * 0.78, y_top * 0.88, "V-dominant\n(D>τ, R>+δ)",
            fontsize=12, color="#C72F3A", ha="center", weight="bold")
    ax.text(d_max * 0.78, -y_top * 0.85, "T/A-dominant\n(D>τ, R<−δ)",
            fontsize=12, color="#2455A4", ha="center", weight="bold")
    ax.text(d_max * 0.45, 0.0, "Balanced\n(|R|≤δ)",
            fontsize=10, color="#888888", ha="center", va="center", style="italic")

    # Axes
    ax.set_xlim(0, d_max * 1.05)
    r_lim = max(delta * 2.5, float(np.abs(d_plot["R"]).max()) * 1.05) if len(d_plot) else 1.0
    ax.set_ylim(-r_lim, r_lim)
    ax.set_xlabel(r"Modality Split  $\mathcal{D}$", labelpad=8)
    ax.set_ylabel(r"Signed Joint Lean  $\mathcal{R}$", labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=5, width=1.0)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)

    # Legend
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fontsize=11)

    # Title and caption
    fig.suptitle(
        r"$\mathcal{D}$–$\mathcal{R}$ geometry of stable samples "
        f"($S\\leq\\kappa$, {split_filter})",
        fontsize=15, weight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.015,
        r"$M_1=$ Visual,  $M_2=$ Text    |    "
        r"$\mathcal{R}>0$: V-lean,  $\mathcal{R}<0$: T-lean    |    "
        f"qwen3_vl_8b / VT / SDR-aware LSTM-TME",
        ha="center", fontsize=11, style="italic", color="#444444",
    )

    # Colorbar showing R direction (V-lean to T-lean)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.66])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_ticks([-1.35, 0, 1.35])
    cbar.set_ticklabels([r"$M_2$-dom", "Neutral", r"$M_1$-dom"])
    cbar.ax.tick_params(labelsize=11, pad=6)

    fig.tight_layout(rect=[0, 0.04, 0.91, 0.95])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[fig5] saved: {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-patterns", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="official_test")
    args = parser.parse_args()
    make_fig5_dr(
        state_patterns_path=args.state_patterns,
        thresholds_path=args.thresholds,
        output_path=args.output,
        split_filter=args.split,
    )
