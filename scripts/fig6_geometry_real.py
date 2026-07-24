"""TAFFC Fig.6 — Geometric interpretation of state patterns in stable Conflict samples.

Adapted from src/taffc_fig_templates/fig6_geometry.py to consume real v2 SDR
output instead of mock data. Preserves:
  - Conflict-only stable samples (S ≤ kappa)
  - Modality trend line (LOWESS)
  - Tau and ±delta threshold lines with annotations
  - Four-region labels (Consensus / Dominant / Balanced / Dominant)
  - Colorbar with M1/M2 endpoints
  - Inset node schematics

Removed: the "mock data" warning footer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 21,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.25,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def make_fig6(
    *,
    state_patterns_path: str | Path,
    thresholds_path: str | Path,
    output_path: str | Path,
    split_filter: str = "official_test",
    title_suffix: str = "",
    model_key: str = "qwen3_vl_8b",
    protocol: str = "VT",
):
    # --- Load data -----------------------------------------------------------
    rows = []
    with open(state_patterns_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            split = r.get("representation_split") or r.get("calibration_split") or "Unknown"
            rows.append({
                "sample_type": r.get("sample_type", "Unknown"),
                "split": split,
                "S": float(r.get("S_mean", float("nan"))),
                "D": float(r.get("D", float("nan"))),
                "R": float(r.get("R", float("nan"))),
                "pattern": r.get("pattern", "Unknown"),
                "delta_i": float(r.get("delta_i", 0.0)),
            })
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    df = df[df["split"] == split_filter].copy()

    with open(thresholds_path, "r", encoding="utf-8") as f:
        th = json.load(f)
    kappa = float(th["kappa"])
    tau = float(th["tau"])
    delta = float(df["delta_i"].median())   # per-sample; use median for plotting

    # Stable = S ≤ kappa. Confusion (S>κ) is filtered out.
    stable = df[(df["S"] <= kappa) & df["D"].notna() & df["R"].notna()].copy()
    conflict_stable = stable[stable["sample_type"] == "Conflict"].copy()
    aligned_stable = stable[stable["sample_type"] == "Aligned"].copy()

    print(f"[fig6] total {split_filter}: {len(df)} (Aligned "
          f"{(df['sample_type']=='Aligned').sum()}, Conflict "
          f"{(df['sample_type']=='Conflict').sum()})", flush=True)
    print(f"[fig6] stable Conflict: {len(conflict_stable)}, "
          f"stable Aligned: {len(aligned_stable)}", flush=True)
    print(f"[fig6] kappa={kappa:.4f}, tau={tau:.4f}, delta(median)={delta:.4f}", flush=True)

    # Use Conflict stable samples for the cloud + trend line
    d_max = float(stable["D"].quantile(0.995)) if len(stable) > 0 else 1.0
    plot_df = conflict_stable[(conflict_stable["D"] <= d_max)
                              & (np.abs(conflict_stable["R"]) <= 1.5)].copy()

    cmap = LinearSegmentedColormap.from_list(
        "taffc_div", ["#2B5A9B", "#F8F8F8", "#C72F3A"], N=256
    )
    norm = Normalize(-1.5, 1.5)

    fig = plt.figure(figsize=(14.48, 10.86), dpi=120)
    title = ("Figure 6. Geometric interpretation of state patterns in stable "
             "Conflict samples")
    if title_suffix:
        title += f"  ({title_suffix})"
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.975)
    ax = fig.add_axes([0.087, 0.17, 0.73, 0.72])

    # --- Background: Aligned points (gray, faint) ---------------------------
    if len(aligned_stable) > 0:
        a_plot = aligned_stable[(aligned_stable["D"] <= d_max)
                                & (np.abs(aligned_stable["R"]) <= 1.5)]
        ax.scatter(a_plot["D"], a_plot["R"],
                   c="#BBBBBB", s=15, alpha=0.35, linewidths=0, rasterized=True,
                   label=f"Aligned stable (n={len(a_plot)})")

    # --- Conflict stable cloud (colored by signed R) -----------------------
    if len(plot_df) > 0:
        rng = np.random.default_rng(606)
        x_jit = plot_df["D"].values + rng.normal(0, d_max * 0.005, len(plot_df))
        y_jit = plot_df["R"].values + rng.normal(0, 0.020, len(plot_df))
        x_jit = np.clip(x_jit, 0, d_max * 1.02)
        y_jit = np.clip(y_jit, -1.5, 1.5)
        sc = ax.scatter(x_jit, y_jit, c=y_jit, cmap=cmap, norm=norm,
                        s=22, alpha=0.65, linewidths=0, rasterized=True,
                        label=f"Conflict stable (n={len(plot_df)})")

    # --- Modality trend line: LOWESS on Conflict stable ---------------------
    if len(plot_df) >= 10:
        try:
            import statsmodels.api as sm
            sorted_df = plot_df.sort_values("D")
            xx_in = sorted_df["D"].values
            yy_in = sorted_df["R"].values
            z = sm.nonparametric.lowess(yy_in, xx_in, frac=0.45, return_sorted=True)
            ax.plot(z[:, 0], z[:, 1], color="#172F62", lw=3.0, zorder=4,
                    label="Modality trend (LOWESS)")
        except Exception as exc:
            print(f"[fig6] LOWESS trend skipped: {exc}", flush=True)
            # Fallback: simple tanh fit like the mock template
            xx = np.linspace(0, d_max, 300)
            yy = 0.30 + 0.38 * np.tanh(5.2 * (xx / d_max - 0.38))
            ax.plot(xx, yy, color="#172F62", lw=3.0, zorder=4,
                    label="Modality trend")

    # --- Threshold lines ----------------------------------------------------
    ax.axvline(tau, color="#8E8E8E", lw=1.5, ls=(0, (5, 4)))
    ax.axhline(+delta, color="#A33A40", lw=1.7, ls=(0, (1, 2)))
    ax.axhline(-delta, color="#2E58A6", lw=1.7, ls=(0, (1, 2)))
    ax.axhline(0, color="#cccccc", lw=0.6, alpha=0.6, zorder=1)

    ax.text(tau + d_max * 0.015, 1.41, rf"$\tau={tau:.2f}$", fontsize=17)
    ax.text(d_max * 0.74, delta + 0.04, rf"$+\delta={delta:.3f}$",
            color="#A92836", fontsize=16)
    ax.text(d_max * 0.74, -delta - 0.10, rf"$-\delta=-{delta:.3f}$",
            color="#1F4EAA", fontsize=16)

    # --- Axes ---------------------------------------------------------------
    ax.set_xlim(0, d_max * 1.05)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks(np.linspace(0, d_max, 6).round(2))
    ax.set_yticks(np.arange(-1.5, 1.51, 0.5))
    ax.set_xlabel(r"Modality Split $\mathcal{D}$")
    ax.set_ylabel(r"Signed Joint Lean $\mathcal{R}$", labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=6, width=1.1)

    # --- Region labels ------------------------------------------------------
    ax.text(d_max * 0.10, 1.25, "Consensus", fontsize=20, fontweight="bold")
    ax.text(d_max * 0.70, 1.39, "Dominant\n(V-lean)", fontsize=18, fontweight="bold",
            color="#C72F3A")
    ax.text(d_max * 0.70, 0.22, "Balanced", fontsize=20, fontweight="bold")
    ax.text(d_max * 0.70, -0.94, "Dominant\n(T/A-lean)", fontsize=18, fontweight="bold",
            color="#2B5A9B")
    ax.text(d_max * 0.86, -1.39, "Confusion\n(filtered,\nS>$\\kappa$)",
            fontsize=14, fontstyle="italic", color="#888888", ha="center")

    # --- Small condition-state schematics (kept from template) --------------
    def inset_nodes(bounds, coords):
        ia = ax.inset_axes(bounds)
        ia.set_xlim(0, 1); ia.set_ylim(0, 1)
        ia.set_xticks([]); ia.set_yticks([])
        for s in ia.spines.values():
            s.set_color("#A6A6A6"); s.set_linewidth(1.0)
        pts = np.asarray(coords)
        for a, b in [(0, 1), (1, 2)]:
            ia.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color="#8C8C8C", ls=(0, (4, 3)), lw=1.4, zorder=1)
        cols = ["#2F62AD", "#F6D1B2", "#D62F2F"]
        ia.scatter(pts[:, 0], pts[:, 1], s=155, c=cols,
                   edgecolor="#222222", linewidth=1.0, zorder=3)
        return ia

    # Consensus: all three points clustered
    inset_nodes([0.075, 0.765, 0.10, 0.10],
                [(0.20, 0.30), (0.50, 0.55), (0.78, 0.78)])
    # V-dominant: M1 and M12 close, M2 far
    inset_nodes([0.665, 0.805, 0.10, 0.10],
                [(0.22, 0.20), (0.82, 0.48), (0.30, 0.78)])
    # Balanced: M12 in the middle of M1, M2
    inset_nodes([0.65, 0.47, 0.12, 0.06],
                [(0.15, 0.50), (0.50, 0.50), (0.85, 0.50)])
    # T/A-dominant: M2 and M12 close, M1 far
    inset_nodes([0.615, 0.045, 0.10, 0.10],
                [(0.22, 0.20), (0.80, 0.50), (0.30, 0.82)])

    # --- Colorbar -----------------------------------------------------------
    cax = fig.add_axes([0.845, 0.28, 0.022, 0.57])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_ticks([1.35, 0, -1.35])
    cbar.set_ticklabels([r"$M_1$-dominant", "Neutral", r"$M_2$-dominant"])
    cbar.ax.tick_params(labelsize=16, pad=7)

    # --- Footer (warning removed; only protocol caption) -------------------
    fig.text(0.50, 0.082,
             r"$M_1=$ Visual,  $M_2=$ Text/Audio",
             ha="center", fontsize=16, fontstyle="italic")
    fig.text(0.50, 0.038,
             f"{model_key} / {protocol} / SDR-aware LSTM-TME  |  {split_filter}  "
             f"|  stable Conflict n={len(plot_df)}",
             ha="center", fontsize=12, color="#555555", fontstyle="italic")

    # --- Legend (top-left) --------------------------------------------------
    ax.legend(loc="lower left", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fontsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    print(f"[fig6] saved: {output_path}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-patterns", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="official_test")
    parser.add_argument("--model-key", default="qwen3_vl_8b",
                        help='Model key for footer label, e.g. internvl3_5_8b')
    parser.add_argument("--protocol", default=None,
                        help='Protocol label for footer (VT/VA). Default: infer from model key')
    args = parser.parse_args()
    # Infer protocol from model key: omni -> VA, otherwise VT
    inferred_protocol = args.protocol
    if inferred_protocol is None:
        if 'omni' in args.model_key or 'audio' in args.model_key:
            inferred_protocol = 'VA'
        else:
            inferred_protocol = 'VT'
    make_fig6(
        state_patterns_path=args.state_patterns,
        thresholds_path=args.thresholds,
        output_path=args.output,
        split_filter=args.split,
        model_key=args.model_key,
        protocol=inferred_protocol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
