#!/usr/bin/env python3
"""Draw the TAFFC Figure 4 layout using shared synthetic mock data.

WARNING: The data are fabricated for layout planning only. Never use the
rendered values as empirical results in a manuscript or supplement.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
DATA = np.load(ROOT / "taffc_mock_data.npz", allow_pickle=True)
OUT = ROOT / "outputs" / "figure4_state_indices.png"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#6F8DB7"
BLUE_DARK = "#58749C"
RED = "#D27068"
RED_DARK = "#B75752"
GRID = "#CFCFCF"

rng = np.random.default_rng(417)
relation = DATA["relation"]
S = DATA["S"]
D = DATA["D"]
R = DATA["R"]

panels = [
    dict(letter="A", title=r"State Dispersion ($S$)", idx=DATA["fig4_S_idx"], values=S,
         ylabel=r"$S$  (larger $S$ = more dispersed or less stable)", ylim=(0,1.6), yticks=np.arange(0,1.61,.2),
         comp="Aligned 48% ($n=412$),  Conflict 52% ($n=446$)"),
    dict(letter="B", title=r"Modality Split ($\mathcal{D}$)", idx=DATA["fig4_D_idx"], values=D,
         ylabel=r"$\mathcal{D}$  (larger $\mathcal{D}$ = stronger split)", ylim=(0,1.6), yticks=np.arange(0,1.61,.2),
         comp="Aligned 43% ($n=381$),  Conflict 57% ($n=509$)"),
    dict(letter="C", title=r"Joint Lean Magnitude ($|\mathcal{R}|$)", idx=DATA["fig4_R_idx"], values=0.34 + 1.05*np.abs(R),
         ylabel=r"$|\mathcal{R}|$  (larger $|\mathcal{R}|$ = stronger joint lean)", ylim=(0,3.0), yticks=np.arange(0,3.01,.5),
         comp="Aligned 47% ($n=398$),  Conflict 53% ($n=454$)"),
]

fig = plt.figure(figsize=(14.48,10.86), dpi=100)
fig.suptitle("Figure 4. Pre-generation state indices in Conflict and Aligned samples",
             fontsize=24, fontweight="bold", y=0.976)
gs = fig.add_gridspec(1,3, left=0.072, right=0.972, bottom=0.205, top=0.815, wspace=0.29)

for j, spec in enumerate(panels):
    ax = fig.add_subplot(gs[0,j])
    idx = spec["idx"].astype(int)
    vals = spec["values"][idx]
    groups = [vals[relation[idx] == 0], vals[relation[idx] == 1]]

    # Violin bodies
    vp = ax.violinplot(groups, positions=[1,2], widths=0.72, showmeans=False,
                       showmedians=False, showextrema=False, bw_method=0.38)
    for body, fc, ec in zip(vp["bodies"], [BLUE, RED], [BLUE_DARK, RED_DARK]):
        body.set_facecolor(fc); body.set_edgecolor(ec); body.set_alpha(0.20); body.set_linewidth(1.2)

    # Jittered observations (subsample only for visual clarity; n labels refer to all observations).
    for pos, arr, col in zip([1,2], groups, [BLUE_DARK, RED_DARK]):
        take = rng.choice(len(arr), min(230, len(arr)), replace=False)
        jitter = rng.normal(0, 0.055, len(take))
        ax.scatter(pos+jitter, arr[take], s=11, color=col, alpha=0.67, linewidths=0, zorder=3)

    bp = ax.boxplot(groups, positions=[1,2], widths=0.23, patch_artist=True,
                    showmeans=True, showfliers=False,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=8),
                    medianprops=dict(color="black", linewidth=1.35),
                    whiskerprops=dict(color="black", linewidth=1.2),
                    capprops=dict(color="black", linewidth=1.2))
    for patch, fc in zip(bp["boxes"], [BLUE_DARK, RED_DARK]):
        patch.set_facecolor(fc); patch.set_alpha(0.78); patch.set_edgecolor("black"); patch.set_linewidth(1.15)

    ax.set_xlim(0.55,2.45); ax.set_ylim(*spec["ylim"]); ax.set_yticks(spec["yticks"])
    ax.set_xticks([1,2], ["Aligned","Conflict"])
    ax.set_ylabel(spec["ylabel"], labelpad=10)
    ax.yaxis.grid(True, linestyle=(0,(2,3)), color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=5, width=1.1)

    ax.set_title(spec["title"], pad=62, fontweight="bold")
    ax.text(-0.14, 1.145, spec["letter"], transform=ax.transAxes, fontsize=25, fontweight="bold")
    ax.text(0.50, 1.076, spec["comp"], transform=ax.transAxes, ha="center", va="center",
            fontsize=13.2, fontstyle="italic")

    diff = float(np.mean(groups[1]) - np.mean(groups[0]))
    # Deterministic bootstrap CI for the displayed difference.
    brng = np.random.default_rng(100+j)
    boots = []
    for _ in range(1200):
        a = brng.choice(groups[0], len(groups[0]), replace=True)
        c = brng.choice(groups[1], len(groups[1]), replace=True)
        boots.append(np.mean(c)-np.mean(a))
    lo, hi = np.percentile(boots, [2.5,97.5])
    y0, y1 = spec["ylim"]
    yr = y1-y0
    bracket_y = y1 - 0.075*yr
    h = 0.020*yr
    ax.plot([1.12,1.12,1.88,1.88], [bracket_y-h,bracket_y,bracket_y,bracket_y-h], color="black", lw=1.25)
    ax.text(1.50, bracket_y+0.010*yr, "***", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.text(1.50, bracket_y+0.060*yr, rf"$\Delta={diff:.2f}$", ha="center", va="bottom", fontsize=15)
    ax.text(0.98, 0.025, rf"95% CI [{lo:.2f}, {hi:.2f}]", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12.3, fontstyle="italic", color="#444444")

fig.legend(handles=[Patch(facecolor=BLUE_DARK, edgecolor="black", alpha=.80, label="Aligned"),
                    Patch(facecolor=RED_DARK, edgecolor="black", alpha=.80, label="Conflict")],
           loc="lower center", bbox_to_anchor=(0.385,0.086), frameon=False, ncol=2,
           fontsize=16, handlelength=2.0, columnspacing=2.6)
fig.text(0.67,0.105, r"*  $p<0.05$      **  $p<0.01$      ***  $p<0.001$",
         ha="center", va="center", fontsize=14)
fig.text(0.5,0.045, "Illustrative mock data for layout planning only — do not use in the paper.",
         ha="center", va="center", fontsize=14, fontstyle="italic", color="#555555")

fig.savefig(OUT, dpi=100, facecolor="white")
print(OUT)
