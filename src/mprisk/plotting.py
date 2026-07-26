"""Polished plotting for Fig.4, Fig.5, Fig.6.

Reuses mprisk state-pattern semantics but renders with a single seed,
tuned thresholds, and presentation-grade visuals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

CONFLICT_COLOR = "#d1495b"
ALIGNED_COLOR = "#2e4057"
PATTERN_COLORS = {
    "Confusion":  "#d1495b",
    "Consensus":  "#2e4057",
    "Balanced":   "#66a182",
    "Dominant":   "#e6b800",
}
PATTERN_ORDER = ["Confusion", "Consensus", "Balanced", "Dominant"]

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)


def load_state_patterns(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            split = (
                r.get("representation_split")
                or r.get("calibration_split")
                or r.get("master_split")
                or "Unknown"
            )
            rows.append({
                "sample_id": r.get("sample_id"),
                "sample_type": r.get("sample_type", "Unknown"),
                "split": split,
                "S": float(r.get("S_mean", float("nan"))),
                "D": float(r.get("D", float("nan"))),
                "R": float(r.get("R", float("nan"))),
                "abs_R": abs(float(r.get("R", 0.0))),
                "delta_i": float(r.get("delta_i", 0.0)),
                "lean": r.get("lean", "Balanced"),
                "pattern": r.get("pattern", "Unknown"),
                "kappa": float(r.get("kappa", float("nan"))) if "kappa" in r else float("nan"),
                "tau": float(r.get("tau", float("nan"))) if "tau" in r else float("nan"),
            })
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def plot_fig04_sdr_distributions(
    dfs: dict[str, pd.DataFrame],
    out_path: str | Path,
    *,
    split_filter: str = "official_test",
    title_suffix: str = "",
) -> Path:
    """Per-model S / D / |R| distributions, Conflict vs Aligned."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_keys = list(dfs.keys())
    n_models = len(model_keys)
    fig, axes = plt.subplots(
        nrows=3, ncols=n_models,
        figsize=(3.2 * max(n_models, 1), 6.5),
        sharex="col",
    )
    if n_models == 1:
        axes = axes.reshape(3, 1)
    metrics = [("S", "State Dispersion  $S$"),
               ("D", "Modality Split  $\\mathcal{D}$"),
               ("abs_R", "Joint Lean  $|\\mathcal{R}|$")]
    for col, mk in enumerate(model_keys):
        df = dfs[mk]
        if split_filter:
            df = df[df["split"] == split_filter]
        df = df.dropna(subset=["S", "D", "abs_R"])
        for row, (field, label) in enumerate(metrics):
            ax = axes[row, col]
            for stype, color in [("Aligned", ALIGNED_COLOR),
                                 ("Conflict", CONFLICT_COLOR)]:
                vals = df[df["sample_type"] == stype][field].values
                if len(vals) < 2:
                    continue
                sns.kdeplot(
                    vals, ax=ax,
                    color=color, fill=True, alpha=0.35, linewidth=1.4,
                    bw_adjust=0.9, cut=0,
                    label=f"{stype} (n={len(vals)})",
                )
            ax.set_ylabel(label if col == 0 else "")
            ax.set_xlabel("")
            if row == 0:
                ax.set_title(mk, fontsize=10, weight="bold")
            if row == 2:
                ax.set_xlabel("density")
            if col == 0 and row == 0:
                ax.legend(frameon=False, loc="upper right")
    fig.suptitle(
        f"Per-model pre-generation state distributions ({split_filter}){title_suffix}",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig05_pattern_stacks(
    dfs: dict[str, pd.DataFrame],
    out_path: str | Path,
    *,
    split_filter: str = "official_test",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_keys = list(dfs.keys())
    fig, axes = plt.subplots(
        nrows=1, ncols=len(model_keys),
        figsize=(3.0 * len(model_keys), 4.0),
        sharey=True,
    )
    if len(model_keys) == 1:
        axes = [axes]
    for ax, mk in zip(axes, model_keys):
        df = dfs[mk]
        if split_filter:
            df = df[df["split"] == split_filter]
        ct = (
            df.groupby(["sample_type", "pattern"]).size()
            .unstack(fill_value=0)
        )
        for p in PATTERN_ORDER:
            if p not in ct.columns:
                ct[p] = 0
        ct = ct[PATTERN_ORDER]
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
        order = ["Aligned", "Conflict"]
        order = [s for s in order if s in ct_pct.index]
        left = np.zeros(len(order))
        for p in PATTERN_ORDER:
            vals = ct_pct.loc[order, p].values
            ax.barh(order, vals, left=left, color=PATTERN_COLORS[p],
                    edgecolor="white", linewidth=0.5, label=p)
            for i, v in enumerate(vals):
                if v > 6:
                    ax.text(left[i] + v / 2, i, f"{v:.0f}",
                            ha="center", va="center", fontsize=8,
                            color="white", weight="bold")
            left += vals
        ax.set_xlim(0, 100)
        ax.set_xlabel("Proportion (%)")
        ax.set_title(mk, fontsize=10, weight="bold")
        if mk == model_keys[0]:
            ax.set_ylabel("Sample type")
        if mk == model_keys[-1]:
            ax.legend(
                title="State Pattern",
                bbox_to_anchor=(1.02, 1.0), loc="upper left",
                frameon=False,
            )
    fig.suptitle(
        f"Four State Pattern proportions ({split_filter})",
        y=0.98, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig06_stable_d_r(
    dfs: dict[str, pd.DataFrame],
    thresholds: dict[str, dict[str, float]],
    out_path: str | Path,
    *,
    split_filter: str = "official_test",
) -> Path:
    """D vs signed R scatter, stable samples only (S <= kappa)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_keys = list(dfs.keys())
    fig, axes = plt.subplots(
        nrows=1, ncols=len(model_keys),
        figsize=(3.6 * len(model_keys), 3.8),
        sharey=False,
    )
    if len(model_keys) == 1:
        axes = [axes]
    for ax, mk in zip(axes, model_keys):
        df = dfs[mk]
        if split_filter:
            df = df[df["split"] == split_filter]
        kappa = float(thresholds[mk].get("kappa", np.nan))
        tau = float(thresholds[mk].get("tau", np.nan))
        stable = df[(df["S"] <= kappa) & df["D"].notna() & df["R"].notna()].copy()
        for stype, color in [("Aligned", ALIGNED_COLOR),
                             ("Conflict", CONFLICT_COLOR)]:
            sub = stable[stable["sample_type"] == stype]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["D"], sub["R"],
                s=22, alpha=0.65, color=color, edgecolor="white", linewidth=0.3,
                label=f"{stype} (n={len(sub)})",
            )
        if not np.isnan(tau):
            ax.axvline(tau, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(tau, ax.get_ylim()[1] * 0.95, " $\\tau$", fontsize=8, color="gray")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Modality Split  $\\mathcal{D}$")
        if mk == model_keys[0]:
            ax.set_ylabel("Joint Lean  $\\mathcal{R}$\n(<0 T/A-lean, >0 V-lean)")
        ax.set_title(mk, fontsize=10, weight="bold")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle(
        f"Stable-sample $\\mathcal{{D}}$-signed-$\\mathcal{{R}}$ geometry "
        f"($S\\leq\\kappa$, {split_filter})",
        y=1.00, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
