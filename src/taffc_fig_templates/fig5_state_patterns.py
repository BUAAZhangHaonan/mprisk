#!/usr/bin/env python3
"""Draw the TAFFC Figure 5 layout using shared synthetic mock data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
DATA = np.load(ROOT / "taffc_mock_data.npz", allow_pickle=True)
OUT = ROOT / "outputs" / "figure5_state_patterns.png"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.25,
    "figure.facecolor": "white",
})

COLORS = {
    "Consensus":"#2F5597", "Balanced":"#F2B38C", "Dominant":"#C95D61", "Confusion":"#D2D2D2",
    "Non-misread":"#7185A4", "Misread":"#C95D61"
}
relation = DATA["relation"]
state = DATA["state"]
misread = DATA["misread"]
state_names = [str(x) for x in DATA["state_names"]]

fig = plt.figure(figsize=(14.48,10.86), dpi=100)
fig.suptitle("Figure 5. Four-state distributions across input relations and Misread outcomes",
             fontsize=24, fontweight="bold", y=0.976)
gs = fig.add_gridspec(1,2, width_ratios=[1.0,1.18], left=0.075, right=0.97,
                      bottom=0.215, top=0.82, wspace=0.30)

# (a) State composition by input relation
ax1 = fig.add_subplot(gs[0,0])
order = [0,1,2,3]  # Consensus, Balanced, Dominant, Confusion
x = np.array([0,1.05]); width=0.70
bottom = np.zeros(2)
rel_totals = np.array([(relation==0).sum(), (relation==1).sum()])
for st in order:
    counts = np.array([((relation==0)&(state==st)).sum(), ((relation==1)&(state==st)).sum()])
    pct = counts/rel_totals*100
    ax1.bar(x, pct, width, bottom=bottom, color=COLORS[state_names[st]], edgecolor="#333333", linewidth=.9)
    for k in range(2):
        txt_color = "white" if st in (0,2) else "black"
        ax1.text(x[k], bottom[k]+pct[k]/2, f"{pct[k]:.0f}%\n({counts[k]:,})", ha="center", va="center",
                 fontsize=15, color=txt_color)
    bottom += pct
ax1.set_ylim(0,100); ax1.set_xlim(-0.50,1.55)
ax1.set_xticks(x, [f"Aligned\n($n={rel_totals[0]:,}$)", f"Conflict\n($n={rel_totals[1]:,}$)"])
ax1.set_yticks(np.arange(0,101,20)); ax1.set_ylabel("Proportion of samples (%)", labelpad=10)
ax1.yaxis.grid(True, linestyle=(0,(2,3)), color="#C9C9C9", alpha=.8); ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.set_title("State-pattern composition in\nAligned and Conflict samples", pad=26, fontweight="bold")
ax1.text(-0.16,1.13,"(a)",transform=ax1.transAxes,fontsize=22,fontweight="bold")

# (b) Misread composition within each state on Conflict only
ax2 = fig.add_subplot(gs[0,1])
right_order = [3,0,1,2]  # Confusion, Consensus, Balanced, Dominant
rx = np.arange(4); rwidth=.69
for k,st in enumerate(right_order):
    idx = (relation==1)&(state==st)
    n = idx.sum(); n_m = int(misread[idx].sum()); n_n = int(n-n_m)
    p_n = n_n/n*100; p_m=n_m/n*100
    ax2.bar(k,p_n,rwidth,color=COLORS["Non-misread"],edgecolor="#333333",linewidth=.9)
    ax2.bar(k,p_m,rwidth,bottom=p_n,color=COLORS["Misread"],edgecolor="#333333",linewidth=.9)
    ax2.text(k,p_n/2,f"{p_n:.0f}%\n({n_n:,})",ha="center",va="center",fontsize=14.5,color="white")
    ax2.text(k,p_n+p_m/2,f"{p_m:.0f}%\n({n_m:,})",ha="center",va="center",fontsize=14.5,color="white")
ax2.set_ylim(0,100); ax2.set_xlim(-.55,3.55)
labels=[]
for st in right_order:
    n=((relation==1)&(state==st)).sum(); labels.append(f"{state_names[st]}\n($n={n:,}$)")
ax2.set_xticks(rx,labels); ax2.set_yticks(np.arange(0,101,20))
ax2.set_ylabel("Proportion of Conflict samples (%)",labelpad=10)
ax2.yaxis.grid(True,linestyle=(0,(2,3)),color="#C9C9C9",alpha=.8); ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_title("Misread composition within each\nstate pattern (Conflict only)",pad=26,fontweight="bold")
ax2.text(-0.16,1.13,"(b)",transform=ax2.transAxes,fontsize=22,fontweight="bold")

# Legends
legend_state = [Patch(facecolor=COLORS[n],edgecolor="#333333",label=n) for n in ["Consensus","Balanced","Dominant","Confusion"]]
ax1.legend(handles=legend_state, loc="upper center", bbox_to_anchor=(0.50,-0.13), ncol=4,
           frameon=False, fontsize=14, handlelength=1.5, columnspacing=1.2)
legend_out = [Patch(facecolor=COLORS[n],edgecolor="#333333",label=n) for n in ["Non-misread","Misread"]]
ax2.legend(handles=legend_out, loc="upper center", bbox_to_anchor=(0.50,-0.13), ncol=2,
           frameon=False, fontsize=15, handlelength=1.6, columnspacing=3.0)
fig.text(0.5,0.045,"Illustrative mock data for layout planning only — do not use in the paper.",
         ha="center",fontsize=14,fontstyle="italic",color="#555555")
fig.savefig(OUT,dpi=100,facecolor="white")
print(OUT)
