#!/usr/bin/env python3
"""Draw the TAFFC Figure 8 representation/sensitivity template."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=np.load(ROOT/"taffc_mock_data.npz",allow_pickle=True)
OUT=ROOT/"outputs"/"figure8_representation_quality.png"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family":"serif",
    "font.serif":["Times New Roman","Times","Liberation Serif","DejaVu Serif"],
    "mathtext.fontset":"stix",
    "font.size":11.5,
    "axes.titlesize":17,
    "axes.labelsize":14,
    "xtick.labelsize":10.5,
    "ytick.labelsize":10.5,
    "axes.linewidth":1.0,
    "figure.facecolor":"white",
})

names=[str(x) for x in DATA["method_names"]]
xy=DATA["target_xy"]; y=DATA["y_target"].astype(int)
fractions=DATA["fractions"]; in_curves=DATA["in_curves"]; cross_curves=DATA["cross_curves"]
sil=DATA["silhouette"]; purity=DATA["knn_purity"]; cba=DATA["cross_bal_acc"]
BLUE="#154BFF"; RED="#FF2020"; GRID="#D4D4D4"

fig=plt.figure(figsize=(14.48,10.86),dpi=100)
fig.suptitle("Figure 8. Frozen representation quality and sensitivity to Conflict supervision",
             fontsize=22,fontweight="bold",y=.978)
gs=fig.add_gridspec(2,3,left=.05,right=.965,bottom=.135,top=.88,wspace=.28,hspace=.50,height_ratios=[1.04,.92])
letters=["(a)","(b)","(c)","(d)","(e)","(f)"]

for m in range(3):
    ax=fig.add_subplot(gs[0,m])
    display_xy=xy[m].copy()
    separation=[0.0,0.32,0.72][m]
    display_xy[:,0]+=np.where(y==1,separation,-separation)
    ax.scatter(display_xy[y==0,0],display_xy[y==0,1],s=13,c=BLUE,alpha=.80,edgecolor="#0C2D8F",linewidth=.25,label="Non-misread",rasterized=True)
    ax.scatter(display_xy[y==1,0],display_xy[y==1,1],s=13,c=RED,alpha=.78,edgecolor="#9E1111",linewidth=.25,label="Misread",rasterized=True)
    ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_xticks(np.arange(-6,7,2)); ax.set_yticks(np.arange(-6,7,2))
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(names[m],fontweight="bold",pad=11)
    ax.grid(color=GRID,ls=(0,(2,3)),lw=.7,alpha=.75); ax.set_axisbelow(True)
    ax.text(.97,.95,f"Silhouette = {sil[m]:.2f}\n5-NN purity = {purity[m]:.2f}\nCross-domain Bal. Acc. = {cba[m]:.2f}",
            transform=ax.transAxes,ha="right",va="top",fontsize=9.5)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.145),ncol=2,frameon=True,fontsize=10.5,
              handletextpad=.45,columnspacing=1.7)
    ax.text(-.16,1.10,letters[m],transform=ax.transAxes,fontsize=16,fontweight="bold")

fig.text(.5,.470,"Representations are from Conflict-only target-domain test set (no aligned data). Colors indicate Misread status.",
         ha="center",va="center",fontsize=12,fontstyle="italic",
         bbox=dict(boxstyle="round,pad=.35",fc="#F7F7F7",ec="#A8A8A8",lw=.8))

for m in range(3):
    ax=fig.add_subplot(gs[1,m])
    ax.plot(fractions,in_curves[m],color=BLUE,marker="o",lw=1.8,ms=5,label="In-domain (solid)")
    ax.plot(fractions,cross_curves[m],color=RED,marker="o",lw=1.6,ms=5,ls="--",label="Cross-domain (dashed)")
    for x0,y0 in zip(fractions,in_curves[m]):
        ax.text(x0,y0+.022,f"{y0:.2f}",ha="center",color=BLUE,fontsize=10.5)
    for x0,y0 in zip(fractions,cross_curves[m]):
        ax.text(x0,y0-.040,f"{y0:.2f}",ha="center",color=RED,fontsize=10.5)
    ax.set_xlim(4,106); ax.set_ylim(.30,1.00); ax.set_xticks(fractions); ax.set_yticks(np.arange(.3,1.01,.1))
    ax.set_xlabel("Conflict supervision retained (%)"); ax.set_ylabel("Balanced Accuracy")
    ax.set_title(names[m],fontweight="bold",pad=10)
    ax.grid(axis="y",color=GRID,ls=(0,(2,3)),lw=.7,alpha=.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right",frameon=True,fontsize=9.5)
    ax.text(-.16,1.08,letters[3+m],transform=ax.transAxes,fontsize=16,fontweight="bold")

fig.text(.5,.072,"Probe: Conflict-only Misread probe trained on frozen sample-level representations.",
         ha="center",va="center",fontsize=12,fontstyle="italic",
         bbox=dict(boxstyle="round,pad=.35",fc="#F7F7F7",ec="#A8A8A8",lw=.8))
fig.text(.5,.025,"Illustrative mock data for layout planning only — do not use in the paper.",
         ha="center",fontsize=12.5,fontstyle="italic",color="#555555")
fig.savefig(OUT,dpi=100,facecolor="white")
print(OUT)
