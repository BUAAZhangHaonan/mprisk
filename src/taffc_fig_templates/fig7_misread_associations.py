#!/usr/bin/env python3
"""Draw the TAFFC Figure 7 association template from shared mock data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT=Path(__file__).resolve().parent
DATA=np.load(ROOT/"taffc_mock_data.npz",allow_pickle=True)
OUT=ROOT/"outputs"/"figure7_misread_associations.png"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family":"serif",
    "font.serif":["Times New Roman","Times","Liberation Serif","DejaVu Serif"],
    "mathtext.fontset":"stix",
    "font.size":11.5,
    "axes.titlesize":15.5,
    "axes.labelsize":14,
    "xtick.labelsize":11,
    "ytick.labelsize":11,
    "axes.linewidth":1.0,
    "figure.facecolor":"white",
})

names=[str(x) for x in DATA["model_names"]]
protocols=[str(x) for x in DATA["protocols"]]
SN=DATA["m_sn"]; DN=DATA["m_dn"]; RN=DATA["m_rn"]
MIS=DATA["m_mis"]; PROB=DATA["m_prob"]
RATES=DATA["fig7_rates"]; RATE_ERR=DATA["fig7_errors"]
MODEL_COLORS=["#D7191C","#F27A0A","#1F4E99"]
cmap=LinearSegmentedColormap.from_list("taffc_div",["#2455A4","#F8F8F8","#E52B2B"],256)
norm=Normalize(-1.5,1.5)

fig=plt.figure(figsize=(14.48,10.86),dpi=100)
fig.suptitle("Figure 7. State indices and modality bias are associated with Misread in Conflict samples",
             fontsize=22,fontweight="bold",y=.978)
gs=fig.add_gridspec(2,3,left=.047,right=.965,bottom=.13,top=.885,wspace=.28,hspace=.40,height_ratios=[1.02,1.0])

features=[SN,DN,np.abs(RN)]
titles=["Dispersion vs Misread Rate","Modality Split vs Misread Rate","Joint Lean Magnitude vs Misread Rate"]
xlabels=[r"Normalized state dispersion  $S/\kappa$",r"Normalized modality split  $\mathcal{D}/\tau$",r"Normalized joint lean magnitude  $|\mathcal{R}|/\delta$"]
threshold_notes=["dispersion\nthreshold","split\nthreshold","lean magnitude\nthreshold"]
centers=np.arange(.4,1.81,.2); edges=np.arange(.3,2.01,.2)
letters=["(a)","(b)","(c)","(d)","(e)","(f)"]

for j in range(3):
    ax=fig.add_subplot(gs[0,j])
    for m in range(3):
        ys=[]; es=[]
        for b in range(len(centers)):
            sel=(features[j][m]>=edges[b])&(features[j][m]<edges[b+1])
            if sel.sum()<5:
                ys.append(np.nan); es.append(np.nan)
            else:
                # Expected mock Misread rate from the same model-sample observations.
                p=float(PROB[m,sel].mean())
                ys.append(100*p)
                es.append(100*np.sqrt(max(p*(1-p),1e-6)/sel.sum()))
        # Rates and confidence-width placeholders are stored in the shared synthetic data file.
        # They are consistent across reruns and deliberately mimic the intended final layout.
        ys=RATES[j,m].astype(float)
        es=RATE_ERR[j,m].astype(float)
        ax.errorbar(centers,ys,yerr=es,color=MODEL_COLORS[m],marker="o",markersize=4.5,
                    lw=1.6,capsize=2.5,label=names[m])
        for x,y in zip(centers,ys):
            ax.text(x,y+1.35,f"{int(round(y))}",ha="center",va="bottom",fontsize=9.2,color=MODEL_COLORS[m])
    ax.axvline(1.0,color="#888888",ls=(0,(5,4)),lw=1.1)
    ax.set_xlim(.32,1.87); ax.set_ylim(0,60); ax.set_yticks(np.arange(0,61,10))
    ax.set_xticks(centers); ax.set_xlabel(xlabels[j]); ax.set_ylabel("Misread Rate (%)")
    ax.set_title(titles[j],fontweight="bold",pad=10)
    ax.grid(axis="y",color="#D6D6D6",ls=(0,(2,3)),lw=.7,alpha=.75); ax.set_axisbelow(True)
    ax.legend(loc="upper left",frameon=False,fontsize=9.3,handlelength=2.0)
    ax.text(.90,.05,threshold_notes[j],transform=ax.transAxes,ha="right",va="bottom",
            fontsize=9.6,fontstyle="italic",color="#555555")
    ax.text(-.16,1.11,letters[j],transform=ax.transAxes,fontsize=16,fontweight="bold")

for m in range(3):
    ax=fig.add_subplot(gs[1,m])
    non=MIS[m]==0; mis=MIS[m]==1
    ax.scatter(DN[m,non],RN[m,non],facecolors="none",edgecolors=cmap(norm(RN[m,non])),s=10,marker="o",
               linewidths=.55,alpha=.62,label="Non-misread",rasterized=True)
    ax.scatter(DN[m,mis],RN[m,mis],facecolors="none",edgecolors=cmap(norm(RN[m,mis])),s=13,marker="^",
               linewidths=.65,alpha=.80,label="Misread",rasterized=True)
    ax.axvline(1.0,color="#888888",ls=(0,(5,4)),lw=1.0)
    ax.axhline(0,color="#8F8F8F",ls=(0,(2,3)),lw=.9)
    ax.set_xlim(0,2.0); ax.set_ylim(-1.5,1.5)
    ax.set_xticks([0,.5,1,1.5,2]); ax.set_yticks(np.arange(-1.5,1.51,.5))
    ax.set_xlabel(r"Normalized split significance  $\mathcal{D}/\tau$")
    ax.set_ylabel(r"Signed joint lean  $\mathcal{R}/\delta$" if m==0 else "")
    ax.set_title(f"{names[m]}  ({protocols[m]})",fontweight="bold",pad=10)
    rho=float(spearmanr(DN[m],np.abs(RN[m])).statistic)
    dom=(MIS[m]==1)&(DN[m]>1.0)&(np.abs(RN[m])>1.0)
    if dom.sum():
        pos=(RN[m,dom]>0).mean(); follow=max(pos,1-pos)*100
    else: follow=np.nan
    ax.text(.03,.96,rf"$\rho={rho:.2f}$"+f"\nMisreads following\ndominant modality: {follow:.0f}%",
            transform=ax.transAxes,va="top",fontsize=9.5)
    ax.text(.04,.67,"Consensus",transform=ax.transAxes,color="#1E4E9B",fontstyle="italic",fontsize=10)
    ax.text(.84,.90,"Dominant\n(visual)",transform=ax.transAxes,color="#D7191C",ha="center",fontstyle="italic",fontsize=9)
    ax.text(.80,.44,"Balanced",transform=ax.transAxes,color="#555555",fontstyle="italic",fontsize=9)
    lower="audio" if protocols[m]=="V-A" else "audio/text"
    ax.text(.86,.05,f"Dominant\n({lower})",transform=ax.transAxes,color="#1E4E9B",ha="center",fontstyle="italic",fontsize=9)
    ax.legend(loc="lower left",frameon=True,framealpha=.93,fontsize=8.6,handletextpad=.35,borderpad=.45)
    ax.text(-.16,1.08,letters[3+m],transform=ax.transAxes,fontsize=16,fontweight="bold")
    divider=make_axes_locatable(ax); cax=divider.append_axes("right",size="2.4%",pad=.05)
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    cb.set_ticks([-1.5,-1,-.5,0,.5,1,1.5]); cb.ax.tick_params(labelsize=8.5)
    if m==2:
        cb.set_label("Signed direction (visual +, audio/text −)",fontsize=9.2,labelpad=4)

# Shared legends / footnote
handles=[plt.Line2D([0],[0],color=c,marker="o",lw=1.8,label=n) for c,n in zip(MODEL_COLORS,names)]
fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.365,.061),ncol=3,frameon=True,
           fontsize=10.5,columnspacing=2.0,handlelength=2.2)
shape_handles=[plt.Line2D([0],[0],marker="o",color="black",markerfacecolor="none",ls="",label="Non-misread"),
               plt.Line2D([0],[0],marker="^",color="black",markerfacecolor="none",ls="",label="Misread"),
               plt.Line2D([0],[0],color="#666666",ls=(0,(5,4)),label="Threshold (at 1.0)")]
fig.legend(handles=shape_handles,loc="lower center",bbox_to_anchor=(.775,.061),ncol=3,frameon=True,
           fontsize=10.5,columnspacing=1.6,handlelength=2.2)
fig.text(.5,.024,"Illustrative mock data for layout planning only — do not use in the paper.",
         ha="center",fontsize=12.5,fontstyle="italic",color="#555555")
fig.savefig(OUT,dpi=100,facecolor="white")
print(OUT)
