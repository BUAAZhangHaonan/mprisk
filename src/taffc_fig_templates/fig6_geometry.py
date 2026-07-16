#!/usr/bin/env python3
"""Draw the TAFFC Figure 6 geometry template from shared mock data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent
DATA = np.load(ROOT / "taffc_mock_data.npz", allow_pickle=True)
OUT = ROOT / "outputs" / "figure6_geometry.png"
OUT.parent.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family":"serif",
    "font.serif":["Times New Roman","Times","Liberation Serif","DejaVu Serif"],
    "mathtext.fontset":"stix",
    "font.size":14,
    "axes.labelsize":21,
    "xtick.labelsize":15,
    "ytick.labelsize":15,
    "axes.linewidth":1.25,
    "figure.facecolor":"white",
})

relation=DATA["relation"]; state=DATA["state"]
Dg=DATA["D_geom"]; Rg=DATA["R_geom"]
tau=float(DATA["tau"]); delta=float(DATA["delta"])
idx=np.flatnonzero((relation==1)&(state!=3))

cmap=LinearSegmentedColormap.from_list("taffc_div",["#2B5A9B","#F8F8F8","#C72F3A"],N=256)
norm=Normalize(-1.5,1.5)

fig=plt.figure(figsize=(14.48,10.86),dpi=100)
fig.suptitle("Figure 6. Geometric interpretation of state patterns in stable Conflict samples",
             fontsize=24,fontweight="bold",y=.975)
ax=fig.add_axes([.087,.17,.73,.72])

# Repeat each shared observation with tiny deterministic jitter to mimic the dense
# model-sample cloud of the reference layout without changing the underlying data.
rng=np.random.default_rng(606)
xplot=np.repeat(Dg[idx],3)+rng.normal(0,.007,len(idx)*3)
yplot=np.repeat(Rg[idx],3)+rng.normal(0,.045,len(idx)*3)
xplot=np.clip(xplot,0,.82); yplot=np.clip(yplot,-1.5,1.5)
sc=ax.scatter(xplot,yplot,c=yplot,cmap=cmap,norm=norm,s=17,alpha=.25,linewidths=0,rasterized=True)
xx=np.linspace(0,.82,300)
yy=.30+.38*np.tanh(5.2*(xx-.38))
ax.plot(xx,yy,color="#172F62",lw=3.0,zorder=4)
ax.axvline(tau,color="#8E8E8E",lw=1.5,ls=(0,(5,4)))
ax.axhline(delta,color="#A33A40",lw=1.7,ls=(0,(1,2)))
ax.axhline(-delta,color="#2E58A6",lw=1.7,ls=(0,(1,2)))
ax.text(tau+.012,1.41,rf"$\tau={tau:.2f}$",fontsize=17)
ax.text(.735,delta+.015,rf"$+\delta={delta:.2f}$",color="#A92836",fontsize=16)
ax.text(.735,-delta-.035,rf"$-\delta=-{delta:.2f}$",color="#1F4EAA",fontsize=16)

ax.set_xlim(0,.83); ax.set_ylim(-1.5,1.5)
ax.set_xticks(np.arange(0,.81,.1)); ax.set_yticks(np.arange(-1.5,1.51,.5))
ax.set_xlabel(r"Split Significance $\mathcal{D}$")
ax.set_ylabel(r"Signed Joint Lean $\mathcal{R}$",labelpad=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=6,width=1.1)

# Region labels
ax.text(.065,1.25,"Consensus",fontsize=20,fontweight="bold")
ax.text(.59,1.39,"Dominant",fontsize=20,fontweight="bold")
ax.text(.595,.22,"Balanced",fontsize=20,fontweight="bold")
ax.text(.555,-.94,"Dominant",fontsize=20,fontweight="bold")
ax.text(.73,-1.39,"Confusion\nfiltered out",fontsize=16,fontstyle="italic",color="#888888",ha="center")

# Small condition-state schematics.
def inset_nodes(bounds, coords, label=None):
    ia=ax.inset_axes(bounds)
    ia.set_xlim(0,1); ia.set_ylim(0,1); ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values(): s.set_color("#A6A6A6"); s.set_linewidth(1.0)
    pts=np.asarray(coords)
    for a,b in [(0,1),(1,2)]:
        ia.plot([pts[a,0],pts[b,0]],[pts[a,1],pts[b,1]],color="#8C8C8C",ls=(0,(4,3)),lw=1.4,zorder=1)
    cols=["#2F62AD","#F6D1B2","#D62F2F"]
    ia.scatter(pts[:,0],pts[:,1],s=155,c=cols,edgecolor="#222222",linewidth=1.0,zorder=3)
    return ia

inset_nodes([.075,.765,.13,.13],[(.17,.20),(.50,.52),(.83,.82)])
inset_nodes([.665,.805,.13,.14],[(.22,.16),(.82,.48),(.23,.82)])
inset_nodes([.65,.47,.16,.08],[(.15,.50),(.50,.50),(.85,.50)])
inset_nodes([.615,.045,.14,.13],[(.22,.18),(.80,.47),(.10,.80)])

# Colorbar and semantic endpoints
cax=fig.add_axes([.845,.28,.022,.57])
cbar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax)
cbar.set_ticks([1.35,0,-1.35]); cbar.set_ticklabels([r"$M_1$-dominant","Neutral",r"$M_2$-dominant"])
cbar.ax.tick_params(labelsize=16,pad=7)

fig.text(.50,.082,r"$M_1=$ Visual,  $M_2=$ Text/Audio",ha="center",fontsize=16,fontstyle="italic")
fig.text(.50,.038,"Illustrative mock data for layout planning only — do not use in the paper.",
         ha="center",fontsize=14,fontstyle="italic",color="#555555")
fig.savefig(OUT,dpi=100,facecolor="white")
print(OUT)
