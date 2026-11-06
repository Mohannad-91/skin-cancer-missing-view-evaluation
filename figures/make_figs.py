import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
MAL=np.array([0,1,6,7,9]); C={"head":"#1f5aa6","margin":"#d9822b","entropy":"#7a7a7a","random":"#bbbbbb"}
# ---------- Figure 1: architecture ----------
fig,ax=plt.subplots(figsize=(7.4,3.9)); ax.set_xlim(0,124); ax.set_ylim(-2,51); ax.axis("off")
def box(x,y,w,h,t,fc="#eef3fa",ec="#1f5aa6",fs=7.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(x1,y1,x2,y2,c="#444",ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,color=c,lw=1,linestyle=ls))
box(0,40,17,9,"Clinical\nclose-up",fc="#fff",ec="#555"); box(0,24,17,9,"Dermoscopic\nimage",fc="#fff",ec="#555"); box(0,6,17,10,"Metadata\nage, sex,\nsite, tone",fc="#fff",ec="#555")
box(22,40,20,9,"Encoder C\nConvNeXt-T or\nPanDerm-B",fs=7); box(22,24,20,9,"Encoder D\nConvNeXt-T or\nPanDerm-B",fs=7); box(22,6,20,10,"Metadata MLP\n64-d; hidden with\np = 0.2 in training",fs=7)
for y in (44.5,28.5,11): arr(17.6,y,21.4,y)
box(47,40,13,9,"z_C\n512-d",fc="#fff"); box(47,24,13,9,"z_D\n512-d",fc="#fff")
arr(42.6,44.5,46.4,44.5); arr(42.6,28.5,46.4,28.5)
box(47,34.2,13,4.6,"learned token\nif view missing",fc="#fdf1e4",ec="#d9822b",fs=5.6)
box(66,20,20,22,"Fusion head\n\ninputs: z_C, z_D,\nmetadata, 2 flags\n\nLN - 512 - 11",fs=7)
arr(60.6,44.5,65.4,38); arr(60.6,28.5,65.4,26); arr(42.6,11,65.4,22); arr(60.6,36.5,65.4,32,c="#d9822b",ls="--")
box(92,42,31,7,"p(both views)",fc="#e7f4e4",ec="#3a8a3a"); box(92,31.5,31,7,"p(clinical only)",fc="#e7f4e4",ec="#3a8a3a"); box(92,21,31,7,"p(dermoscopic only)",fc="#e7f4e4",ec="#3a8a3a")
for y in (45.5,35,24.5): arr(86.6,31,91.4,y)
box(66,2,20,10,"Referral head\n(z_C + metadata)",fc="#f2eefc",ec="#6a4fb3",fs=7); box(92,3.5,31,7,"send for dermoscopy?",fc="#f2eefc",ec="#6a4fb3")
ax.plot([44.5,44.5],[44.5,17],color="#6a4fb3",lw=1,ls=":"); arr(44.5,17,65.4,9,c="#6a4fb3",ls=":"); arr(86.6,7,91.4,7,c="#6a4fb3")
arr(42.6,8.5,65.4,6,c="#6a4fb3",ls=":")
fig.savefig("figures/Figure1.png",dpi=300,bbox_inches="tight"); plt.close(fig)
# ---------- Figure 2: referral curves ----------
def curves(z):
    y=z["y"]; ym=np.isin(y,MAL); n=len(y)
    pc,pb=z["T_c"],z["T_both"]; pmc=pc[:,MAL].sum(1); dc=pmc>=z["thr_c"]; db=pb[:,MAL].sum(1)>=z["thr_both"]
    ent=-(pc*np.log(np.clip(pc,1e-12,1))).sum(1)
    S={"head":z["ref"],"margin":-np.abs(pmc-0.5),"entropy":ent}
    qs=np.linspace(0,1,41); out={}
    for k,s in S.items():
        o=np.argsort(-s); r=[]
        for q in qs:
            m=np.zeros(n,bool); m[o[:int(round(q*n))]]=True; d=np.where(m,db,dc); r.append((d[ym].mean(),(~d[~ym]).mean()))
        out[k]=np.array(r)
    rng=np.random.default_rng(0); rr=[]
    for q in qs:
        a=[]
        for _ in range(20):
            m=np.zeros(n,bool); m[rng.permutation(n)[:int(round(q*n))]]=True; d=np.where(m,db,dc); a.append((d[ym].mean(),(~d[~ym]).mean()))
        rr.append(np.mean(a,0))
    out["random"]=np.array(rr); return qs,out
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),sharex=True)
lab={"head":"Learned referral head","margin":"Margin |p - 0.5|","entropy":"Entropy","random":"Random"}
for j,(r,name) in enumerate([("res1","ConvNeXt-Tiny"),("res2","PanDerm-Base")]):
    qs,o=curves(np.load(f"results/{r}/full/oof.npz"))
    for k in ["random","entropy","margin","head"]:
        axs[0,j].plot(qs*100,o[k][:,1],color=C[k],lw=1.8 if k=="head" else 1.2,label=lab[k])
        axs[1,j].plot(qs*100,o[k][:,0],color=C[k],lw=1.8 if k=="head" else 1.2)
    axs[0,j].set_title(name,fontsize=9.5,weight="bold"); axs[1,j].set_xlabel("Lesions sent for dermoscopy (%)")
    axs[1,j].axhline(0.95,color="#999",ls=":",lw=0.8); axs[1,j].set_ylim(0.90,0.97)
axs[0,0].set_ylabel("Malignant specificity"); axs[1,0].set_ylabel("Malignant sensitivity")
axs[0,0].legend(frameon=False,fontsize=7.5,loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig2_referral_unmatched.png",dpi=300); plt.close(fig)
# ---------- Figure 3: skin tone ----------
t=pd.read_csv("data/lesion_tone.csv"); grp=t.group.values; rng=np.random.default_rng(1)
rows=[]
for r,name in [("res1","ConvNeXt-T"),("res2","PanDerm-B")]:
    z=np.load(f"results/{r}/full/oof.npz"); ym=np.isin(z["y"],MAL)
    for s,sl in [("c","Clinical"),("d","Dermoscopic"),("both","Both")]:
        dec=z["T_"+s][:,MAL].sum(1)>=z["thr_"+s]
        for g in ["dark","medium","light"]:
            idx=np.where((grp==g)&ym)[0]; v=dec[idx]; bs=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(1000)]
            rows.append(dict(bb=name,view=sl,g=g,sens=v.mean(),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
d=pd.DataFrame(rows); d.to_csv("figures/fig3_data.csv",index=False)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True); GC={"dark":"#5b3a29","medium":"#b07a4f","light":"#e2c29b"}; GL={"dark":"Darker (0-2)","medium":"Medium (3)","light":"Lighter (4-5)"}
for j,bb in enumerate(["ConvNeXt-T","PanDerm-B"]):
    for i,v in enumerate(["Clinical","Dermoscopic","Both"]):
        for k,g in enumerate(["dark","medium","light"]):
            rr=d[(d.bb==bb)&(d.view==v)&(d.g==g)].iloc[0]; x=i+(k-1)*0.26
            axs[j].bar(x,rr.sens,0.24,color=GC[g],label=GL[g] if (i==0 and j==0) else None)
            axs[j].errorbar(x,rr.sens,yerr=[[rr.sens-rr.lo],[rr.hi-rr.sens]],color="k",lw=0.8,capsize=2)
    axs[j].set_xticks(range(3)); axs[j].set_xticklabels(["Clinical","Dermoscopic","Both"]); axs[j].set_title(bb,fontsize=9.5,weight="bold")
    axs[j].axhline(0.95,color="#999",ls=":",lw=0.8); axs[j].set_ylim(0.7,1.0)
axs[0].set_ylabel("Sensitivity at operating point"); fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,fontsize=7.5,loc="upper center",ncol=3,bbox_to_anchor=(0.5,0.99))
fig.tight_layout(rect=(0,0,1,0.88)); fig.savefig("figures/Figure3.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print(d.round(3).to_string(index=False))
