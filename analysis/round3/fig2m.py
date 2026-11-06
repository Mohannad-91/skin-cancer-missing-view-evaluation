import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("analysis/round3/matched_ref.py").read().split("rows=[]")[0])
C={"Learned head":"#1f5aa6","Margin rule":"#d9822b","Entropy":"#7a7a7a","Random":"#bbbbbb"}
qs=np.linspace(0,1,21)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
for j,(r,name) in enumerate([("res1/full","ConvNeXt-Tiny"),("res2/full","PanDerm-Base")]):
    z=np.load(f"results/{r}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL); fold=z["fold"]; n=len(y)
    pc,pb=z["T_c"],z["T_both"]; pmc=pc[:,MAL].sum(1); pmb=pb[:,MAL].sum(1); ent=-(pc*np.log(np.clip(pc,1e-12,1))).sum(1)
    S={"Learned head":z["ref"],"Margin rule":-np.abs(pmc-0.5),"Entropy":ent}
    def spec(m): d=crossfit(np.where(m,pmb,pmc),ym,fold); return (~d[~ym]).mean()
    for k in ["Random","Entropy","Margin rule","Learned head"]:
        v=[]
        for q in qs:
            if k=="Random":
                v.append(np.mean([spec(np.isin(np.arange(n),np.random.default_rng(s).permutation(n)[:int(round(q*n))])) for s in range(10)]))
            else:
                o=np.argsort(-S[k]); m=np.zeros(n,bool); m[o[:int(round(q*n))]]=True; v.append(spec(m))
        axs[j].plot(qs*100,v,color=C[k],lw=1.8 if k=="Learned head" else 1.2,label=k)
    axs[j].set_title(name,fontsize=9.5,weight="bold"); axs[j].set_xlabel("Lesions sent for dermoscopy (%)")
    for s in ["top","right"]: axs[j].spines[s].set_visible(False)
axs[0].set_ylabel("Malignant specificity\nat 95% sensitivity"); axs[0].legend(frameon=False,fontsize=7.5,loc="lower right")
fig.tight_layout(); fig.savefig("figures/Figure2.png",dpi=300); print("ok")
