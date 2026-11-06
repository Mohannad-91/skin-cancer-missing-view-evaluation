import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
exec(open("analysis/round3/matched_ref.py").read().split("rows=[]")[0])
grp=pd.read_csv("data/lesion_tone.csv").group.values
names={0:"AKIEC",1:"BCC",6:"OTHER",7:"MEL",9:"SCCKA"}
for r,bb in [("res2/full","PanDerm s0"),("res3A/full_s1","PanDerm s1"),("res3A/full_s2","PanDerm s2"),("res3B/full","ConvNeXt matched"),("res3A/moddrop","moddrop")]:
    z=np.load(f"results/{r}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL); fold=z["fold"]; n=len(y)
    pc,pb=z["T_c"],z["T_both"]; pmc=pc[:,MAL].sum(1); pmb=pb[:,MAL].sum(1); ent=-(pc*np.log(np.clip(pc,1e-12,1))).sum(1)
    print("==",bb,"malAUROC both %.3f c %.3f d %.3f"%tuple(roc_auc_score(ym,z["T_"+s][:,MAL].sum(1)) for s in ["both","c","d"]))
    def dec(s,q):
        o=np.argsort(-s); m=np.zeros(n,bool); m[o[:int(round(q*n))]]=True; return crossfit(np.where(m,pmb,pmc),ym,fold)
    sp=lambda d:(~d[~ym]).mean()
    d0=crossfit(pmc,ym,fold); d1=crossfit(pmb,ym,fold); print(" clin spec %.3f  derm-all spec %.3f"%(sp(d0),sp(d1)))
    if not np.isnan(z["ref"]).all():
        for q in [0.2,0.5]:
            dh=dec(z["ref"],q); dm=dec(-np.abs(pmc-0.5),q); de=dec(ent,q)
            dr=[crossfit(np.where(np.isin(np.arange(n),np.random.default_rng(100+k).permutation(n)[:int(round(q*n))]),pmb,pmc),ym,fold) for k in range(5)]
            b=[];c=[];e=[]
            for _ in range(500):
                i=rng.integers(0,n,n); a=~ym[i]; sh=(~dh[i][a]).mean()
                b.append(sh-(~dm[i][a]).mean()); c.append(sh-(~de[i][a]).mean()); e.append(sh-np.mean([(~x[i][a]).mean() for x in dr]))
            f=lambda v:"%+.3f (%.3f to %.3f)"%(np.mean(v),*np.percentile(v,[2.5,97.5]))
            print(f" q={q} spec head {sp(dh):.3f} margin {sp(dm):.3f} entropy {sp(de):.3f} | head-margin {f(b)} head-entropy {f(c)} head-random {f(e)}")
    for s in ["c","d","both"]:
        dd=z["T_"+s][:,MAL].sum(1)>=z["thr_"+s]
        df=pd.DataFrame(dict(miss=(~dd[ym]).astype(int),grp=grp[ym],cls=[names[c] for c in y[ym]]))
        m=smf.logit("miss ~ C(grp, Treatment('medium')) + C(cls, Treatment('BCC'))",df).fit(disp=0)
        k="C(grp, Treatment('medium'))[T.dark]"; ci=np.exp(m.conf_int().loc[k])
        sens=[dd[ym&(grp==g)].mean() for g in ["dark","medium","light"]]
        print(f" tone {s}: sens {sens[0]:.3f}/{sens[1]:.3f}/{sens[2]:.3f} OR {np.exp(m.params[k]):.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={m.pvalues[k]:.3f}")
