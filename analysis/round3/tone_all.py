import numpy as np, pandas as pd, statsmodels.formula.api as smf
MAL=np.array([0,1,6,7,9]); names={0:"AKIEC",1:"BCC",6:"OTHER",7:"MEL",9:"SCCKA"}
grp=pd.read_csv("data/lesion_tone.csv").group.values; rng=np.random.default_rng(1)
for r,bb in [("res1/full","ConvNeXt-T KD"),("res1/no_kd","ConvNeXt-T noKD"),("res2/full","PanDerm-B")]:
    z=np.load(f"results/{r}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL)
    for s in ["c","d","both"]:
        dec=z["T_"+s][:,MAL].sum(1)>=z["thr_"+s]
        df=pd.DataFrame(dict(miss=(~dec[ym]).astype(int),grp=grp[ym],cls=[names[c] for c in y[ym]]))
        m=smf.logit("miss ~ C(grp, Treatment('medium')) + C(cls, Treatment('BCC'))",df).fit(disp=0)
        k="C(grp, Treatment('medium'))[T.dark]"; ci=np.exp(m.conf_int().loc[k])
        sens={g:dec[ym&(grp==g)].mean() for g in ["dark","medium","light"]}
        print(f"{bb:16s} {s:4s} sens d/m/l {sens['dark']:.3f}/{sens['medium']:.3f}/{sens['light']:.3f}  OR dark {np.exp(m.params[k]):.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={m.pvalues[k]:.3f}")
print({g:int((np.isin(np.load('results/res2/full/oof.npz')['y'],MAL)&(grp==g)).sum()) for g in ['dark','medium','light']})
