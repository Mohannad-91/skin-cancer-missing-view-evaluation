import numpy as np, pandas as pd
import statsmodels.formula.api as smf
CL=np.array(["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]); MAL=np.array([0,1,6,7,9])
t=pd.read_csv("data/lesion_tone.csv"); grp=t.group.values
for e,s in [("full","c"),("full","both"),("full","d")]:
    z=np.load(f"results/res1/{e}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL)
    dec=z["T_"+s][:,MAL].sum(1)>=z["thr_"+s]
    df=pd.DataFrame(dict(cls=CL[y],grp=grp,hit=dec.astype(int)))[ym]
    tab=df.pivot_table(index="cls",columns="grp",values="hit",aggfunc=["mean","size"])
    w=df[df.grp=="dark"].cls.value_counts(normalize=True)
    def std(g):  # sensitivity of group g re-weighted to dark-skin malignant class mix
        m=tab["mean"][g]; return float((w*m.reindex(w.index)).sum())
    print(f"== {s}: per-class sensitivity by tone")
    print(tab["mean"].round(3).join(tab["size"].add_prefix("n_")).to_string())
    print(f"  crude sens dark={df[df.grp=='dark'].hit.mean():.3f}; medium={df[df.grp=='medium'].hit.mean():.3f}; "
          f"medium standardized to dark case-mix={std('medium'):.3f}; light standardized={std('light'):.3f}")
    df2=df[df.cls.isin(["MEL","BCC","AKIEC"])].copy(); df2["miss"]=1-df2.hit
    m=smf.logit("miss ~ C(grp, Treatment('medium')) + C(cls)", df2).fit(disp=0)
    print("  logit(miss) ~ tone + class (MEL,BCC,AKIEC):"); print(np.exp(m.params).round(2).to_string(), "\n  p:", m.pvalues.round(4).to_dict())
