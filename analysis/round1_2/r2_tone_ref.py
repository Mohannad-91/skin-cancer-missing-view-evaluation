import numpy as np, pandas as pd, statsmodels.formula.api as smf
CL=np.array(["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]); MAL=np.array([0,1,6,7,9])
def load(r):
    z=np.load(f"results/{r}/full/oof.npz"); return z
z=load("res2"); y=z["y"]; ym=np.isin(y,MAL); n=len(y)
t=pd.read_csv("data/lesion_tone.csv"); grp=t.group.values
assert (z["lesion_id"]==t.lesion_id.values).all()
print("== Skin tone, PanDerm full model")
for s in ["c","d","both"]:
    dec=z["T_"+s][:,MAL].sum(1)>=z["thr_"+s]
    sens={g:dec[(grp==g)&ym].mean() for g in ["dark","medium","light"]}
    df=pd.DataFrame(dict(cls=CL[y],grp=grp,miss=(~dec).astype(int)))[ym]; df=df[df.cls.isin(["MEL","BCC","AKIEC"])]
    m=smf.logit("miss ~ C(grp, Treatment('medium')) + C(cls)",df).fit(disp=0)
    k="C(grp, Treatment('medium'))[T.dark]"
    print(f"  {s:4s} sens dark {sens['dark']:.3f} medium {sens['medium']:.3f} light {sens['light']:.3f} | adj OR dark {np.exp(m.params[k]):.2f} "
          f"CI [{np.exp(m.conf_int().loc[k,0]):.2f}, {np.exp(m.conf_int().loc[k,1]):.2f}] p={m.pvalues[k]:.4f}")
dark=grp=="dark"
pmc=z["T_c"][:,MAL].sum(1); pmb=z["T_both"][:,MAL].sum(1); dc=pmc>=z["thr_c"]; db=pmb>=z["thr_both"]
gain=(db==ym).astype(int)-(dc==ym).astype(int); print("  dermoscopy helps %.3f, hurts %.3f of lesions"%((gain>0).mean(),(gain<0).mean()))
def ev(score,q):
    r=np.zeros(n,bool); r[np.argsort(-score)[:int(round(q*n))]]=True; dec=np.where(r,db,dc)
    return dec[ym].mean(),(~dec[~ym]).mean(),dec[ym&dark].mean()
m05=-np.abs(pmc-0.5); pols={"margin":m05,"head":z["ref"],"dark_first":m05+10*dark}
print("  budget policy sens spec sens_dark")
for q in [0,0.12,0.2,0.3,0.5,1.0]:
    for k,s_ in pols.items(): print("  %.2f %-10s %.3f %.3f %.3f"%((q,k)+ev(s_,q)))
