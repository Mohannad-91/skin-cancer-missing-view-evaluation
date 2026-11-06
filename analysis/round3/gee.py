import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, warnings; warnings.filterwarnings("ignore")
MAL=np.array([0,1,6,7,9]); names={0:"AKIEC",1:"BCC",6:"OTHER",7:"MEL",9:"SCCKA"}
m=pd.read_csv("data/MILK10k_Training_Metadata.csv")
m=m.sort_values("image_type").groupby("lesion_id").first().sort_index()
grp=pd.read_csv("data/lesion_tone.csv").group.values
runs={"cx":"res1/full","cn":"res1/no_kd","cm":"res3B/full","p0":"res2/full","p1":"res3A/full_s1","p2":"res3A/full_s2","md":"res3A/moddrop"}
z0=np.load("results/res2/full/oof.npz"); assert (m.index.values==z0["lesion_id"]).all()
age=m.age_approx.values; site=m.site.fillna("unknown").values
for s in ["c","d","both"]:
    rows=[]
    for k,r in runs.items():
        z=np.load(f"results/{r}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL)
        dec=z["T_"+s][:,MAL].sum(1)>=z["thr_"+s]
        idx=np.where(ym)[0]
        rows.append(pd.DataFrame(dict(lesion=idx,miss=(~dec[idx]).astype(int),grp=grp[idx],cls=[names[c] for c in y[idx]],age=age[idx],site=site[idx],run=k)))
    D=pd.concat(rows); D["age"]=D.age.fillna(D.age.median())
    for f,lab in [("miss ~ C(grp, Treatment('medium')) + C(cls, Treatment('BCC')) + C(run)","class"),("miss ~ C(grp, Treatment('medium')) + C(cls, Treatment('BCC')) + age + C(site) + C(run)","class+age+site")]:
        g=smf.gee(f,"lesion",D,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
        kk="C(grp, Treatment('medium'))[T.dark]"; ci=np.exp(g.conf_int().loc[kk])
        print(f"{s:4s} {lab:15s} pooled OR dark {np.exp(g.params[kk]):.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={g.pvalues[kk]:.3f}")
t=pd.Series(m.skin_tone_class.values)
print(t[grp=="dark"].value_counts())
yy=z0["y"]; cls=np.array(["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"])
print(pd.Series(cls[yy[grp=="dark"]]).value_counts(normalize=True).round(3))
