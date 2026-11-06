import numpy as np, pandas as pd
CL=["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]; MAL=np.array([0,1,6,7,9])
md=pd.read_csv("data/MILK10k_Training_Metadata.csv")
cl=md[md.image_type.str.startswith("clinical")].set_index("lesion_id")
gt=pd.read_csv("data/MILK10k_Training_GroundTruth.csv").sort_values("lesion_id").reset_index(drop=True)
tone=gt.lesion_id.map(cl.skin_tone_class).values; site=gt.lesion_id.map(cl.site).values; age=gt.lesion_id.map(cl.age_approx).values
grp=np.where(tone<=2,"dark",np.where(tone==3,"medium","light"))
z=np.load("results/res1/full/oof.npz"); y=z["y"]; ym=np.isin(y,MAL); fold=z["fold"]
assert (y==gt[CL].values.argmax(1)).all()
pd.DataFrame(dict(lesion_id=gt.lesion_id,tone=tone,group=grp,fold=fold)).to_csv("data/lesion_tone.csv",index=False)
print("Case mix by tone group (% of group):")
cm=pd.crosstab(grp,pd.Series(np.array(CL)[y],name="class"),normalize="index")*100
cm["malignant%"]=pd.Series(ym).groupby(grp).mean().values*100; cm["n"]=pd.Series(grp).value_counts().reindex(cm.index).values
cm["age_mean"]=pd.Series(age).groupby(grp).mean().reindex(cm.index).values
print(cm.round(1).to_string())
print("\nSite x tone (% of group):"); print((pd.crosstab(grp,site,normalize="index")*100).round(0).to_string())
rng=np.random.default_rng(5)
def ss(dec,m): return dec[m&ym].mean(), (~dec[m&~ym]).mean()
print("\nSens/spec at operating point by tone, 95% CI (bootstrap within group):")
rows=[]
for s in ["c","d","both"]:
    pm=z["T_"+s][:,MAL].sum(1); dec=pm>=z["thr_"+s]
    for g in ["dark","medium","light"]:
        m=grp==g; idx=np.where(m)[0]; bs=[]
        for _ in range(1000):
            i=rng.choice(idx,len(idx)); mm=np.zeros(len(y),bool); 
            d_=dec[i]; t_=ym[i]; bs.append((d_[t_].mean(), (~d_[~t_]).mean()))
        bs=np.array(bs); se,sp=ss(dec,m)
        rows.append(dict(state=s,group=g,n=m.sum(),n_mal=(m&ym).sum(),sens=se,sens_ci=np.percentile(bs[:,0],[2.5,97.5]).round(3),spec=sp,spec_ci=np.percentile(bs[:,1],[2.5,97.5]).round(3)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# sensitivity gap dark vs medium for c and both, bootstrap difference
for s in ["c","both"]:
    pm=z["T_"+s][:,MAL].sum(1); dec=pm>=z["thr_"+s]
    D=np.where((grp=="dark")&ym)[0]; M=np.where((grp=="medium")&ym)[0]
    d=[dec[rng.choice(D,len(D))].mean()-dec[rng.choice(M,len(M))].mean() for _ in range(2000)]
    print(f"sens gap dark-medium [{s}]: {dec[D].mean()-dec[M].mean():+.3f} CI {np.percentile(d,[2.5,97.5]).round(3)}")
# which malignant classes are missed in dark skin (clinical only)?
pm=z["T_c"][:,MAL].sum(1); dec=pm>=z["thr_c"]
print("\nMissed malignant by class, dark vs medium (clinical-only):")
for g in ["dark","medium"]:
    m=(grp==g)&ym; print(g, pd.Series(np.array(CL)[y[m]]).groupby(np.array(CL)[y[m]]).size().to_dict(), "missed:", pd.Series(np.array(CL)[y[m&~dec]]).value_counts().to_dict())
