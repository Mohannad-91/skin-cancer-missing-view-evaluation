import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
MAL=np.array([0,1,6,7,9]); rng=np.random.default_rng(0)
L=lambda r:np.load(f"results/{r}/oof.npz")
nk=L("res1/no_kd"); po=L("res1/paired_only"); pd_=L("res2/full"); kd=L("res1/full"); co=L("res1/clin_only"); do=L("res1/derm_only")
y=nk["y"]; n=len(y); assert (y==pd_["y"]).all()
f1=lambda p,i:f1_score(y[i],p[i].argmax(1),average="macro",labels=range(11))
def diff(pa,pb,B=1000):
    v=[];
    for _ in range(B):
        i=rng.integers(0,n,n); v.append(f1(pa,i)-f1(pb,i))
    a=np.arange(n); return "%+.3f (%.3f to %.3f)"%(f1(pa,a)-f1(pb,a),*np.percentile(v,[2.5,97.5]))
a=np.arange(n)
for s in ["both","c","d"]:
    print(s,"noKD F1 %.3f"%f1(nk["T_"+s],a),"KD %.3f"%f1(kd["T_"+s],a),"malAUROC noKD %.3f"%roc_auc_score(np.isin(y,MAL),nk["T_"+s][:,MAL].sum(1)))
    print("  noKD - paired_only",diff(nk["T_"+s],po["T_"+s]))
    print("  PanDerm - noKD",diff(pd_["T_"+s],nk["T_"+s]))
print("noKD-clin_only",diff(nk["T_c"],co["T_c"])); print("noKD-derm_only",diff(nk["T_d"],do["T_d"]))
