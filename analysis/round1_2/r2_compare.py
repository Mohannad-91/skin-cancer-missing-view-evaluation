import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
MAL=np.array([0,1,6,7,9]); L=list(range(11))
def Z(r,e): z=np.load(f"results/{r}/{e}/oof.npz"); return z
def mf1(y,p): return f1_score(y,p.argmax(1),average="macro",labels=L,zero_division=0)
z1=Z("res1","full"); z2=Z("res2","full"); y=z2["y"]; assert (z1["y"]==y).all()
assert (z2["lesion_id"]==np.sort(z2["lesion_id"])).all()
rng=np.random.default_rng(7); n=len(y); idx=[rng.integers(0,n,n) for _ in range(1000)]
def boot(pa,pb,fn):
    d=[fn(y[i],pa[i])-fn(y[i],pb[i]) for i in idx]; return fn(y,pa)-fn(y,pb), np.percentile(d,[2.5,97.5])
mauc=lambda yy,p: roc_auc_score(np.isin(yy,MAL),p[:,MAL].sum(1))
print("PanDerm (r2) minus ConvNeXt (r1), full model:")
for s in ["both","c","d"]:
    a,b=z2["T_"+s],z1["T_"+s]
    d,ci=boot(a,b,mf1); d2,ci2=boot(a,b,mauc)
    print(f"  {s:4s} macroF1 {d:+.3f} {ci.round(3)} | malAUROC {d2:+.3f} {ci2.round(3)}")
print("\nRound 2 comparisons (macro-F1):")
for a,sa,b,sb in [("full","c","paired_only","c"),("full","d","paired_only","d"),("full","both","paired_only","both"),
                  ("full","c","clin_only","c"),("full","d","derm_only","d"),("full","both","derm_only","d")]:
    d,ci=boot(Z("res2",a)["T_"+sa],Z("res2",b)["T_"+sb],mf1); print(f"  {a}:{sa} - {b}:{sb} = {d:+.3f} {ci.round(3)}")
print("\nEnsemble r1+r2 (mean of calibrated probs):")
for s in ["both","c","d"]:
    pe=(z1["T_"+s]+z2["T_"+s])/2; d,ci=boot(pe,z2["T_"+s],mf1)
    print(f"  {s}: ens macroF1 {mf1(y,pe):.3f} (r2 {mf1(y,z2['T_'+s]):.3f}, r1 {mf1(y,z1['T_'+s]):.3f}); ens-r2 {d:+.3f} {ci.round(3)}; malAUROC {mauc(y,pe):.3f}")
