import numpy as np
from sklearn.metrics import f1_score
rng=np.random.default_rng(0)
L=lambda r:np.load(f"results/{r}/oof.npz")
R={k:L(v) for k,v in dict(ck="res1/full",cn="res1/no_kd",cp="res1/paired_only",cc="res1/clin_only",cd="res1/derm_only",pf="res2/full",pp="res2/paired_only",pc="res2/clin_only",pdo="res2/derm_only").items()}
y=R["pf"]["y"]; n=len(y)
f1=lambda p,i:f1_score(y[i],p[i].argmax(1),average="macro",labels=range(11),zero_division=0)
idx=[rng.integers(0,n,n) for _ in range(1000)]
def d(a,sa,b,sb):
    pa,pb=R[a]["T_"+sa],R[b]["T_"+sb]; v=np.array([f1(pa,i)-f1(pb,i) for i in idx]); a_=np.arange(n)
    est=f1(pa,a_)-f1(pb,a_); lo,hi=np.percentile(v,[5,95]); print(f"{a}.{sa} - {b}.{sb}: {est:+.3f} 90%CI {lo:+.3f} to {hi:+.3f} equiv(±0.03): {lo>-0.03 and hi<0.03}")
for a,b in [("ck","cc"),("pf","pc")]: d(a,"c",b,"c")
for a,b in [("ck","cd"),("pf","pdo")]: d(a,"d",b,"d")
for a,b in [("ck","cp"),("pf","pp")]: d(a,"both",b,"both")
d("pf","both","ck","both"); d("pf","both","cn","both"); d("ck","both","cn","both")
