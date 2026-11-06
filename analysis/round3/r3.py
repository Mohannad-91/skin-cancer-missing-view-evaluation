import numpy as np
from sklearn.metrics import f1_score
rng=np.random.default_rng(0)
P={"pj":"res2/full","pp":"res2/paired_only","pc":"res2/clin_only","pd":"res2/derm_only","md":"res3A/moddrop","mi":"res3A/paired_meanimp","s1":"res3A/full_s1","s2":"res3A/full_s2","cm":"res3B/full","ck":"res1/full","cn":"res1/no_kd"}
R={k:np.load(f"results/{v}/oof.npz") for k,v in P.items()}
y=R["pj"]["y"]; n=len(y)
for k in R: assert (R[k]["y"]==y).all(),k
for k in ["md","mi","s1","s2","cm"]: assert (R[k]["lesion_id"]==R["pj"]["lesion_id"]).all() and (R[k]["fold"]==R["pj"]["fold"]).all(),k
f1=lambda p,i:f1_score(y[i],p[i].argmax(1),average="macro",labels=range(11),zero_division=0)
idx=[rng.integers(0,n,n) for _ in range(1000)]; a=np.arange(n)
def d(A,sa,B,sb=None):
    sb=sb or sa; pa=R[A]["T_"+sa]; pb=R[B]["T_"+sb]; v=np.array([f1(pa,i)-f1(pb,i) for i in idx]); e=f1(pa,a)-f1(pb,a)
    l95,h95=np.percentile(v,[2.5,97.5]); l90,h90=np.percentile(v,[5,95])
    print(f"{A}.{sa} - {B}.{sb}: {e:+.3f} (95% {l95:+.3f} to {h95:+.3f}; 90% {l90:+.3f} to {h90:+.3f}) equiv={l90>-0.03 and h90<0.03}")
for s in ["both","c","d"]:
    d("pj",s,"md"); d("s1",s,"md"); d("s2",s,"md")
for s in ["c","d"]: d("pj",s,"mi"); d("mi",s,"pp")
d("mi","both","pj")
for s in ["both","c","d"]: d("pj",s,"cm")
d("cm","c","pp"); d("cm","both","cn"); d("cm","c","cn")
# ensemble of three seeds
import numpy as np
for s in ["both","c","d"]:
    R["sx"]=None
