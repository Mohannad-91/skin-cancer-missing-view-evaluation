import numpy as np, pandas as pd
from sklearn.metrics import f1_score
CL = ["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]
def Z(e): z=np.load(f"results/res1/{e}/oof.npz"); d=z["fold"]>=0; return z, d
def mf1(y,p): return f1_score(y,p.argmax(1),average="macro",labels=list(range(11)),zero_division=0)
def ece(p,y,b=15):
    c,pr=p.max(1),p.argmax(1); a=(pr==y); e=0
    for lo,hi in zip(np.linspace(0,1,b+1)[:-1],np.linspace(0,1,b+1)[1:]):
        m=(c>lo)&(c<=hi)
        if m.any(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
rng=np.random.default_rng(3); B=1000
pairs=[("full","both","paired_only","both"),("full","c","paired_only","c"),("full","d","paired_only","d"),
       ("full","c","clin_only","c"),("full","d","derm_only","d"),("full","both","no_kd","both"),
       ("full","c","no_kd","c"),("full","d","no_kd","d"),("full","both","derm_only","d")]
zf,d=Z("full"); y=zf["y"][d]; n=len(y)
idxs=[rng.integers(0,n,n) for _ in range(B)]
print("Paired bootstrap, macro-F1 difference (A - B):")
for a,sa,b,sb in pairs:
    pa=Z(a)[0]["T_"+sa][d]; pb=Z(b)[0]["T_"+sb][d]
    diff=[mf1(y[i],pa[i])-mf1(y[i],pb[i]) for i in idxs]
    print(f"  {a}:{sa:4s} - {b}:{sb:4s} = {mf1(y,pa)-mf1(y,pb):+.3f}  CI {np.percentile(diff,[2.5,97.5]).round(3)}")
print("\nECE: global T minus per-state T (positive = per-state better):")
for e in ["full","no_kd","paired_only"]:
    z,_=Z(e)
    for s in ["both","c","d"]:
        G,T=z["G_"+s][d],z["T_"+s][d]
        diff=[ece(G[i],y[i])-ece(T[i],y[i]) for i in idxs[:300]]
        print(f"  {e:12s}{s:5s} {ece(G,y)-ece(T,y):+.4f} CI {np.percentile(diff,[2.5,97.5]).round(4)}")
print("\nPer-class F1 (full, per-state T) and support:")
rows={}
for s in ["both","c","d"]:
    p=zf["T_"+s][d]; rows[s]=f1_score(y,p.argmax(1),average=None,labels=list(range(11)),zero_division=0)
t=pd.DataFrame(rows,index=CL); t["n"]=np.bincount(y,minlength=11); print(t.round(3))
print("\nSkin tone (full):"); st=pd.read_csv("results/res1/full/skin_tone.csv"); print(st.round(3).to_string(index=False))
