import numpy as np
from sklearn.metrics import f1_score
rng=np.random.default_rng(0)
P={"ck":"res1/full","cp":"res1/paired_only","cc":"res1/clin_only","cd":"res1/derm_only","pj":"res2/full","pp":"res2/paired_only","pc":"res2/clin_only","pd":"res2/derm_only","md":"res3A/moddrop","mi":"res3A/paired_meanimp","cm":"res3B/full","cn":"res1/no_kd"}
R={k:np.load(f"results/{v}/oof.npz") for k,v in P.items()}
y=R["pj"]["y"]; n=len(y); a=np.arange(n)
f1=lambda p,i:f1_score(y[i],p[i].argmax(1),average="macro",labels=range(11),zero_division=0)
idx=[rng.integers(0,n,n) for _ in range(1000)]
def d(lab,A,sa,B,sb=None):
    sb=sb or sa; pa=R[A]["T_"+sa]; pb=R[B]["T_"+sb]; v=np.array([f1(pa,i)-f1(pb,i) for i in idx]); e=f1(pa,a)-f1(pb,a)
    l,h=np.percentile(v,[5,95]); l9,h9=np.percentile(v,[2.5,97.5])
    print(f"{lab:45s} {e:+.3f} | 95% {l9:+.3f} to {h9:+.3f} | 90% {l:+.3f} to {h:+.3f} | eq03 {l>-0.03 and h<0.03} eq02 {l>-0.02 and h<0.02}")
for lab,A,B,s in [("CX vs paired-only c","ck","cp","c"),("CX vs paired-only d","ck","cp","d"),("CX vs paired-only both","ck","cp","both"),("CX vs clin spec","ck","cc","c"),("CX vs derm spec","ck","cd","d"),
                  ("PD vs paired-only c","pj","pp","c"),("PD vs paired-only d","pj","pp","d"),("PD vs paired-only both","pj","pp","both"),
                  ("PD vs meanimp c","pj","mi","c"),("PD vs meanimp d","pj","mi","d"),("PD vs meanimp both","pj","mi","both"),
                  ("PD vs moddrop c","pj","md","c"),("PD vs moddrop d","pj","md","d"),("PD vs moddrop both","pj","md","both"),
                  ("PD vs clin spec","pj","pc","c"),("PD vs derm spec","pj","pd","d"),
                  ("moddrop vs paired-only c","md","pp","c"),("moddrop vs paired-only d","md","pp","d"),
                  ("PD vs CX matched both","pj","cm","both"),("PD vs CX matched c","pj","cm","c"),("PD vs CX matched d","pj","cm","d")]:
    d(lab,A,s,B)
