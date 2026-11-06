import numpy as np, pandas as pd
MAL=np.array([0,1,6,7,9]); rng=np.random.default_rng(0)
def thr95(p,ym):
    s=np.sort(p[ym]); k=int(np.floor(0.05*len(s))); return s[k]  # >= keeps >=95%
def crossfit(p,ym,fold):
    d=np.zeros(len(p),bool)
    for f in np.unique(fold):
        tr=fold!=f; t=thr95(p[tr],ym[tr]); d[fold==f]=p[fold==f]>=t
    return d
def boot(d,ym,B=1000):
    n=len(d); se=[];sp=[]
    for _ in range(B):
        i=rng.integers(0,n,n); a=ym[i]; dd=d[i]; se.append(dd[a].mean()); sp.append((~dd[~a]).mean())
    return np.percentile(se,[2.5,97.5]),np.percentile(sp,[2.5,97.5])
rows=[]
for r,bb in [("res1/full","ConvNeXt-T (KD)"),("res1/no_kd","ConvNeXt-T (no KD)"),("res2/full","PanDerm-B")]:
    z=np.load(f"results/{r}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL); fold=z["fold"]; n=len(y)
    pc,pb=z["T_c"],z["T_both"]; pmc=pc[:,MAL].sum(1); pmb=pb[:,MAL].sum(1)
    ent=-(pc*np.log(np.clip(pc,1e-12,1))).sum(1)
    # original fold-threshold decisions for oracle definitions
    dc=pmc>=z["thr_c"]; db=pmb>=z["thr_both"]
    changed=dc!=db; helpful=changed&(db==ym)
    S={"Learned head":z["ref"],"Margin rule":-np.abs(pmc-0.5),"Entropy":ent}
    def add(pol,q,m):
        p=np.where(m,pmb,pmc); d=crossfit(p,ym,fold); ci=boot(d,ym)
        rows.append(dict(backbone=bb,policy=pol,budget=q,referred=m.mean(),sens=d[ym].mean(),sens_lo=ci[0][0],sens_hi=ci[0][1],spec=d[~ym].__invert__().mean(),spec_lo=ci[1][0],spec_hi=ci[1][1]))
    add("Clinical only",0,np.zeros(n,bool)); add("Dermoscopy for all",1,np.ones(n,bool))
    for q in [0.1,0.2,0.3,0.5]:
        for k,s in S.items():
            o=np.argsort(-s); m=np.zeros(n,bool); m[o[:int(round(q*n))]]=True; add(k,q,m)
        sp=[];se=[]
        for _ in range(20):
            m=np.zeros(n,bool); m[rng.permutation(n)[:int(round(q*n))]]=True; p=np.where(m,pmb,pmc); d=crossfit(p,ym,fold); se.append(d[ym].mean()); sp.append((~d[~ym]).mean())
        rows.append(dict(backbone=bb,policy="Random",budget=q,referred=q,sens=np.mean(se),spec=np.mean(sp)))
    add("Oracle (all changed)",None,changed); add("Oracle (beneficial only)",None,helpful)
    print(bb,"changed %.3f helpful %.3f harmful %.3f"%(changed.mean(),helpful.mean(),(changed&~helpful).mean()))
d=pd.DataFrame(rows); d.to_csv("matched_referral.csv",index=False)
pd.set_option("display.width",200); print(d.round(3).to_string(index=False))
