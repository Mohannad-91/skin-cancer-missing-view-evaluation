import numpy as np
exec(open("analysis/round3/matched_ref.py").read().split("rows=[]")[0])
out=[]
for r,bb in [("res1/full","ConvNeXt KD"),("res1/no_kd","ConvNeXt noKD"),("res2/full","PanDerm")]:
    z=np.load(f"results/{r}/oof.npz"); y=z["y"]; ym=np.isin(y,MAL); fold=z["fold"]; n=len(y)
    pc,pb=z["T_c"],z["T_both"]; pmc=pc[:,MAL].sum(1); pmb=pb[:,MAL].sum(1)
    ent=-(pc*np.log(np.clip(pc,1e-12,1))).sum(1)
    def dec(s,q):
        o=np.argsort(-s); m=np.zeros(n,bool); m[o[:int(round(q*n))]]=True; return crossfit(np.where(m,pmb,pmc),ym,fold)
    for q in [0.2,0.3,0.5]:
        dh=dec(z["ref"],q); dm=dec(-np.abs(pmc-0.5),q); de=dec(ent,q)
        dr=[]; 
        for k in range(20):
            m=np.zeros(n,bool); m[np.random.default_rng(100+k).permutation(n)[:int(round(q*n))]]=True; dr.append(crossfit(np.where(m,pmb,pmc),ym,fold))
        b=[];c=[];e=[]
        for _ in range(1000):
            i=rng.integers(0,n,n); a=~ym[i]
            sh=(~dh[i][a]).mean(); b.append(sh-(~dm[i][a]).mean()); c.append(sh-(~de[i][a]).mean()); e.append(sh-np.mean([(~x[i][a]).mean() for x in dr[:5]]))
        f=lambda v:"%+.3f (%.3f to %.3f)"%(np.mean(v),*np.percentile(v,[2.5,97.5]))
        print(bb,q,"head-margin",f(b),"head-entropy",f(c),"head-random",f(e))
