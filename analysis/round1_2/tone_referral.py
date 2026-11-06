import numpy as np, pandas as pd
MAL=np.array([0,1,6,7,9])
t=pd.read_csv("data/lesion_tone.csv"); dark=(t.group=="dark").values
z=np.load("results/res1/full/oof.npz"); y=z["y"]; ym=np.isin(y,MAL); n=len(y)
pmc=z["T_c"][:,MAL].sum(1); pmb=z["T_both"][:,MAL].sum(1)
dc=pmc>=z["thr_c"]; db=pmb>=z["thr_both"]
def ev(score,q):
    r=np.zeros(n,bool); r[np.argsort(-score)[:int(round(q*n))]]=True; dec=np.where(r,db,dc)
    return dict(sens=dec[ym].mean(), spec=(~dec[~ym]).mean(), sens_dark=dec[ym&dark].mean(), sens_nondark=dec[ym&~dark].mean(), dark_referred=r[dark].mean())
m05=-np.abs(pmc-0.5)
pol={"margin (tone-blind)":m05, "referral_head":z["ref"], "dark first, then margin":m05+10*dark}
rows=[]
for q in [0.0,0.12,0.2,0.3,0.4,0.5,1.0]:
    for k,s in pol.items(): rows.append(dict(budget=q,policy=k,**ev(s,q)))
r=pd.DataFrame(rows); pd.set_option("display.width",200)
print(r.round(3).to_string(index=False))
print("dark share of lesions:", dark.mean().round(3))
