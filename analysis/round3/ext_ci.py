import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
MAL=np.array([0,1,6,7,9]); rng=np.random.default_rng(0)
z=np.load("results/res2/full/oof.npz"); ym=np.isin(z["y"],MAL)
def thr95(p,y): s=np.sort(p[y]); return s[int(np.floor(0.05*len(s)))]
pooled={s:thr95(z["T_"+s][:,MAL].sum(1),ym) for s in ["both","c","d"]}
d7=pd.read_csv("results/res2/external_derm7pt_predictions.csv"); pad=pd.read_csv("results/res2/external_pad_ufes20_predictions.csv")
def ci(y,p,t,B=1000):
    n=len(y); A=[];SE=[];SP=[]
    for _ in range(B):
        i=rng.integers(0,n,n); yy=y[i]; pp=p[i]
        if yy.min()==yy.max(): continue
        A.append(roc_auc_score(yy,pp)); d=pp>=t; SE.append(d[yy==1].mean()); SP.append((~d[yy==0]).mean())
    d=p>=t; q=lambda v:"(%.3f-%.3f)"%tuple(np.percentile(v,[2.5,97.5]))
    return "AUROC %.3f %s | sens %.3f %s | spec %.3f %s"%(roc_auc_score(y,p),q(A),d[y==1].mean(),q(SE),(~d[y==0]).mean(),q(SP))
for s in ["both","c","d"]:
    y=d7.malignant.values; p=d7["pmal_"+s].values
    print("Derm7pt",s,"fold-mean thr %.3f:"%d7["thr_op_"+s].iloc[0],ci(y,p,d7["thr_op_"+s].iloc[0]))
    print("Derm7pt",s,"pooled-OOF thr %.3f:"%pooled[s],ci(y,p,pooled[s]))
y=pad.malignant.values; p=pad.pmal_c.values
print("PAD per image fold-mean:",ci(y,p,pad.thr_op_c.iloc[0])); print("PAD per image pooled:",ci(y,p,pooled["c"]))
g=pad.groupby("patient_id").agg(y=("malignant","max"),p=("pmal_c","max"))
print("PAD per patient (max) pooled:",ci(g.y.values,g.p.values,pooled["c"]))
print("pad n mal",y.sum(),len(y),"patients",len(g),g.y.sum(),"d7 mal",d7.malignant.sum())
print("---- PAD, ACK benign")
pad["m2"]=pad.diagnosis.isin(["BCC","SCC","MEL"]).astype(int)
y=pad.m2.values; print("PAD img fold-mean:",ci(y,p,pad.thr_op_c.iloc[0])); print("PAD img pooled:",ci(y,p,pooled["c"]))
g=pad.groupby("patient_id").agg(y=("m2","max"),p=("pmal_c","max")); print("PAD patient pooled:",ci(g.y.values,g.p.values,pooled["c"]), "n mal img",y.sum(),"pat",g.y.sum())
