import numpy as np, pandas as pd
from sklearn.metrics import f1_score
z=np.load("results/res1/full/oof.npz"); y=z["y"]; P=z["T_both"]; fold=z["fold"]
def mf1(y,p,w): return f1_score(y,(p*w).argmax(1),average="macro",labels=list(range(11)),zero_division=0)
grid=np.exp(np.linspace(np.log(0.25),np.log(8),25))
def fit(y,p,iters=3):
    w=np.ones(11)
    for _ in range(iters):
        for k in range(11):
            best=(mf1(y,p,w),w[k])
            for g in grid:
                w2=w.copy(); w2[k]=g; s=mf1(y,p,w2)
                if s>best[0]+1e-9: best=(s,g)
            w[k]=best[1]
    return w
pred=np.zeros(len(y),int)
for f in range(5):
    w=fit(y[fold!=f],P[fold!=f]); pred[fold==f]=(P[fold==f]*w).argmax(1)
print("OOF macro-F1 argmax: %.4f | cross-fitted class weights: %.4f" % (mf1(y,P,np.ones(11)), f1_score(y,pred,average='macro',labels=list(range(11)),zero_division=0)))
W=fit(y,P); print("weights on all OOF:", dict(zip(["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"],W.round(2))))
s=pd.read_csv("benchmark/milk10k_submission_v2.csv"); C=s.columns[1:]
q=s[C].values*W; q=q/q.sum(1,keepdims=True); s[C]=q
s.to_csv("benchmark/milk10k_submission_v3_classweights.csv",index=False,float_format="%.6f")
print("v3 written; argmax changes vs v2:", (q.argmax(1)!=pd.read_csv('benchmark/milk10k_submission_v2.csv')[C].values.argmax(1)).sum())
