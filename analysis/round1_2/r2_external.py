import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
d7=pd.read_csv("results/res2/external_derm7pt_predictions.csv"); pad=pd.read_csv("results/res2/external_pad_ufes20_predictions.csv")
print("Derm7pt n=%d malignant=%d; diagnoses:"%(len(d7),d7.malignant.sum())); print(d7.diagnosis.value_counts().head(12).to_string())
def ss(y,p,t): dec=p>=t; return dec[y==1].mean(), (~dec[y==0]).mean()
def thr_sens(p,y,target=0.95):
    pos=np.sort(p[y==1]); k=int(np.floor((1-target)*len(pos))); return pos[min(k,len(pos)-1)]
rng=np.random.default_rng(0)
y=d7.malignant.values
print("\nDerm7pt AUROC / transferred-threshold sens,spec:")
for s in ["both","c","d"]:
    p=d7["pmal_"+s].values; print(f"  {s:4s} AUROC {roc_auc_score(y,p):.3f}  op(MILK) sens/spec {np.round(ss(y,p,d7['thr_op_'+s].iloc[0]),3)}")
# site recalibration: threshold from k local cases, evaluate on the rest (repeated)
print("\nDerm7pt local recalibration (threshold for 95% sens from k random local lesions; eval on the rest; 200 repeats):")
for s in ["both","c"]:
    p=d7["pmal_"+s].values
    for k in [50,100,200,400]:
        res=[]
        for _ in range(200):
            i=rng.permutation(len(y)); cal,ev=i[:k],i[k:]
            if y[cal].sum()<3: continue
            t=thr_sens(p[cal],y[cal]); res.append(ss(y[ev],p[ev],t))
        r=np.array(res); print(f"  {s:4s} k={k:3d}: sens {r[:,0].mean():.3f} [{np.percentile(r[:,0],5):.3f},{np.percentile(r[:,0],95):.3f}] spec {r[:,1].mean():.3f}")
# referral on Derm7pt with local thresholds (k=200 calibration, eval rest)
i=rng.permutation(len(y)); cal,ev=i[:200],i[200:]
tc=thr_sens(d7.pmal_c.values[cal],y[cal]); tb=thr_sens(d7.pmal_both.values[cal],y[cal])
pc,pb=d7.pmal_c.values[ev],d7.pmal_both.values[ev]; ye=y[ev]; dc=pc>=tc; db=pb>=tb
print("\nDerm7pt referral after local recalibration (k=200): clin-only sens/spec %s, paired %s"%(np.round(ss(ye,pc,tc),3),np.round(ss(ye,pb,tb),3)))
for name,sc in {"margin":-np.abs(pc-0.5),"head":d7.ref_score.values[ev]}.items():
    out=[]
    for q in [0.2,0.5]:
        r=np.zeros(len(ye),bool); r[np.argsort(-sc)[:int(q*len(ye))]]=True; dec=np.where(r,db,dc); out.append((q,round(dec[ye==1].mean(),3),round((~dec[ye==0]).mean(),3)))
    print("  ",name,out)
# PAD-UFES
print("\nPAD-UFES-20: n=%d images, patients=%d"%(len(pad),pad.patient_id.nunique())); print(pad.diagnosis.value_counts().to_string())
p=pad.pmal_c.values; y=pad.malignant.values
print("  all, AKIEC(ACK) counted malignant: AUROC %.3f"%roc_auc_score(y,p))
m=pad.diagnosis!="ACK"; print("  excluding ACK: AUROC %.3f (n=%d, malignant=%d)"%(roc_auc_score(y[m],p[m]),m.sum(),y[m].sum()))
y2=pad.diagnosis.isin(["BCC","SCC","MEL"]).astype(int).values
print("  ACK as benign: AUROC %.3f"%roc_auc_score(y2,p))
g=pad.assign(y2=y2).groupby("patient_id").agg(p=("pmal_c","max"),y=("y2","max"))
print("  patient-level (max over images, ACK benign): AUROC %.3f, n patients %d"%(roc_auc_score(g.y,g.p),len(g)))
print("  by Fitzpatrick (ACK benign):")
for f,gg in pad.assign(y2=y2).groupby("fitz"):
    if 0<gg.y2.sum()<len(gg): print("    FST %s n=%d mal=%d AUROC %.3f"%(f,len(gg),gg.y2.sum(),roc_auc_score(gg.y2,gg.pmal_c)))
print("    missing FST:", pad.fitz.isna().sum())
