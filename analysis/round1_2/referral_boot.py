import numpy as np, pandas as pd, sys
CLASSES = ["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]
MAL = np.array([0,1,6,7,9])
def load(e):
    z = np.load(f"results/res1/{e}/oof.npz"); d = z["fold"] >= 0
    return dict(y=z["y"][d], pc=z["T_c"][d], pb=z["T_both"][d], tc=z["thr_c"][d], tb=z["thr_both"][d], ref=z["ref"][d])
def ent(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def policy_scores(D):
    pmc = D["pc"][:,MAL].sum(1)
    s = {"entropy": ent(D["pc"]), "margin": -np.abs(pmc - D["tc"]), "margin05": -np.abs(pmc-0.5)}
    if not np.all(np.isnan(D["ref"])): s["referral_head"] = D["ref"]
    # oracle: refer lesions whose binary decision changes between clinical-only and paired
    pmb = D["pb"][:,MAL].sum(1)
    s["oracle"] = ((pmc >= D["tc"]) != (pmb >= D["tb"])).astype(float) + 1e-6*np.random.default_rng(0).random(len(pmc))
    return s
def curve(D, idx, score, qs):
    y = D["y"][idx]; ym = np.isin(y, MAL)
    pmc = D["pc"][idx][:,MAL].sum(1); pmb = D["pb"][idx][:,MAL].sum(1)
    dc = pmc >= D["tc"][idx]; db = pmb >= D["tb"][idx]
    order = np.argsort(-score[idx]); n = len(idx); out = []
    for q in qs:
        r = np.zeros(n, bool); r[order[:int(round(q*n))]] = True
        dec = np.where(r, db, dc)
        out.append((dec[ym].mean(), (~dec[~ym]).mean()))
    return np.array(out)  # sens, spec
def avoided(c, qs, tol):
    full = c[-1]; ok = (c[:,0] >= full[0]-tol) & (c[:,1] >= full[1]-tol)
    return 1 - qs[np.argmax(ok)] if ok.any() else 0.0
def gap_closed(c):
    # mean fraction of the specificity gap closed across referral budgets (area-based, threshold-free)
    g = c[-1,1] - c[0,1]
    return np.mean((c[:,1]-c[0,1]) / g) if g > 0 else np.nan
qs = np.round(np.linspace(0,1,101),2)
rng = np.random.default_rng(1)
rows = []
for e in ["full","no_kd"]:
    D = load(e); S = policy_scores(D); n = len(D["y"])
    base = np.arange(n)
    for tol in (0.005, 0.01, 0.02):
        for name, sc in S.items():
            rows.append(dict(exp=e, tol=tol, policy=name, avoided=avoided(curve(D, base, sc, qs), qs, tol)))
    # bootstrap at tol=0.01 + gap closed
    B = 300; boot = {k: [] for k in S}; bgap = {k: [] for k in S}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for k, sc in S.items():
            c = curve(D, idx, sc, qs); boot[k].append(avoided(c, qs, 0.01)); bgap[k].append(gap_closed(c))
    for k in S:
        c0 = curve(D, base, S[k], qs)
        rows.append(dict(exp=e, tol="boot0.01", policy=k, avoided=avoided(c0, qs, 0.01),
                         ci=np.percentile(boot[k],[2.5,97.5]).round(3).tolist(),
                         gap_closed=round(gap_closed(c0),3), gap_ci=np.nanpercentile(bgap[k],[2.5,97.5]).round(3).tolist()))
    if "referral_head" in S:
        for k in ("margin","entropy"):
            d = np.array(bgap["referral_head"]) - np.array(bgap[k])
            print(e, "gap_closed head minus", k, "mean %.3f CI %s P(d<=0)=%.3f" % (d.mean(), np.percentile(d,[2.5,97.5]).round(3), (d<=0).mean()))
            d2 = np.array(boot["referral_head"]) - np.array(boot[k])
            print(e, "avoided head minus", k, "mean %.3f CI %s P(d<=0)=%.3f" % (d2.mean(), np.percentile(d2,[2.5,97.5]).round(3), (d2<=0).mean()))
    c = curve(D, base, S["random"] if "random" in S else S["entropy"], qs)
    print(e, "clin-only sens/spec", c[0].round(3), "paired sens/spec", c[-1].round(3))
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r[r.tol!="boot0.01"].pivot_table(index=["exp","policy"], columns="tol", values="avoided").round(2))
print(r[r.tol=="boot0.01"][["exp","policy","avoided","ci","gap_closed","gap_ci"]].to_string(index=False))
