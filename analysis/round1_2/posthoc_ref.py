import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
exec(open("analysis/round1_2/referral_boot.py").read().split("qs = np.round")[0])
qs = np.round(np.linspace(0,1,101),2)
def feats(D):
    pc = D["pc"]; pmc = pc[:,MAL].sum(1)
    return np.column_stack([pc, pmc, pmc - D["tc"], np.abs(pmc - D["tc"]), np.abs(pmc-0.5), ent(pc), pc.max(1), D["ref"] if not np.all(np.isnan(D["ref"])) else np.zeros(len(pc))])
rng = np.random.default_rng(2)
for e in ["full","no_kd"]:
    z = np.load(f"results/res1/{e}/oof.npz"); fold = z["fold"][z["fold"]>=0]
    D = load(e); X = feats(D); y = D["y"]; ym = np.isin(y, MAL)
    dc = D["pc"][:,MAL].sum(1) >= D["tc"]; db = D["pb"][:,MAL].sum(1) >= D["tb"]
    gain = (db == ym).astype(float) - (dc == ym).astype(float)       # benefit of getting dermoscopy
    preds = {}
    for name, mk in [("gbm", lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0)),
                     ("ridge", lambda: Ridge(alpha=1.0))]:
        p = np.zeros(len(y))
        for f in np.unique(fold):            # cross-fitting: model for fold f never sees fold f
            m = mk().fit(X[fold != f], gain[fold != f]); p[fold == f] = m.predict(X[fold == f])
        preds["posthoc_" + name] = p
    S = policy_scores(D); S.update(preds)
    n = len(y); base = np.arange(n)
    res = {k: (avoided(curve(D, base, s, qs), qs, 0.01), gap_closed(curve(D, base, s, qs))) for k, s in S.items()}
    B = 300; bg = {k: [] for k in S}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for k, s in S.items(): bg[k].append(gap_closed(curve(D, idx, s, qs)))
    print(f"== {e}  (gain>0: {np.mean(gain>0):.3f}, gain<0: {np.mean(gain<0):.3f})")
    for k in S:
        print(f"{k:16s} avoided@0.01={res[k][0]:.2f}  gap_closed={res[k][1]:.3f}  CI={np.percentile(bg[k],[2.5,97.5]).round(3)}")
    for a in ("posthoc_gbm","posthoc_ridge"):
        for b in ("margin05","referral_head","margin"):
            if b in S:
                d = np.array(bg[a]) - np.array(bg[b]); print(f"  {a} - {b}: {d.mean():+.3f} CI {np.percentile(d,[2.5,97.5]).round(3)}")
