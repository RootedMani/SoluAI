"""Question: can a small, hand-readable subset of the RDKit descriptors keep the gain?

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import *
import json
df, d = load("aqsoldb"); y = d["y"]; R, X0 = d["R"], d["X0"]; FP = X0[:,14:]; rn = list(d["rnames"])
mk = lambda: lgbm(objective="huber", alpha=1.0)
rank = {}
for sp in ["scaffold","random"]:
    tr, te, folds = split(df, sp)
    m = mk().fit(np.hstack([R,FP])[tr], y[tr])
    c = np.abs(m.predict(np.hstack([R,FP])[tr][:3000], pred_contrib=True)[:, :R.shape[1]]).mean(0)
    order = np.argsort(-c); rank[sp] = [rn[i] for i in order]
    print(sp, "top:", [(rn[i], round(float(c[i]),3)) for i in order[:45]], flush=True)
    for n in [15, 25, 40, 70]:
        X = np.hstack([R[:, order[:n]], FP]); o,p,_ = cv_and_test(mk, X, y, tr, te, folds)
        print(f"{sp:9s} top{n:3d}+FP CV {rmse(o,y[tr]):.3f} | test {rmse(p,y[te]):.3f} R2 {r2(y[te],p):.3f}", flush=True)
    X = np.hstack([X0[:, :14], FP]); o,p,_ = cv_and_test(mk, X, y, tr, te, folds)
    print(f"{sp:9s} desc14+FP huber CV {rmse(o,y[tr]):.3f} | test {rmse(p,y[te]):.3f} R2 {r2(y[te],p):.3f}", flush=True)
json.dump(rank, open(str(CACHE / "rank.json"),"w"))
