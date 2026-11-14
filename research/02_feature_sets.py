"""Question: do richer features beat the v1 features (14 descriptors + 1024-bit fingerprint)?

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import *
import sys, pickle
df, d = load("aqsoldb"); y = d["y"]
X0, R, C, M = d["X0"], d["R"], d["C"], d["M"]
sets = {
 "base(desc14+fp1024)": X0,
 "rdkit217": R,
 "rdkit217+fp1024": np.hstack([R, X0[:,14:]]),
 "rdkit217+count2048": np.hstack([R, C]),
 "rdkit217+maccs": np.hstack([R, M]),
 "rdkit217+maccs+count2048": np.hstack([R, M, C]),
}
out = {}
for sp in ["random","scaffold"]:
    tr, te, folds = split(df, sp)
    for k, X in sets.items():
        oof, pt, _ = cv_and_test(lgbm, X, y, tr, te, folds)
        out[(sp,k)] = (oof, pt)
        print(f"{sp:9s} {k:28s} CV {rmse(oof,y[tr]):.3f}  test RMSE {rmse(pt,y[te]):.3f} R2 {r2(y[te],pt):.3f} within1 {np.mean(np.abs(pt-y[te])<=1):.3f}", flush=True)
pickle.dump(out, open(str(CACHE / "e1.pkl"),"wb"))
