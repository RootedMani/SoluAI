"""Question: do E-state / Crippen-type / atom-pair / torsion features add anything on top?

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import *
df, d = load("aqsoldb"); y = d["y"]; d2 = np.load(str(CACHE / "aqsoldb2.npz")); rel = pd.read_csv(str(CACHE / "rel.csv")); reliable = np.isin(rel.group_r.values, ["G3","G5"])
R, FP, M, C = d["R"], d["X0"][:,14:], d["M"], d["C"]; E, CR, AP, TT = d2["E"], d2["CR"], d2["AP"], d2["TT"]
sets = {"R+FP": [R,FP], "R+E": [R,E], "R+CR": [R,CR], "R+E+CR": [R,E,CR], "R+E+CR+FP": [R,E,CR,FP], "R+E+CR+AP": [R,E,CR,AP], "R+E+CR+M+FP+AP+TT": [R,E,CR,M,FP,AP,TT]}
mk = lambda: lgbm(objective="huber", alpha=1.0)
for sp in ["random","scaffold"]:
    tr, te, folds = split(df, sp)
    for k, parts in sets.items():
        X = np.hstack(parts); o,p,_ = cv_and_test(mk, X, y, tr, te, folds); yt=y[te]; m=reliable[te]
        print(f"{sp:9s} {k:20s} CV {rmse(o,y[tr]):.3f} | test {rmse(p,yt):.3f} R2 {r2(yt,p):.3f} w1 {np.mean(np.abs(p-yt)<=1):.3f} | reliable {rmse(p[m],yt[m]):.3f} R2 {r2(yt[m],p[m]):.3f}", flush=True)
