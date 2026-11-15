"""Question: does it help to trust repeat-measured values more, or to use a robust (Huber) loss?

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import *
import pickle
df, d = load("aqsoldb"); y = d["y"]; rel = pd.read_csv(str(CACHE / "rel.csv"))
X = np.hstack([d["R"], d["X0"][:,14:]])
g = rel.group_r.values; reliable = np.isin(g, ["G3","G5"]); bad = np.isin(g, ["G2","G4"])
def rep(tag, oof, p, tr, te):
    yt=y[te]; m=reliable[te]
    print(f"{tag:42s} CV {rmse(oof,y[tr]):.3f} | test {rmse(p,yt):.3f} R2 {r2(yt,p):.3f} w1 {np.mean(np.abs(p-yt)<=1):.3f} | reliable {rmse(p[m],yt[m]):.3f} R2 {r2(yt[m],p[m]):.3f} w1 {np.mean(np.abs(p[m]-yt[m])<=1):.3f}", flush=True)
for sp in ["random","scaffold"]:
    tr, te, folds = split(df, sp); print(sp)
    oof0,p0,_ = cv_and_test(lgbm, X, y, tr, te, folds); rep("plain", oof0,p0,tr,te)
    for wr, wb in [(2,0.5),(3,0.5),(3,1)]:
        w = np.ones(len(y)); w[reliable]=wr; w[bad]=wb
        o,p,_ = cv_and_test(lgbm, X, y, tr, te, folds, w); rep(f"weights reliable={wr} conflicting={wb}", o,p,tr,te)
    o,p,_ = cv_and_test(lambda: lgbm(objective="huber", alpha=1.0), X, y, tr, te, folds); rep("huber(1.0)", o,p,tr,te)
    # self-cleaning: down-weight training points whose OOF residual is large (single-measurement only)
    for c in [1.5, 2.0]:
        w = np.ones(len(y)); r = np.abs(oof0 - y[tr]); ww = np.minimum(1, (c/np.maximum(r,1e-9))**2); ww[reliable[tr]] = 1
        w[tr] = ww; w[reliable] = 2
        # honest CV: weights must come from inner info; for test evaluation it is clean (weights from train OOF only)
        m = lgbm().fit(X[tr], y[tr], sample_weight=w[tr]); p = m.predict(X[te]); rep(f"self-clean c={c} (+reliable x2) [test only]", oof0, p, tr, te)
    o,p,_ = cv_and_test(lambda: lgbm(n_estimators=3000, learning_rate=0.015, num_leaves=63, colsample_bytree=0.2), X, y, tr, te, folds); rep("bigger lgbm", o,p,tr,te)
