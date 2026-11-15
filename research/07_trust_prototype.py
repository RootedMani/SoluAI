"""Question: can a second model (the referee) predict how large the error will be?
This prototype became soluai.uncertainty.TrustModel.

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import *
import pickle
from sklearn.model_selection import KFold
df, d = load("aqsoldb"); y = d["y"]; X0 = d["X0"]; B = (X0[:,14:]>0).astype(np.float32)
inter = B@B.T; n = B.sum(1); K = inter/np.maximum(n[:,None]+n[None,:]-inter,1)
e1 = pickle.load(open(str(CACHE / "e1.pkl"),"rb"))
def signals(Ks, y_ref, absres_ref, pred, desc, k=5):
    idx = np.argsort(-Ks,1)[:,:k]; s = np.take_along_axis(Ks, idx, 1); w = s+1e-6
    yn = y_ref[idx]; mu = (w*yn).sum(1)/w.sum(1)
    return np.column_stack([s[:,0], s.mean(1), np.sqrt((w*(yn-mu[:,None])**2).sum(1)/w.sum(1)), (w*absres_ref[idx]).sum(1)/w.sum(1),
                            np.abs(pred-mu), pred, desc[:,0], desc[:,1], desc[:,10]])
names = ["max_sim","mean_sim5","nb_std","nb_err","nb_gap","pred","logP","MolWt","heavy"]
def sig_model(): return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, random_state=0, verbose=-1, n_jobs=4)
for sp in ["random","scaffold"]:
    tr, te, folds = split(df, sp); ytr, yte = y[tr], y[te]; oof, pt = e1[(sp,"rdkit217+fp1024")]
    a_tr = np.abs(oof-ytr); Ktr = K[np.ix_(tr,tr)].copy()
    for a,b in folds: Ktr[np.ix_(b,b)] = -1
    S_tr = signals(Ktr, ytr, a_tr, oof, X0[tr]); S_te = signals(K[np.ix_(te,tr)], ytr, a_tr, pt, X0[te])
    # cross-fitted sigma on train (same folds), then conformal q
    sig_oof = np.zeros(len(tr))
    for a,b in folds: sig_oof[b] = sig_model().fit(S_tr[a], a_tr[a]).predict(S_tr[b])
    sig_oof = np.maximum(sig_oof, 0.15); m = sig_model().fit(S_tr, a_tr); sig_te = np.maximum(m.predict(S_te), 0.15)
    score = a_tr/sig_oof; nn = len(score); q = np.quantile(score, min(1, np.ceil((nn+1)*.9)/nn), method="higher")
    err = np.abs(pt-yte); hw = q*sig_te
    qfix = np.quantile(a_tr, .9)
    print(f"\n{sp}: fixed ±{qfix:.2f} coverage {np.mean(err<=qfix):.3f} | adaptive q={q:.2f} coverage {np.mean(err<=hw):.3f} mean hw {hw.mean():.2f} median hw {np.median(hw):.2f}  corr(sig,err) {np.corrcoef(sig_te,err)[0,1]:.2f}")
    print(" importance", dict(zip(names, m.feature_importances_)))
    order = np.argsort(sig_te)
    for frac in [0.2,0.33,0.5,0.67,1.0]:
        i = order[:int(frac*len(order))]
        print(f"  most-confident {int(frac*100):3d}% n={len(i):4d} rmse {rmse(pt[i],yte[i]):.3f} R2 {r2(yte[i],pt[i]):.3f} within1 {np.mean(err[i]<=1):.3f} within0.5 {np.mean(err[i]<=.5):.3f} cov {np.mean(err[i]<=hw[i]):.3f} mean hw {hw[i].mean():.2f}")
    ter = np.quantile(sig_oof,[1/3,2/3])
    for lab,mk in [("high",sig_te<=ter[0]),("medium",(sig_te>ter[0])&(sig_te<=ter[1])),("low",sig_te>ter[1])]:
        print(f"  tier {lab:6s} n={mk.sum():4d} rmse {rmse(pt[mk],yte[mk]):.3f} R2 {r2(yte[mk],pt[mk]):.3f} within1 {np.mean(err[mk]<=1):.3f} cov {np.mean(err[mk]<=hw[mk]):.3f}")
    # old-style domain flag comparison
    ms = S_te[:,0]; print("  old flag: sim>=0.3 rmse %.3f (n=%d) ; <0.3 rmse %.3f (n=%d)"%(rmse(pt[ms>=.3],yte[ms>=.3]),(ms>=.3).sum(),rmse(pt[ms<.3],yte[ms<.3]),(ms<.3).sum()))
