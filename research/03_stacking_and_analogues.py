"""Question: do other learners, stacking, or a correction from similar measured molecules help?

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
# other learners + stacking + analogue (kNN residual) correction
from common import *
import pickle, itertools
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
df, d = load("aqsoldb"); y = d["y"]
X0, R, C, M = d["X0"], d["R"], d["C"], d["M"]
B = (X0[:,14:]>0).astype(np.float32)
def tanimoto(A, Bm):
    inter = A @ Bm.T; na = A.sum(1)[:,None]; nb = Bm.sum(1)[None,:]
    return inter/np.maximum(na+nb-inter,1)
K = tanimoto(B,B)
e1 = pickle.load(open(str(CACHE / "e1.pkl"),"rb"))
FEAT = "rdkit217+fp1024"
res = {}
for sp in ["random","scaffold"]:
    tr, te, folds = split(df, sp); ytr, yte = y[tr], y[te]
    P = {}; 
    P["lgbm"] = e1[(sp,FEAT)]
    # SVR with Tanimoto kernel
    Ktr = K[np.ix_(tr,tr)]; Kte = K[np.ix_(te,tr)]
    oof = np.zeros(len(tr))
    for a,b in folds:
        m = SVR(kernel="precomputed", C=5, epsilon=0.1).fit(Ktr[np.ix_(a,a)], ytr[a]); oof[b] = m.predict(Ktr[np.ix_(b,a)])
    m = SVR(kernel="precomputed", C=5, epsilon=0.1).fit(Ktr, ytr); P["svr_tani"] = (oof, m.predict(Kte))
    # SVR RBF on quantile-scaled rdkit descriptors
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0), SVR(C=10, epsilon=0.1, gamma=1/400))
    Rc = R[:, np.nanstd(R,0)>0]
    P["svr_desc"] = cv_and_test(mk, Rc, y, tr, te, folds)[:2]
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(500, max_features=0.33, min_samples_leaf=1, n_jobs=4, random_state=0))
    P["et"] = cv_and_test(mk, np.hstack([R, M]), y, tr, te, folds)[:2]
    for k,(o,p) in P.items(): print(f"{sp:9s} {k:10s} CV {rmse(o,ytr):.3f} test {rmse(p,yte):.3f} R2 {r2(yte,p):.3f}", flush=True)
    # stack (non-negative-ish linear blend fitted on OOF)
    keys = list(P); O = np.column_stack([P[k][0] for k in keys]); T = np.column_stack([P[k][1] for k in keys])
    lr = LinearRegression(positive=True).fit(O, ytr); ps = lr.predict(T); os_ = np.zeros(len(tr))
    for a,b in folds: os_[b] = LinearRegression(positive=True).fit(O[a], ytr[a]).predict(O[b])
    print(f"{sp:9s} STACK {dict(zip(keys, lr.coef_.round(2)))} CV {rmse(os_,ytr):.3f} test {rmse(ps,yte):.3f} R2 {r2(yte,ps):.3f} within1 {np.mean(np.abs(ps-yte)<=1):.3f} within0.5 {np.mean(np.abs(ps-yte)<=.5):.3f}", flush=True)
    # analogue correction on top of stack
    def corr(Ks, resid, tau, p, kappa, k=5):
        Ks = Ks.copy(); idx = np.argsort(-Ks, 1)[:, :k]; s = np.take_along_axis(Ks, idx, 1); w = np.where(s>=tau, s**p, 0)
        return (w*resid[idx]).sum(1)/(w.sum(1)+kappa)
    r_tr = ytr - os_
    Kcv = Ktr.copy()
    for a,b in folds: Kcv[np.ix_(b,b)] = -1   # only neighbours from other folds
    best=None
    for tau,p,kappa,k in itertools.product([0.4,0.5,0.6,0.7],[1,3],[0.25,0.5,1.0],[3,5,10]):
        s = rmse(os_+corr(Kcv, r_tr, tau,p,kappa,k), ytr)
        if best is None or s<best[0]: best=(s,tau,p,kappa,k)
    s,tau,p,kappa,k = best
    pc = ps + corr(Kte, r_tr, tau,p,kappa,k)
    print(f"{sp:9s} STACK+ANALOGUE best{best} test {rmse(pc,yte):.3f} R2 {r2(yte,pc):.3f} within1 {np.mean(np.abs(pc-yte)<=1):.3f}", flush=True)
    res[sp] = dict(P=P, stack=(os_,ps), final=pc, tr=tr, te=te)
pickle.dump(res, open(str(CACHE / "e2.pkl"),"wb"))
