"""Shared helpers for the exploratory experiments (same splits and folds as the real pipeline).

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from pathlib import Path

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from soluai.data import load_clean
from soluai.splits import random_split, scaffold_split, cv_splitter
warnings.filterwarnings("ignore")
def load(name):
    d = np.load(str(CACHE / f"{name}.npz"), allow_pickle=True); df = load_clean(name)
    return df, d
def split(df, kind, seed=42):
    sm = df.smiles.tolist()
    tr, te = random_split(len(df), .2, seed) if kind=="random" else scaffold_split(sm, .2, seed)
    folds = cv_splitter(kind, 5, seed, [sm[i] for i in tr])
    return tr, te, folds
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def r2(y,p): return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
def lgbm(**kw):
    p = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.3, reg_lambda=5.0, random_state=42, n_jobs=4, verbose=-1, deterministic=True, force_col_wise=True)
    p.update(kw); return lgb.LGBMRegressor(**p)
def cv_and_test(make, X, y, tr, te, folds, w=None):
    Xtr, ytr = X[tr], y[tr]; oof = np.zeros(len(tr))
    for a, b in folds:
        m = make(); m.fit(Xtr[a], ytr[a], **({"sample_weight": w[tr][a]} if w is not None else {})); oof[b] = m.predict(Xtr[b])
    m = make(); m.fit(Xtr, ytr, **({"sample_weight": w[tr]} if w is not None else {}))
    return oof, m.predict(X[te]), m
