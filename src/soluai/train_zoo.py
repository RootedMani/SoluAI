"""Train the five v2 models on *all* merged molecules and save them to ``models/zoo-v2.0.0``.

    python -m soluai.train_zoo          (about 10 minutes on 2 CPU cores, needs ~2 GB RAM)

The honest test scores were measured beforehand on held-out molecules, in
``experiments/v2-consensus-stacking`` (``zoo_metrics.py``). Those scores and the consensus
weights are copied from there; this script only builds the final models that the website uses.
"""

from __future__ import annotations

import json
import logging
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR

from . import ROOT
from .zoo import DESCRIPTOR_LIST, MEMBERS, ZOO_DIR, boosting_matrix, svr_kernel, v2_feature_matrix

log = logging.getLogger(__name__)
EXP = ROOT / "experiments" / "v2-consensus-stacking"
SEED = 42


def boosting(objective: str = "regression") -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=31, subsample=0.9, subsample_freq=1,
                             colsample_bytree=0.3, min_child_samples=10, objective=objective, verbose=-1,
                             n_jobs=2, random_state=SEED)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = pd.read_csv(EXP / "data" / "merged.csv")
    metrics = json.loads((EXP / "results" / "zoo_metrics.json").read_text())
    y = data["logS"].to_numpy()
    log.info("Featurising %d molecules", len(data))
    F = v2_feature_matrix([Chem.MolFromSmiles(s) for s in data["smiles"]])
    B = boosting_matrix(F)
    ZOO_DIR.mkdir(parents=True, exist_ok=True)

    t = time.time()
    joblib.dump({"lgbm": boosting().fit(B, y), "huber": boosting("huber").fit(B, y)}, ZOO_DIR / "boosting.joblib", compress=3)
    log.info("LightGBM x2 done (%.0f s)", time.time() - t)

    from catboost import CatBoostRegressor

    t = time.time()
    cat = CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=6, loss_function="RMSE", verbose=0,
                            thread_count=2, random_seed=SEED).fit(B, y)
    cat.save_model(str(ZOO_DIR / "catboost.cbm"))
    log.info("CatBoost done (%.0f s)", time.time() - t)

    t = time.time()
    K = svr_kernel(F, F)
    svr = SVR(kernel="precomputed", C=5, epsilon=0.1).fit(K, y)
    del K
    sv = svr.support_
    np.savez_compressed(ZOO_DIR / "svr.npz", sv_mcount=F["mcount"][sv], sv_ap=F["ap"][sv],
                        dual_coef=svr.dual_coef_[0].astype(np.float64), intercept=np.float64(svr.intercept_[0]))
    log.info("SVR done: %d support molecules (%.0f s)", len(sv), time.time() - t)

    t = time.time()
    keep = F["full"].std(0) > 0
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0).fit(F["full"][:, keep])
    X = np.hstack([qt.transform(F["full"][:, keep]), F["maccs"].astype(np.float64)])
    nets = []
    for s in range(3):
        m = MLPRegressor(hidden_layer_sizes=(512, 256), alpha=1e-2, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, n_iter_no_change=15, random_state=s).fit(X, y)
        nets.append([(W.astype(np.float32), b.astype(np.float32)) for W, b in zip(m.coefs_, m.intercepts_)])
    joblib.dump({"qt": qt, "keep": keep, "nets": nets}, ZOO_DIR / "mlp.joblib", compress=3)
    log.info("Neural networks done (%.0f s)", time.time() - t)

    weights = metrics["splits"]["random"]["weights"]
    (ZOO_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))
    import catboost, rdkit, sklearn

    (ZOO_DIR / "zoo.json").write_text(json.dumps({
        "version": "2.0.0", "members": MEMBERS, "weights": weights, "n_training_molecules": len(data),
        "training_data": "experiments/v2-consensus-stacking/data/merged.csv",
        "packages": {"rdkit": rdkit.__version__, "lightgbm": lgb.__version__, "catboost": catboost.__version__,
                     "scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "descriptors": DESCRIPTOR_LIST,
    }, indent=1))
    log.info("Saved to %s", ZOO_DIR)


if __name__ == "__main__":
    main()
