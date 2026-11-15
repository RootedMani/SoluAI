"""Question: the deployed referee learns from two cross-validations at once (scaffold folds and
random folds). Do its 90 % ranges still hold on held-out test molecules of both kinds?

For each test split, everything is fitted on the training part only, exactly as train_final()
does on the full data, and then checked once on the test part.
Exploratory code: see research/README.md. Needs no cache; ~10 min.
"""
import json

import numpy as np
from rdkit import Chem

from soluai import MODELS_DIR
from soluai.config import load_config
from soluai.data import load_clean
from soluai.features import Featurizer
from soluai.models import make_model
from soluai.splits import cv_splitter, random_split, scaffold_split
from soluai.train import V2_FEATURES, cross_validate, fit_trust
from soluai.uncertainty import trust_report

cfg = load_config()
seed, n_bits = cfg["seed"], cfg["features"]["morgan_bits"]
params = json.loads((MODELS_DIR / (MODELS_DIR / "LATEST").read_text().strip() / "metadata.json").read_text())["params"]
df = load_clean("aqsoldb")
smiles, y = df["smiles"].tolist(), df["logS"].to_numpy()
X = Featurizer(V2_FEATURES, cfg["features"]["morgan_radius"], n_bits).transform(smiles)
for split in ["random", "scaffold"]:
    tr, te = random_split(len(df), 0.2, seed) if split == "random" else scaffold_split(smiles, 0.2, seed)
    tr_smiles = [smiles[i] for i in tr]
    runs = []
    for kind in ["scaffold", "random"]:
        folds = cv_splitter(kind, cfg["split"]["cv_folds"], seed, tr_smiles)
        runs.append((cross_validate("lgbm", params, X[tr], y[tr], folds, seed)[1], folds))
    pred = make_model("lgbm", params, seed).fit(X[tr], y[tr]).predict(X[te])
    for label, r in [("one exam (matching folds)", [runs[0] if split == "scaffold" else runs[1]]), ("two exams (as deployed)", runs)]:
        trust, _, _, signals = fit_trust(cfg, X[tr], y[tr], r, n_bits)
        rep = trust_report(y[te], pred, trust.half_width(signals(X[te], pred)), trust)
        print(f"{split:8s} test | referee trained on {label:26s} q={trust.q:.2f} coverage {rep['all']['coverage']:.3f} "
              f"mean ±{rep['all']['mean_half_width']:.2f} | " +
              " | ".join(f"{t}: {v['share']:.0%} of test, within x10 {v['within_1']:.1%}, coverage {v['coverage']:.1%}"
                         for t, v in rep["tiers"].items() if v), flush=True)
