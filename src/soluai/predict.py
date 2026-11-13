"""Load the saved model and make predictions."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

from . import MODELS_DIR
from .features import DESCRIPTOR_NAMES, Featurizer, descriptors
from .uncertainty import nearest_neighbours

#: Solubility classes used by AqSolDB (Sorkun et al. 2019), in log(mol/L).
CATEGORIES = [
    (0.0, "Highly soluble"),
    (-2.0, "Soluble"),
    (-4.0, "Slightly soluble"),
    (-np.inf, "Insoluble"),
]


def category(logS: float) -> str:
    """Solubility class for a logS value."""
    for cutoff, label in CATEGORIES:
        if logS >= cutoff:
            return label
    return CATEGORIES[-1][1]


def latest_model_dir() -> Path:
    """Folder of the newest saved model (written to ``models/LATEST`` by training)."""
    return MODELS_DIR / (MODELS_DIR / "LATEST").read_text().strip()


class Predictor:
    """A trained SoluAI model, ready to predict.

    >>> p = Predictor()
    >>> p.predict_many(["CCO", "CCCCCC"])  # ethanol is more soluble than hexane
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else latest_model_dir()
        bundle = joblib.load(self.model_dir / "model.joblib")
        self.model = bundle["model"]
        # One molecule at a time needs no parallel threads (and the free web server has <1 CPU).
        self.model.set_params(n_jobs=1)
        self.version: str = bundle["version"]
        self.featurizer = Featurizer(bundle["feature_kind"], bundle["radius"], bundle["n_bits"])
        self.feature_names: list[str] = bundle["feature_names"]
        self.conformal: dict = bundle["conformal"]
        self.descriptor_ranges: dict[str, list[float]] = bundle["descriptor_ranges"]
        self.shap_slopes: dict[str, float] = bundle.get("shap_slopes", {})

    # -- training set (for "nearest neighbours" and similarity) -------------------
    @cached_property
    def training_set(self) -> pd.DataFrame:
        """The molecules the model learned from."""
        return pd.read_csv(self.model_dir / "training_set.csv")

    @cached_property
    def training_fps(self) -> list:
        """Morgan fingerprints of the training molecules (computed once)."""
        return [self.featurizer.fingerprint(Chem.MolFromSmiles(s)) for s in self.training_set["smiles"]]

    # -- predictions -------------------------------------------------------------
    def features(self, mol: Chem.Mol) -> np.ndarray:
        """Feature row (1 x n) for a molecule."""
        return self.featurizer.transform_mol(mol).reshape(1, -1)

    def predict_mol(self, mol: Chem.Mol) -> float:
        """Predicted logS (mol/L) of one RDKit molecule."""
        return float(self.model.predict(self.features(mol))[0])

    def predict_many(self, smiles: list[str]) -> np.ndarray:
        """Predicted logS for a list of SMILES."""
        return self.model.predict(self.featurizer.transform(smiles))

    def shap_values(self, mol: Chem.Mol) -> tuple[np.ndarray, float]:
        """Exact SHAP values (LightGBM TreeSHAP) and the base value (average prediction)."""
        contrib = self.model.predict(self.features(mol), pred_contrib=True)[0]
        return contrib[:-1], float(contrib[-1])

    def domain(self, mol: Chem.Mol, k: int = 3) -> dict:
        """Similarity to the training data, interval half-width and nearest neighbours."""
        fp = self.featurizer.fingerprint(mol)
        nn = nearest_neighbours(fp, self.training_fps, k)
        max_sim = nn[0][1]
        thr = self.conformal["threshold"]
        inside = max_sim >= thr
        desc = descriptors(mol)
        out_of_range = [
            n for n, v in zip(DESCRIPTOR_NAMES, desc)
            if not (self.descriptor_ranges[n][0] <= v <= self.descriptor_ranges[n][1])
        ]
        ts = self.training_set
        return {
            "max_similarity": round(max_sim, 3),
            "in_domain": bool(inside and not out_of_range),
            "out_of_range_descriptors": out_of_range,
            "half_width": self.conformal["q"]["in" if inside else "out"],
            "neighbours": [
                {"name": str(ts.at[i, "name"]), "smiles": ts.at[i, "smiles"],
                 "logS": round(float(ts.at[i, "logS"]), 2), "similarity": round(s, 3)}
                for i, s in nn
            ],
        }
