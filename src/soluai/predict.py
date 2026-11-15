"""Load the saved model and make predictions."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from . import MODELS_DIR
from .features import DESCRIPTOR_NAMES, Featurizer, descriptors
from .uncertainty import TrustModel, trust_signals

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
        self.featurizer = Featurizer(bundle["feature_kind"], bundle["radius"], bundle["n_bits"], bundle.get("extra_names"))
        self.feature_names: list[str] = bundle["feature_names"]
        self.conformal: dict = bundle["conformal"]
        self.descriptor_ranges: dict[str, list[float]] = bundle["descriptor_ranges"]
        self.shap_slopes: dict[str, float] = bundle.get("shap_slopes", {})
        # Model v2 has a "referee" that sizes the range for each molecule; v1 models do not.
        self.trust: TrustModel | None = bundle.get("trust")
        if self.trust is not None:
            self.trust.model.set_params(n_jobs=1)

    # -- training set (for "nearest neighbours" and similarity) -------------------
    @cached_property
    def training_set(self) -> pd.DataFrame:
        """The molecules the model learned from."""
        return pd.read_csv(self.model_dir / "training_set.csv")

    @cached_property
    def training_fps(self) -> list:
        """Morgan fingerprints of the training molecules (computed once)."""
        return [self.featurizer.fingerprint(Chem.MolFromSmiles(s)) for s in self.training_set["smiles"]]

    @cached_property
    def _measured_index(self) -> tuple[dict[str, int], dict[str, int]]:
        """Row of each training molecule by SMILES: exact, and with 3D details (stereo) ignored."""
        exact: dict[str, int] = {}
        flat: dict[str, int] = {}
        for i, smi in enumerate(self.training_set["smiles"]):
            exact.setdefault(smi, i)
            if any(ch in smi for ch in "@/\\"):
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
            flat.setdefault(smi, i)
        return exact, flat

    def measured(self, mol: Chem.Mol) -> dict | None:
        """The lab-measured logS of this molecule, if it is in the data the model learned from."""
        exact, flat = self._measured_index
        i = exact.get(Chem.MolToSmiles(mol))
        is_exact = i is not None
        if i is None:
            i = flat.get(Chem.MolToSmiles(mol, isomericSmiles=False))
        if i is None:
            return None
        ts = self.training_set
        return {"logS": round(float(ts.at[i, "logS"]), 2), "name": str(ts.at[i, "name"]).strip(), "exact": is_exact}

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

    def domain(self, mol: Chem.Mol, k: int = 3, pred: float | None = None) -> dict:
        """Similarity to the training data, the 90 % range's half-width, confidence and nearest neighbours.

        With model v2 the half-width comes from the trust model (the "referee"), which looks
        at the closest training molecules and at the prediction itself (``pred``; computed
        here if not given). Older models use one fixed width for familiar molecules and one
        for unfamiliar ones.
        """
        fp = self.featurizer.fingerprint(mol)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, self.training_fps))
        top = np.argsort(-sims)[:k]
        nn = [(int(i), float(sims[i])) for i in top]
        max_sim = nn[0][1]
        thr = self.conformal["threshold"]
        inside = max_sim >= thr
        desc = descriptors(mol)
        out_of_range = [
            n for n, v in zip(DESCRIPTOR_NAMES, desc)
            if not (self.descriptor_ranges[n][0] <= v <= self.descriptor_ranges[n][1])
        ]
        ts = self.training_set
        out = {
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
        if self.trust is not None:
            d = dict(zip(DESCRIPTOR_NAMES, desc))
            pred = self.predict_mol(mol) if pred is None else pred
            signals = trust_signals(sims, self._training_logS, self._training_abs_err, np.array([pred]),
                                    d["MolLogP"], d["MolWt"], d["HeavyAtomCount"], self.trust.k)
            sigma = float(self.trust.sigma(signals)[0])
            hw = self.trust.q * sigma
            # Safety net: the referee has seen very few molecules this far from the training data,
            # so outside the applicability domain the label is never better than "low".
            label = self.trust.tier(hw) if out["in_domain"] else "low"
            out.update({"half_width": hw, "expected_error": round(sigma, 2), "confidence": label})
        return out

    @cached_property
    def _training_logS(self) -> np.ndarray:
        return self.training_set["logS"].to_numpy(float)

    @cached_property
    def _training_abs_err(self) -> np.ndarray:
        """How wrong the model was on each training molecule when it had not seen it (cross-validation)."""
        return self.training_set["oof_abs_err"].to_numpy(float)
