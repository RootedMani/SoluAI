"""How sure is the model?  Two simple, honest tools.

1. **Applicability domain** - we measure how similar a new molecule is to the
   *most similar* molecule the model was trained on (Tanimoto similarity of
   Morgan fingerprints, 0 = nothing in common, 1 = identical). If even the
   closest training molecule is not very similar, the molecule is "outside the
   model's experience" and the prediction should be trusted less.

2. **Conformal 90 % interval** - we look at the model's errors on molecules it
   did *not* see during training (cross-validation). If 90 % of those errors
   were smaller than ``q``, we report ``prediction ± q``. Because errors are
   larger for unfamiliar molecules, ``q`` is computed separately for
   "familiar" and "unfamiliar" molecules (a so-called Mondrian conformal
   predictor). This needs no assumptions about the shape of the errors.

Model v2 replaces the single "familiar / unfamiliar" switch with a second,
small model - the **referee** (:class:`TrustModel`):

3. **Trust model** - for every molecule we compute nine *trust signals*
   (how similar are its closest training molecules, do those neighbours agree
   with each other, was the main model wrong on them, does the prediction
   agree with them, how big and oily is the molecule ...). The referee learns,
   from honest out-of-fold errors, how large the main model's error usually
   is for a given set of signals. Its answer ``sigma`` (expected error) scales
   the conformal interval: ``prediction ± q * sigma``. Easy molecules get a
   narrow range, hard ones a wide range, and on average 90 % are still covered.
   See ``docs/algorithm.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from rdkit import DataStructs


def max_similarity(query_fps, ref_fps) -> np.ndarray:
    """For every query fingerprint, the highest Tanimoto similarity to ``ref_fps``."""
    return np.array([max(DataStructs.BulkTanimotoSimilarity(q, ref_fps)) for q in query_fps])


def nearest_neighbours(query_fp, ref_fps, k: int = 3) -> list[tuple[int, float]]:
    """Indices and similarities of the ``k`` most similar reference molecules."""
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps))
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top]


def conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample-corrected conformal quantile of absolute residuals."""
    n = len(abs_residuals)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(abs_residuals, level, method="higher"))


@dataclass
class ConformalByDomain:
    """Separate interval half-widths for in-domain and out-of-domain molecules."""

    threshold: float
    coverage: float = 0.9
    q: dict[str, float] = field(default_factory=dict)

    def fit(self, abs_residuals: np.ndarray, similarity: np.ndarray) -> "ConformalByDomain":
        """Learn the half-widths from out-of-fold errors and similarities."""
        inside = similarity >= self.threshold
        self.q = {
            "in": conformal_quantile(abs_residuals[inside], self.coverage),
            "out": conformal_quantile(abs_residuals[~inside], self.coverage) if (~inside).sum() >= 20
            else conformal_quantile(abs_residuals, self.coverage),
        }
        return self

    def half_width(self, similarity: np.ndarray | float) -> np.ndarray:
        """Interval half-width for each molecule given its max similarity."""
        sim = np.atleast_1d(similarity)
        return np.where(sim >= self.threshold, self.q["in"], self.q["out"])


# ----------------------------------------------------------------------------- trust model (v2)
#: The nine trust signals, in column order, with a plain-language meaning.
TRUST_SIGNALS: dict[str, str] = {
    "max_sim": "similarity of the closest training molecule",
    "mean_sim": "average similarity of the 5 closest training molecules",
    "nb_spread": "how much the measured logS of those neighbours disagree",
    "nb_error": "how wrong the model was on those neighbours (when it had not seen them)",
    "nb_gap": "distance between the prediction and the neighbours' average logS",
    "prediction": "the predicted logS itself (very insoluble molecules are harder)",
    "logP": "oiliness",
    "MolWt": "molecular weight",
    "heavy_atoms": "number of non-hydrogen atoms",
}


def tanimoto_matrix(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of every row of ``bits_a`` (0/1) with every row of ``bits_b``."""
    a, b = bits_a.astype(np.float32), bits_b.astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


def trust_signals(sims: np.ndarray, y_ref: np.ndarray, abs_err_ref: np.ndarray, pred: np.ndarray,
                  logp: np.ndarray, mol_wt: np.ndarray, heavy: np.ndarray, k: int = 5) -> np.ndarray:
    """The nine trust signals for each query molecule.

    ``sims[i, j]`` is the similarity of query ``i`` to reference (training) molecule ``j``;
    set it to ``-1`` to forbid a reference (e.g. molecules of the same CV fold).
    ``abs_err_ref`` holds the out-of-fold absolute error of each reference molecule.
    """
    sims = np.atleast_2d(sims)
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    s = np.clip(np.take_along_axis(sims, idx, 1), 0.0, 1.0)
    w = s + 1e-6
    yn = y_ref[idx]
    mean = (w * yn).sum(1) / w.sum(1)
    spread = np.sqrt((w * (yn - mean[:, None]) ** 2).sum(1) / w.sum(1))
    nb_err = (w * abs_err_ref[idx]).sum(1) / w.sum(1)
    pred = np.atleast_1d(pred)
    return np.column_stack([s.max(1), s.mean(1), spread, nb_err, np.abs(pred - mean), pred,
                            np.atleast_1d(logp), np.atleast_1d(mol_wt), np.atleast_1d(heavy)])


def _referee(params: dict, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(random_state=seed, n_jobs=4, subsample_freq=1, deterministic=True,
                             force_col_wise=True, verbose=-1, **params)


#: A deliberately small model: the referee has only 9 inputs and a noisy target.
REFEREE_PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 40,
                  "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0}


@dataclass
class TrustModel:
    """The referee: predicts the main model's expected error and scales the 90 % range with it."""

    coverage: float = 0.9
    k: int = 5                      # neighbours used for the signals
    floor: float = 0.15             # smallest expected error we are willing to claim (log units)
    tiers: dict[str, float] = field(default_factory=lambda: {"high": 0.8, "medium": 1.4})
    params: dict = field(default_factory=lambda: dict(REFEREE_PARAMS))
    seed: int = 42
    q: float = 0.0
    model: lgb.LGBMRegressor | None = None

    def fit(self, signals: np.ndarray, abs_err: np.ndarray, folds) -> np.ndarray:
        """Learn expected error from out-of-fold errors. Returns the cross-fitted ``sigma``.

        ``sigma`` for each molecule comes from a referee that never saw that molecule,
        so the conformal factor ``q`` computed from ``abs_err / sigma`` is honest.
        """
        sigma = np.zeros(len(abs_err))
        for tr, va in folds:
            sigma[va] = _referee(self.params, self.seed).fit(signals[tr], abs_err[tr]).predict(signals[va])
        sigma = np.maximum(sigma, self.floor)
        self.q = conformal_quantile(abs_err / sigma, self.coverage)
        self.model = _referee(self.params, self.seed).fit(signals, abs_err)
        return sigma

    def sigma(self, signals: np.ndarray) -> np.ndarray:
        """Expected absolute error (log units) for each row of signals."""
        return np.maximum(self.model.predict(np.atleast_2d(signals)), self.floor)

    def half_width(self, signals: np.ndarray) -> np.ndarray:
        """Half-width of the range that should contain the true value with ``coverage`` probability."""
        return self.q * self.sigma(signals)

    def tier(self, half_width: float) -> str:
        """Confidence label from the width of the range: ``"high"``, ``"medium"`` or ``"low"``."""
        if half_width <= self.tiers["high"]:
            return "high"
        return "medium" if half_width <= self.tiers["medium"] else "low"


def trust_report(y: np.ndarray, pred: np.ndarray, half_width: np.ndarray, trust: TrustModel,
                 reliable: np.ndarray | None = None) -> dict:
    """How well do the adaptive ranges and confidence labels work on held-out molecules?"""
    err = np.abs(pred - y)

    def part(mask: np.ndarray) -> dict | None:
        if mask.sum() < 5:
            return None
        e, yy = err[mask], y[mask]
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        return {"n": int(mask.sum()), "share": round(float(mask.mean()), 3),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
                "r2": round(1 - float((e ** 2).sum()) / ss_tot, 3) if ss_tot > 0 else None,
                "within_1": round(float((e <= 1.0).mean()), 3), "within_0.5": round(float((e <= 0.5).mean()), 3),
                "coverage": round(float((e <= half_width[mask]).mean()), 3),
                "mean_half_width": round(float(half_width[mask].mean()), 3)}

    labels = np.array([trust.tier(h) for h in half_width])
    order = np.argsort(half_width)
    curve = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        m = np.zeros(len(y), bool)
        m[order[: max(5, int(round(frac * len(y))))]] = True
        curve.append({"fraction": frac, **{k: v for k, v in part(m).items() if k in ("rmse", "r2", "within_1")}})
    out = {
        "q": round(trust.q, 3), "all": part(np.ones(len(y), bool)),
        "median_half_width": round(float(np.median(half_width)), 3),
        "tiers": {t: part(labels == t) for t in ("high", "medium", "low")},
        "confidence_curve": curve,
        "error_vs_expected_correlation": round(float(np.corrcoef(half_width, err)[0, 1]), 3),
    }
    if reliable is not None and reliable.sum() >= 5:
        out["reliable_measurements"] = part(reliable)
        out["single_measurements"] = part(~reliable)
    return out
