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
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
