"""The "human" baseline: Delaney's ESOL equation (2004).

John Delaney fitted a simple straight-line formula to ~2,800 measured
solubilities using four ideas a chemist already knows matter:

    logS = 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*AP

clogP = oiliness, MW = molecular weight, RB = rotatable bonds,
AP = aromatic proportion. We use RDKit's Crippen logP instead of the
commercial clogP program Delaney used, so the numbers differ slightly.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from sklearn.linear_model import LinearRegression

from .features import aromatic_proportion

DELANEY_COEF = {"intercept": 0.16, "logP": -0.63, "MW": -0.0062, "RB": 0.066, "AP": -0.74}


def delaney_variables(mol: Chem.Mol) -> np.ndarray:
    """The four ESOL variables: [logP, MW, rotatable bonds, aromatic proportion]."""
    return np.array(
        [Crippen.MolLogP(mol), Descriptors.MolWt(mol), Lipinski.NumRotatableBonds(mol), aromatic_proportion(mol)]
    )


def delaney_matrix(smiles) -> np.ndarray:
    """ESOL variables for many SMILES (one row each)."""
    return np.vstack([delaney_variables(Chem.MolFromSmiles(s)) for s in smiles])


def delaney_predict(smiles) -> np.ndarray:
    """Delaney's original equation with his published coefficients."""
    X = delaney_matrix(smiles)
    c = DELANEY_COEF
    return c["intercept"] + c["logP"] * X[:, 0] + c["MW"] * X[:, 1] + c["RB"] * X[:, 2] + c["AP"] * X[:, 3]


def delaney_one(mol: Chem.Mol) -> float:
    """Delaney's equation for a single molecule (shown next to the AI's answer on the website)."""
    x = delaney_variables(mol)
    c = DELANEY_COEF
    return float(c["intercept"] + c["logP"] * x[0] + c["MW"] * x[1] + c["RB"] * x[2] + c["AP"] * x[3])


def fit_delaney_refit(smiles, y) -> LinearRegression:
    """Same four variables, but coefficients re-fitted on our training data."""
    return LinearRegression().fit(delaney_matrix(smiles), y)
