"""Delaney's ESOL equation."""

import pytest

from soluai.baselines import DELANEY_COEF, delaney_predict, delaney_variables
from rdkit import Chem


def test_delaney_by_hand_for_benzene():
    logp, mw, rb, ap = delaney_variables(Chem.MolFromSmiles("c1ccccc1"))
    c = DELANEY_COEF
    expected = c["intercept"] + c["logP"] * logp + c["MW"] * mw + c["RB"] * rb + c["AP"] * ap
    assert delaney_predict(["c1ccccc1"])[0] == pytest.approx(expected)
    # Measured logS of benzene is about -1.6; the equation should be in the right area.
    assert -3 < expected < -1


def test_delaney_trend_alcohols():
    p = delaney_predict(["CCO", "CCCCO", "CCCCCCCCO"])
    assert p[0] > p[1] > p[2]
