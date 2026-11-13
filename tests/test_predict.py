"""Prediction sanity: does the trained model agree with basic chemistry?"""

import numpy as np
from rdkit import Chem

from soluai.explain import analyze
from soluai.predict import category


def test_category_thresholds():
    assert category(0.5) == "Highly soluble"
    assert category(-1) == "Soluble"
    assert category(-3) == "Slightly soluble"
    assert category(-6) == "Insoluble"


def test_ethanol_more_soluble_than_hexane(predictor):
    eth, hex_ = predictor.predict_many(["CCO", "CCCCCC"])
    assert eth > hex_ + 1


def test_alcohol_series_trend(predictor):
    p = predictor.predict_many(["CCO", "CCCCO", "CCCCCCO", "CCCCCCCCO"])
    assert np.all(np.diff(p) < 0)


def test_shap_values_add_up(predictor):
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    shap, base = predictor.shap_values(mol)
    # SHAP values always add up exactly: average prediction + pushes = this prediction.
    assert abs(base + shap.sum() - predictor.predict_mol(mol)) < 1e-6


def test_analyze_output(predictor):
    r = analyze(predictor, Chem.MolFromSmiles("OCC(O)CO"))  # glycerol
    assert r["interval"][0] < r["logS"] < r["interval"][1]
    assert r["svg"].startswith("<svg") and r["svg_explained"].startswith("<svg")
    assert len(r["atom_weights"]) == 6
    assert r["explanation"] and isinstance(r["explanation"][0], str)
    # Atom shares + molecule-wide effects + base value must equal the prediction.
    total = r["base_value"] + sum(r["atom_weights"]) + sum(r["molecule_wide"].values())
    assert abs(total - r["logS"]) < 0.05


def test_out_of_domain_flag(predictor):
    near = predictor.domain(Chem.MolFromSmiles("CCO"))
    assert near["in_domain"] and near["max_similarity"] == 1.0
