"""Model v2's "referee": trust signals, adaptive ranges and confidence labels."""

import numpy as np
import pytest
from rdkit import Chem

from soluai.explain import analyze, extended_atom_sets
from soluai.features import EXTRA_DESCRIPTOR_NAMES, Featurizer, extended_descriptors
from soluai.uncertainty import TRUST_SIGNALS, TrustModel, tanimoto_matrix, trust_report, trust_signals


def test_tanimoto_matrix_matches_rdkit():
    from rdkit import DataStructs

    f = Featurizer("fp")
    smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ours = tanimoto_matrix(f.transform(smiles), f.transform(smiles))
    fps = [f.fingerprint(m) for m in mols]
    theirs = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    assert np.allclose(ours, theirs, atol=1e-6)


def test_trust_signals_by_hand():
    sims = np.array([[0.9, 0.5, 0.1], [-1.0, 0.2, 0.4]])        # query 2 may not use reference 0
    y_ref, err_ref = np.array([-1.0, -3.0, -5.0]), np.array([0.1, 0.5, 2.0])
    S = trust_signals(sims, y_ref, err_ref, pred=np.array([-1.5, -4.0]), logp=[1.0, 2.0], mol_wt=[100.0, 200.0],
                      heavy=[7, 14], k=2)
    assert S.shape == (2, len(TRUST_SIGNALS))
    s = dict(zip(TRUST_SIGNALS, S.T))
    assert s["max_sim"] == pytest.approx([0.9, 0.4]) and s["mean_sim"] == pytest.approx([0.7, 0.3])
    # Query 1: neighbours 0 and 1 (weights 0.9, 0.5); query 2: neighbours 2 and 1 (weights 0.4, 0.2).
    assert s["nb_error"] == pytest.approx([(0.9 * 0.1 + 0.5 * 0.5) / 1.4, (0.4 * 2.0 + 0.2 * 0.5) / 0.6], abs=1e-4)
    assert s["nb_gap"] == pytest.approx([abs(-1.5 - (0.9 * -1 + 0.5 * -3) / 1.4), abs(-4 - (0.4 * -5 + 0.2 * -3) / 0.6)], abs=1e-4)
    assert s["nb_spread"][0] > 0 and list(s["prediction"]) == [-1.5, -4.0]


def test_trust_model_learns_who_is_hard_and_keeps_coverage():
    """Synthetic check: errors are 4x bigger when signal 0 is low; the referee must find that out."""
    rng = np.random.default_rng(0)
    n = 4000
    S = rng.uniform(0, 1, size=(n, len(TRUST_SIGNALS)))
    scale = np.where(S[:, 0] > 0.5, 0.2, 0.8)
    err = np.abs(rng.normal(0, scale))
    folds = [(np.setdiff1d(np.arange(n), np.arange(i, n, 5)), np.arange(i, n, 5)) for i in range(5)]
    trust = TrustModel(coverage=0.9)
    sigma = trust.fit(S, err, folds)
    easy, hard = S[:, 0] > 0.5, S[:, 0] <= 0.5
    assert sigma[hard].mean() > 2.5 * sigma[easy].mean()
    # New data from the same process: about 90 % inside the range, overall and in each group.
    S2 = rng.uniform(0, 1, size=(n, len(TRUST_SIGNALS)))
    err2 = np.abs(rng.normal(0, np.where(S2[:, 0] > 0.5, 0.2, 0.8)))
    hw = trust.half_width(S2)
    assert 0.87 < (err2 <= hw).mean() < 0.93
    for mask in (S2[:, 0] > 0.5, S2[:, 0] <= 0.5):
        assert 0.85 < (err2[mask] <= hw[mask]).mean() < 0.95
    assert np.all(trust.sigma(S2) >= trust.floor)
    assert trust.tier(0.5) == "high" and trust.tier(1.0) == "medium" and trust.tier(2.0) == "low"
    rep = trust_report(np.zeros(n), err2, hw, trust)   # y = 0 and pred = err2 gives the same absolute errors
    assert rep["tiers"]["high"]["within_1"] > rep["tiers"]["low"]["within_1"]
    assert [c["fraction"] for c in rep["confidence_curve"]][-1] == 1.0


def test_predictor_gives_confidence(predictor):
    easy = predictor.domain(Chem.MolFromSmiles("CCO"))                       # ethanol: in the training data
    odd = predictor.domain(Chem.MolFromSmiles("C12C3C4C1C5C2C3C45"))         # cubane: nothing like it
    for d in (easy, odd):
        assert d["confidence"] in {"high", "medium", "low"}
        assert d["half_width"] > 0 and d["expected_error"] >= predictor.trust.floor
    assert easy["half_width"] < odd["half_width"]
    assert easy["confidence"] == "high"


def test_ranges_differ_between_molecules(predictor):
    widths = set()
    for smi in ["CCO", "CCCCCCCCCCCCCCCC", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C12C3C4C1C5C2C3C45"]:
        r = analyze(predictor, Chem.MolFromSmiles(smi))
        widths.add(round(r["interval"][1] - r["interval"][0], 2))
        assert r["domain"]["confidence"] in {"high", "medium", "low"}
    assert len(widths) > 1   # v1 gave every familiar molecule the same width


@pytest.mark.parametrize("smiles", ["CC(=O)Oc1ccccc1C(=O)O", "OCC(O)CO", "CCCCCCCC", "NC(=O)c1ccc(cc1)S(N)(=O)=O", "ClC(Cl)(Cl)Cl"])
def test_extended_descriptor_atoms_match_values(smiles):
    """A surface-area or functional-group descriptor is non-zero exactly when some atoms carry it."""
    mol = Chem.MolFromSmiles(smiles)
    sets = extended_atom_sets(mol, EXTRA_DESCRIPTOR_NAMES)
    values = dict(zip(EXTRA_DESCRIPTOR_NAMES, extended_descriptors(mol)))
    assert len(sets) > 100
    for name, atoms in sets.items():
        if name.startswith("VSA_EState"):      # a sum of E-state values, which can cancel to zero
            continue
        assert (len(atoms) > 0) == (values[name] != 0), name


def test_explanation_adds_up_for_extended_model(predictor):
    for smi in ["CCO", "c1ccc2ccccc2c1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]:
        r = analyze(predictor, Chem.MolFromSmiles(smi))
        total = r["base_value"] + sum(r["atom_weights"]) + sum(r["molecule_wide"].values())
        assert abs(total - r["logS"]) < 0.05
        keys = [c["key"] for c in r["contributions"]]
        assert len(keys) == len(set(keys))


def test_far_from_training_data_is_never_confident(predictor):
    d = predictor.domain(Chem.MolFromSmiles("C12C3C4C1C5C2C3C45"))   # cubane
    assert not d["in_domain"] and d["confidence"] == "low"
