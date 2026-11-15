"""Featurization: are the numbers the model sees chemically correct?"""

import numpy as np
import pytest
from rdkit import Chem

from soluai.data import standardize
from soluai.features import DESCRIPTOR_NAMES, Featurizer, aromatic_proportion, descriptors


def desc(smiles: str) -> dict[str, float]:
    return dict(zip(DESCRIPTOR_NAMES, descriptors(Chem.MolFromSmiles(smiles))))


def test_benzene_descriptors():
    d = desc("c1ccccc1")
    assert d["NumHDonors"] == 0
    assert d["NumAromaticRings"] == 1
    assert d["AromaticProportion"] == 1.0
    assert d["MolWt"] == pytest.approx(78.11, abs=0.01)


def test_ethanol_is_polar_and_hexane_is_not():
    eth, hex_ = desc("CCO"), desc("CCCCCC")
    assert eth["NumHDonors"] == 1 and hex_["NumHDonors"] == 0
    assert eth["TPSA"] > 0 and hex_["TPSA"] == 0
    assert eth["MolLogP"] < hex_["MolLogP"]


def test_aromatic_proportion_of_toluene():
    assert aromatic_proportion(Chem.MolFromSmiles("Cc1ccccc1")) == pytest.approx(6 / 7)


@pytest.mark.parametrize("kind,length", [("desc", 14), ("fp", 1024), ("desc+fp", 1038)])
def test_feature_vector_length(kind, length):
    f = Featurizer(kind)
    assert len(f.names) == length
    assert f.transform(["CCO", "c1ccccc1"]).shape == (2, length)


def test_fingerprint_is_binary_and_deterministic():
    f = Featurizer("fp")
    a, b = f.transform(["CC(=O)Oc1ccccc1C(=O)O"]), f.transform(["CC(=O)Oc1ccccc1C(=O)O"])
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert np.array_equal(a, b)


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        Featurizer().transform(["C1CC"])


@pytest.mark.parametrize("smiles,reason", [
    ("[Na+].[Cl-]", "salt_or_mixture"),
    ("CC(=O)[O-].[Na+]", "salt_or_mixture"),
    ("O", "no_carbon"),
    ("C[Hg]C", "non_organic_element"),
    ("not a smiles", "invalid_smiles"),
    ("CCO", "ok"),
])
def test_cleaning_rules(smiles, reason):
    assert standardize(smiles)[1] == reason


def test_extended_feature_vector():
    """Model v2: 14 descriptors + every other RDKit descriptor + fingerprint."""
    from soluai.features import EXTRA_DESCRIPTOR_NAMES, FAMILIES, family

    f = Featurizer("rdkit+fp")
    n = 14 + len(EXTRA_DESCRIPTOR_NAMES) + 1024
    assert len(f.names) == n and len(set(f.names)) == n
    assert f.names[:14] == DESCRIPTOR_NAMES and f.names[-1] == "FP_1023"
    X = f.transform(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    assert X.shape == (3, n)
    # The first 14 columns and the fingerprint are exactly the v1 features.
    assert np.array_equal(X[:, :14], Featurizer("desc").transform(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]))
    assert np.array_equal(X[:, -1024:], Featurizer("fp").transform(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]))
    assert np.all(np.abs(X[np.isfinite(X)]) <= 1e6)
    assert {family(name) for name in EXTRA_DESCRIPTOR_NAMES} <= set(FAMILIES)
    assert family("SlogP_VSA2") == "surface" and family("fr_ester") == "groups" and family("Chi1v") == "shape"
