"""Input validation and name lookup (PubChem is mocked: tests never use the internet)."""

import pytest

from soluai import resolve as R
from soluai.resolve import InputError, resolve

REAL_PUBCHEM = R.pubchem_smiles  # saved before the fixture below replaces it


@pytest.fixture(autouse=True)
def no_internet(monkeypatch):
    monkeypatch.setattr(R, "pubchem_smiles", lambda name, timeout=6.0: None)


def test_offline_name_and_synonym():
    assert resolve("Ethanol").smiles == "CCO"
    assert resolve("acetaminophen").name == "paracetamol"


def test_smiles_input_is_canonicalised():
    r = resolve("OCC")
    assert r.smiles == "CCO" and r.source == "smiles"


@pytest.mark.parametrize("text,fragment", [
    ("", "Please type"),
    ("   ", "Please type"),
    ("water", "solvent"),
    ("[Na+].[Cl-]", "salt"),
    ("NaCl", "inorganic"),
    ("sodium chloride", "inorganic"),
    ("xyzzyplonk", "couldn't find"),
    ("C1CC", "not a valid SMILES"),
    ("x" * 400, "too long"),
    ("C[Hg]C", "element"),
])
def test_friendly_errors(text, fragment):
    with pytest.raises(InputError) as e:
        resolve(text)
    assert fragment in e.value.message


def test_suggestions_for_typos():
    with pytest.raises(InputError) as e:
        resolve("ethanoll")
    assert "ethanol" in e.value.hint


def test_pubchem_used_for_unknown_names(monkeypatch):
    monkeypatch.setattr(R, "pubchem_smiles", lambda name, timeout=6.0: ("Limonene", "CC1=CCC(CC1)C(=C)C"))
    r = resolve("limonene")
    assert r.source == "pubchem" and r.name == "Limonene"


def test_pubchem_parsing(monkeypatch):
    class Resp:
        status_code, ok = 200, True

        @staticmethod
        def json():
            return {"PropertyTable": {"Properties": [{"CID": 1, "SMILES": "CCO", "Title": "Ethanol"}]}}

    monkeypatch.setattr(R.requests, "get", lambda *a, **k: Resp())
    assert REAL_PUBCHEM("ethanol") == ("Ethanol", "CCO")


def test_pubchem_offline_returns_none(monkeypatch):
    def boom(*a, **k):
        raise R.requests.ConnectionError("no internet")

    monkeypatch.setattr(R.requests, "get", boom)
    assert REAL_PUBCHEM("ethanol") is None
