"""The solubility-rules engine for salts (no model needed)."""

import pytest

from soluai import ionic


@pytest.mark.parametrize("text,formula,status", [
    ("NaCl", "NaCl", "soluble"),
    ("nacl", "NaCl", "soluble"),
    ("silver chloride", "AgCl", "insoluble"),
    ("Ca(OH)2", "Ca(OH)₂", "slightly"),
    ("(NH4)2SO4", "(NH₄)₂SO₄", "soluble"),
    ("lead(II) iodide", "PbI₂", "insoluble"),
    ("iron (iii) hydroxide", "Fe(OH)₃", "insoluble"),
    ("baking soda", "NaHCO₃", "soluble"),
    ("BaSO4", "BaSO₄", "insoluble"),
    ("CH3COONa", "NaCH₃COO", "soluble"),
    ("نقره نیترات", "AgNO₃", "soluble"),
    ("نمک خوراکی", "NaCl", "soluble"),
])
def test_rules(text, formula, status):
    r = ionic.analyze_text(text)
    assert r["formula"] == formula and r["status"] == status


@pytest.mark.parametrize("text", ["ethanol", "CCO", "CO", "caffeine", "c1ccccc1", "HCl", ""])
def test_organic_input_is_not_a_salt(text):
    assert ionic.analyze_text(text) is None


def test_every_measured_value_agrees_with_its_rule():
    """The rules must not contradict the lab values we ship (borderline salts may sit on either side)."""
    for (c, a) in ionic.MEASURED:
        r = ionic.describe(ionic.CATION[c], ionic.ANION[a])
        assert r["agrees_with_measurement"], r["formula"]


def test_dissociation_equation_and_charge_balance():
    r = ionic.analyze_text("Al2(SO4)3")
    assert r["equation"] == "Al₂(SO₄)₃(s) → 2 Al³⁺(aq) + 3 SO₄²⁻(aq)"
    for c in ionic.CATIONS:
        for a in ionic.ANIONS:
            nc, na = ionic._counts(c, a)
            assert nc * c.charge + na * a.charge == 0


def test_mixing_gives_a_balanced_precipitation_equation():
    m = ionic.mix(ionic.CATION["Al"], ionic.ANION["SO4"], ionic.CATION["Ba"], ionic.ANION["Cl"])
    assert m["equation"] == "Al₂(SO₄)₃(aq) + 3 BaCl₂(aq) → 2 AlCl₃(aq) + 3 BaSO₄(s)"
    assert m["net_ionic"] == ["Ba²⁺(aq) + SO₄²⁻(aq) → BaSO₄(s)"]


def test_mixing_two_soluble_products_gives_no_precipitate():
    m = ionic.mix(ionic.CATION["Na"], ionic.ANION["NO3"], ionic.CATION["K"], ionic.ANION["Cl"])
    assert m["precipitates"] == [] and "No precipitate" in m["message"]


def test_persian_text():
    r = ionic.analyze_text("AgCl", "fa")
    assert r["status_label"] == "نامحلول" and r["name"] == "نقره کلرید"
