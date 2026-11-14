"""Series, "make it more soluble", quiz, batch and the measured-value lookup."""

from rdkit import Chem

from soluai import lab
from soluai.common_molecules import COMMON, FA_NAMES, SYNONYMS, lookup
from soluai.explain import USP_TERMS, analyze, usp_term


def test_offline_list_is_valid():
    assert len(COMMON) >= 200
    assert all(Chem.MolFromSmiles(smi) is not None for smi, _ in COMMON.values())
    assert all(target in COMMON for target in list(SYNONYMS.values()) + list(FA_NAMES.values()))


def test_persian_names():
    assert lookup("شکر")[0] == "sucrose" and lookup("ویتامین ث")[0] == "ascorbic acid"


def test_measured_lookup(predictor):
    m = predictor.measured(Chem.MolFromSmiles("CCO"))
    assert m["exact"] and abs(m["logS"] - 1.23) < 0.3
    assert predictor.measured(Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO")) is None


def test_analyze_has_new_fields_and_persian(predictor):
    r = analyze(predictor, Chem.MolFromSmiles("CCO"), "fa")
    assert r["kind"] == "organic" and "delaney" in r["classic"] and r["usp"] in {term for _, term in USP_TERMS}
    assert any("؀" <= ch <= "ۿ" for ch in r["explanation"][0])


def test_usp_terms():
    assert usp_term(2000) == "very_soluble" and usp_term(50) == "soluble" and usp_term(0.00001) == "practically_insoluble"


def test_series_solubility_falls_with_chain_length(predictor):
    by_key = {s["key"]: s for s in lab.series(predictor)["series"]}
    for key in ("alkanes", "alcohols", "acids"):
        assert -0.8 < by_key[key]["slope_predicted"] < -0.3
        assert -0.9 < by_key[key]["slope_measured"] < -0.3
    alcohols = {p["n"]: p for p in by_key["alcohols"]["points"]}
    assert alcohols[2]["smiles"] == "CCO" and alcohols[2]["predicted"] > alcohols[8]["predicted"]


def test_suggestions_raise_predicted_solubility(predictor):
    mol = Chem.MolFromSmiles("CCCCCC")
    s = lab.suggest(predictor, mol)
    assert s["ideas"] and all(i["delta"] > 0 and Chem.MolFromSmiles(i["smiles"]) for i in s["ideas"])
    assert s["ideas"] == sorted(s["ideas"], key=lambda i: -i["delta"])
    assert lab.suggest(predictor, Chem.MolFromSmiles(COMMON["glucose"][0]))["ideas"] == []


def test_quiz_rounds_have_a_clear_answer(predictor):
    q = lab.quiz(predictor, n=5, level="easy", seed=3)
    assert len(q["rounds"]) == 5
    for r in q["rounds"]:
        a, b = r["molecules"]
        assert abs(a["measured"] - b["measured"]) >= 1.5 and r["answer"] == int(b["measured"] > a["measured"])
    assert lab.quiz(predictor, n=3, level="hard", seed=3)["rounds"]


def test_batch_mixes_molecules_salts_and_errors(predictor):
    b = lab.batch(predictor, ["ethanol", "c1ccccc1", "AgCl", "not a molecule", "  "])
    kinds = [r["kind"] for r in b["rows"]]
    assert kinds == ["organic", "organic", "ionic", "error"] and b["n_ok"] == 2
    assert b["rows"][0]["low"] < b["rows"][0]["logS"] < b["rows"][0]["high"]


def test_household(predictor):
    h = lab.household(predictor)
    assert h["n_compared"] >= 15 and h["mean_abs_error_model"] < 1.0
    assert {r["kind"] for r in h["rows"]} == {"organic", "ionic"}
