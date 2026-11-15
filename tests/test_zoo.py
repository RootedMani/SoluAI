"""The model zoo (choose a model, see every model's answer) and the four quiz levels."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem, DataStructs

from soluai import lab
from soluai.zoo import MEMBERS, MODEL_IDS, ZOO_DIR, Zoo


@pytest.fixture(scope="module")
def zoo(predictor):
    if not (ZOO_DIR / "zoo.json").exists():
        pytest.skip("no model zoo - run `python -m soluai.train_zoo` first")
    z = Zoo(predictor)
    z.v2.load_all()
    return z


@pytest.fixture(scope="module")
def client(zoo):  # noqa: ARG001 - skips if there is no zoo
    from web.main import app

    with TestClient(app) as c:
        yield c


MOLS = ["CCO", "CCCCCC", "OCC(O)CO", "c1ccc2ccccc2c1", "CC(=O)Oc1ccccc1C(=O)O"]


def test_every_model_predicts_every_molecule(zoo):
    p = zoo.predict_all([Chem.MolFromSmiles(s) for s in MOLS])
    assert set(p) == set(MODEL_IDS)
    for v in p.values():
        assert v.shape == (len(MOLS),) and np.isfinite(v).all() and (-15 < v).all() and (v < 3).all()


def test_consensus_is_the_weighted_average(zoo):
    mols = [Chem.MolFromSmiles(s) for s in MOLS]
    p = zoo.predict_all(mols)
    w = zoo.v2.weights
    assert np.allclose(p["consensus"], sum(w[n] * p[f"zoo-{n}"] for n in MEMBERS))
    assert abs(sum(w.values()) - 1) < 0.05


def test_models_know_alcohol_beats_alkane(zoo):
    p = zoo.predict_all([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCCCC")])
    for mid, v in p.items():
        assert v[0] > v[1] + 1, mid


def test_one_model_equals_its_column_in_predict_all(zoo):
    mols = [Chem.MolFromSmiles(s) for s in MOLS]
    every = zoo.predict_all(mols)
    for mid in MODEL_IDS:
        assert np.allclose(zoo.predict(mid, mols), every[mid])


def test_view_behaves_like_a_predictor(zoo):
    v = zoo.view("zoo-svr")
    mol = Chem.MolFromSmiles("CCO")
    assert isinstance(v.predict_mol(mol), float)
    assert v.domain(mol)["half_width"] in (v.conformal["q"]["in"], v.conformal["q"]["out"])
    assert v.measured(mol)  # passes through to v1.0's training data
    assert zoo.view("zoo-svr") is v


def test_model_info_has_honest_scores(zoo):
    info = {m["id"]: m for m in zoo.info()}
    assert info["consensus"]["default"] and info["chemist"]["explains_atoms"] and not info["v1"]["explains_atoms"]
    for m in ("consensus", "chemist", "v1"):
        assert info[m]["test"]["random"]["r2"] > info["delaney"]["test"]["random"]["r2"] + 0.1


# ---------------------------------------------------------------- quiz levels
@pytest.mark.parametrize("level", list(lab.LEVELS))
def test_quiz_levels_respect_their_gap(zoo, level):
    q = lab.quiz(zoo.view("consensus"), n=6, level=level, seed=1, zoo=zoo)
    assert q["level"] == level and len(q["rounds"]) >= 4
    for r in q["rounds"]:
        a, b = r["molecules"]
        assert abs(a["measured"] - b["measured"]) >= lab.LEVELS[level]["gap"] - 1e-9
        assert r["answer"] == int(b["measured"] > a["measured"])
        assert set(r["model_answers"]) == set(MODEL_IDS)
        assert r["model_answers"]["consensus"] == r["model_answer"]


def test_expert_pairs_look_alike(predictor, zoo):
    q = lab.quiz(zoo.view("v1"), n=6, level="expert", seed=2, zoo=zoo)
    for r in q["rounds"]:
        fa, fb = (predictor.featurizer.fingerprint(Chem.MolFromSmiles(m["smiles"])) for m in r["molecules"])
        assert DataStructs.TanimotoSimilarity(fa, fb) >= 0.45


def test_easy_is_easier_than_hard(zoo):
    gap = lambda lv: np.mean([r["gap"] for s in range(3) for r in lab.quiz(zoo.view("consensus"), 8, lv, seed=s, zoo=zoo)["rounds"]])  # noqa: E731
    assert gap("easy") > gap("medium") and gap("hard") > gap("expert")


# ---------------------------------------------------------------- API
def test_models_endpoint(client):
    r = client.get("/api/models").json()
    assert r["default"] == "consensus" and [m["id"] for m in r["models"]] == MODEL_IDS


def test_predict_with_a_chosen_model(client):
    a = client.post("/api/predict", json={"query": "caffeine", "model": "v1"}).json()
    b = client.post("/api/predict", json={"query": "caffeine", "model": "delaney"}).json()
    assert a["model"]["id"] == "v1" and b["model"]["id"] == "delaney"
    rows = {m["id"]: m["logS"] for m in b["models"]}
    assert set(rows) == set(MODEL_IDS)
    assert b["logS"] == rows["delaney"] and a["logS"] == rows["v1"]
    assert b["explained_by"]["id"] == "chemist" and b["explained_by"]["logS"] == rows["chemist"]
    assert b["interval"][0] < b["logS"] < b["interval"][1]
    assert b["agreement"]["spread"] >= 0


def test_unknown_model_is_rejected(client):
    assert client.post("/api/predict", json={"query": "ethanol", "model": "gpt"}).status_code == 422


def test_other_endpoints_take_a_model(client):
    assert client.get("/api/quiz?n=3&level=medium&model=zoo-mlp").json()["rounds"]
    assert client.get("/api/quiz?n=3&level=expert&model=zoo-cat").json()["level"] == "expert"
    b = client.post("/api/batch", json={"queries": ["ethanol", "hexane"], "model": "zoo-svr"}).json()
    assert b["n_ok"] == 2
    assert client.get("/api/series?model=zoo-huber").json()["series"]
    assert client.post("/api/suggest", json={"query": "naphthalene", "model": "zoo-lgbm"}).status_code == 200


def test_v1_is_the_real_v1_and_chemist_is_latest(zoo, predictor):
    assert zoo.v1.version.startswith("1.") and zoo.base is predictor
    mol = Chem.MolFromSmiles("CCO")
    p = zoo.predict_all([mol])
    assert abs(p["chemist"][0] - predictor.predict_mol(mol)) < 1e-9
    assert abs(p["v1"][0] - zoo.v1.predict_mol(mol)) < 1e-9


def test_only_the_main_model_gets_a_confidence_label(client):
    base = client.post("/api/predict", json={"query": "caffeine", "model": "chemist"}).json()
    other = client.post("/api/predict", json={"query": "caffeine", "model": "zoo-svr"}).json()
    assert base["domain"].get("confidence") in ("high", "medium", "low")
    assert "confidence" not in other["domain"]
    lo, hi = other["interval"]
    assert abs((hi - lo) / 2 - other["domain"]["half_width"]) < 0.02
