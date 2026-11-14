"""The website's JSON API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(predictor):  # noqa: ARG001 - skips if there is no model
    from web.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_predict_by_name(client):
    r = client.post("/api/predict", json={"query": "caffeine"}).json()
    assert r["name"] == "caffeine" and "logS" in r and r["category"]


def test_friendly_error(client):
    r = client.post("/api/predict", json={"query": "C1CC"})
    assert r.status_code == 422
    assert "SMILES" in r.json()["detail"]["message"]


def test_too_long_input(client):
    r = client.post("/api/predict", json={"query": "C" * 301})
    assert r.status_code == 422 and "too long" in r.json()["detail"]["message"]


def test_compare_mixed(client):
    r = client.post("/api/compare", json={"queries": ["ethanol", "not-a-molecule-xyz"]}).json()["results"]
    assert "logS" in r[0] and "error" in r[1]


def test_compare_needs_two(client):
    assert client.post("/api/compare", json={"queries": ["ethanol"]}).status_code == 422


def test_salt_is_answered_by_rules(client):
    r = client.post("/api/predict", json={"query": "AgCl"}).json()
    assert r["kind"] == "ionic" and r["status"] == "insoluble"


def test_persian_answer_and_error(client):
    r = client.post("/api/predict", json={"query": "شکر", "lang": "fa"}).json()
    assert r["name_en"] == "sucrose" and r["name"] == "ساکارز" and r["measured"]
    e = client.post("/api/predict", json={"query": "C1CC", "lang": "fa"})
    assert e.status_code == 422 and "SMILES" in e.json()["detail"]["message"]


def test_new_endpoints(client):
    assert len(client.get("/api/series").json()["series"]) == 7
    assert client.post("/api/suggest", json={"query": "toluene"}).json()["ideas"]
    assert len(client.get("/api/quiz?n=3").json()["rounds"]) == 3
    assert client.post("/api/batch", json={"queries": ["ethanol", "NaCl"]}).json()["n_ok"] == 1
    assert client.post("/api/batch", json={"queries": ["C"] * 201}).status_code == 422
    assert client.get("/api/household?lang=fa").json()["rows"]
    t = client.get("/api/salts").json()
    assert len(t["cells"]) == len(t["cations"]) * len(t["anions"])
    assert client.get("/api/salts/Ba/SO4").json()["status"] == "insoluble"
    assert client.get("/api/salts/Xx/SO4").status_code == 404
    m = client.post("/api/salts/mix", json={"a": ["Ag", "NO3"], "b": ["Na", "Cl"]}).json()
    assert m["precipitates"][0]["formula"] == "AgCl"
    assert len(client.get("/api/molecules").json()["names"]) > 250


def test_offline_assets_are_bundled(client):
    for path in ["/static/vendor/jsme/jsme.nocache.js", "/static/vendor/fonts/vazirmatn-arabic-wght-normal.woff2",
                 "/static/js/boot.js"]:
        assert client.get(path).status_code == 200
    assert "fonts.googleapis.com" not in client.get("/").text


@pytest.mark.parametrize("path", ["/", "/compare", "/explore", "/model", "/chemistry", "/series", "/salts", "/quiz", "/batch"])
def test_pages(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "<html" in r.text
