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


@pytest.mark.parametrize("path", ["/", "/compare", "/explore", "/model", "/chemistry"])
def test_pages(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "<html" in r.text
