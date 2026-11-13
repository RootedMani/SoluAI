"""SoluAI website: a small JSON API plus static HTML pages.

Run locally:   uvicorn web.main:app --reload
Then open:     http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so `import soluai` works without installing the package

from soluai import RESULTS_DIR, __version__  # noqa: E402
from soluai.common_molecules import COMMON  # noqa: E402
from soluai.explain import analyze  # noqa: E402
from soluai.predict import CATEGORIES, Predictor  # noqa: E402
from soluai.resolve import InputError, resolve  # noqa: E402

STATIC = Path(__file__).parent / "static"
PAGES = {"": "index.html", "compare": "compare.html", "explore": "explore.html",
         "model": "model.html", "chemistry": "chemistry.html"}



@lru_cache(maxsize=1)
def predictor() -> Predictor:
    """Load the model once and keep it in memory."""
    p = Predictor()
    _ = p.training_fps  # pre-compute fingerprints so the first request is fast
    return p


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model when the server starts, not on the first visitor's request."""
    predictor()
    yield


app = FastAPI(title="SoluAI", version=__version__, lifespan=lifespan,
              description="Water solubility prediction with explanations.")
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ----------------------------------------------------------------------------- request bodies
Query = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]


class PredictRequest(BaseModel):
    """A molecule name or SMILES."""

    query: Query = Field(..., examples=["ethanol", "CCO"])


class CompareRequest(BaseModel):
    """Two to four molecules."""

    queries: list[Query] = Field(..., min_length=2, max_length=4)


def _error(err: InputError, query: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"message": err.message, "hint": err.hint, "query": query})


def predict_one(query: str) -> dict:
    """Resolve + analyse one query; raises HTTPException with a friendly message."""
    try:
        r = resolve(query)
    except InputError as err:
        raise _error(err, query) from None
    result = analyze(predictor(), r.mol)
    return {"query": query, "name": r.name, "smiles": r.smiles, "source": r.source, **result}


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health() -> dict:
    """Used by Render to check the service is up."""
    return {"status": "ok", "model_version": predictor().version}


@app.get("/api/molecules")
def molecules() -> dict:
    """The offline list of common molecules, grouped by category."""
    groups: dict[str, list[dict]] = {}
    for name, (smiles, cat) in COMMON.items():
        groups.setdefault(cat, []).append({"name": name, "smiles": smiles})
    return {"categories": groups, "solubility_classes": [{"min_logS": c, "label": l} for c, l in CATEGORIES]}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    """Predict the solubility of one molecule and explain it."""
    return predict_one(req.query)


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    """Predict several molecules; failures are reported per molecule."""
    out = []
    for q in req.queries:
        try:
            out.append(predict_one(q))
        except HTTPException as exc:
            out.append({"query": q, "error": exc.detail})
    return {"results": out}


@lru_cache(maxsize=4)
def _results_file(name: str) -> bytes:
    return (RESULTS_DIR / name).read_bytes()


@app.get("/api/dataset")
def dataset() -> JSONResponse:
    """Points for the Explore page (all cleaned AqSolDB molecules)."""
    return JSONResponse(json.loads(_results_file("dataset_points.json")))


@app.get("/api/report")
def report() -> JSONResponse:
    """Everything the 'How the model works' page shows."""
    metrics = json.loads(_results_file("metrics.json"))
    rep = json.loads(_results_file("report.json"))
    return JSONResponse({"metrics": metrics, **rep})


@app.exception_handler(RequestValidationError)
async def _validation(request, exc: RequestValidationError) -> JSONResponse:  # noqa: ANN001
    """Friendly messages for invalid request bodies (e.g. too long, too many molecules)."""
    err = exc.errors()[0] if exc.errors() else {}
    kind = err.get("type", "")
    if kind == "string_too_long":
        msg = "That input is too long (max 300 characters)."
    elif kind in ("too_short", "too_long"):
        msg = "Please compare between 2 and 4 molecules."
    else:
        msg = "That input is not valid."
    return JSONResponse({"detail": {"message": msg, "hint": "", "query": None}}, status_code=422)


# ----------------------------------------------------------------------------- pages
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC / "favicon.svg")


@app.get("/{page}", include_in_schema=False)
@app.get("/", include_in_schema=False)
def page(page: str = "") -> FileResponse:
    """Serve one of the five HTML pages."""
    if page not in PAGES:
        raise HTTPException(404, "Page not found")
    return FileResponse(STATIC / PAGES[page])
