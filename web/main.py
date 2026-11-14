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
import re
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so `import soluai` works without installing the package

from soluai import RESULTS_DIR, __version__, ionic, lab  # noqa: E402
from soluai.common_molecules import COMMON, FA_CATEGORIES, SYNONYMS, fa_name  # noqa: E402
from soluai.explain import analyze  # noqa: E402
from soluai.predict import CATEGORIES, Predictor  # noqa: E402
from soluai.resolve import InputError, resolve  # noqa: E402

STATIC = Path(__file__).parent / "static"
PAGES = {"": "index.html", "compare": "compare.html", "explore": "explore.html",
         "model": "model.html", "chemistry": "chemistry.html", "series": "series.html",
         "salts": "salts.html", "quiz": "quiz.html", "batch": "batch.html"}



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


Lang = Literal["en", "fa"]


class PredictRequest(BaseModel):
    """A molecule name or SMILES. ``lang`` picks the language of the explanation."""

    query: Query = Field(..., examples=["ethanol", "CCO"])
    lang: Lang = "en"


class CompareRequest(BaseModel):
    """Two to four molecules."""

    queries: list[Query] = Field(..., min_length=2, max_length=4)
    lang: Lang = "en"


class BatchRequest(BaseModel):
    """Up to 200 names or SMILES strings."""

    queries: list[Query] = Field(..., min_length=1, max_length=200)
    lang: Lang = "en"


class MixRequest(BaseModel):
    """Two salts, each as [cation key, anion key], e.g. ["Ag", "NO3"] and ["Na", "Cl"]."""

    a: tuple[str, str]
    b: tuple[str, str]
    lang: Lang = "en"


# Persian versions of the friendly error messages written in soluai/resolve.py.
_FA_ERRORS: list[tuple[str, str, str]] = [
    (r"^Please type", "لطفاً نام یک مولکول یا یک رشتهٔ SMILES بنویسید.", "مثال: اتانول، ethanol یا CCO"),
    (r"too long", "این ورودی خیلی طولانی است (حداکثر ۳۰۰ نویسه).", ""),
    (r"^Water is the solvent", "آب خودِ حلال است، پس «انحلال‌پذیری آب در آب» معنا ندارد؛ آب کاملاً با خودش مخلوط می‌شود!",
     "مولکولی را امتحان کنید که در آب حل می‌شود، مثل اتانول یا گلوکز."),
    (r"inorganic substance", "این یک مادهٔ معدنی است و بیرون از محدودهٔ آموزش SoluAI قرار دارد.",
     "SoluAI مولکول‌های آلی (کربن‌دار) را پیش‌بینی می‌کند. برای نمک‌ها صفحهٔ «نمک‌ها» را ببینید."),
    (r"salt or a mixture", "این شبیه یک نمک یا مخلوط است (چند تکهٔ جدا از هم).",
     "SoluAI فقط مولکول‌های آلیِ تکی را پیش‌بینی می‌کند. نام یا فرمول نمک (مثل NaCl) را بنویسید تا قواعد انحلال‌پذیری پاسخ بدهند."),
    (r"contains an element", "این مولکول عنصری دارد (مثلاً یک فلز) که در داده‌های آموزشی SoluAI نبوده است.",
     "عنصرهای مجاز: C, H, N, O, S, P, F, Cl, Br, I, B, Si, Se"),
    (r"not a valid SMILES", "این یک رشتهٔ SMILES معتبر نیست.", "پرانتزها و شماره‌های حلقه باید بسته شوند؛ مثلاً بنزن: c1ccccc1"),
    (r"couldn't find a molecule", "متأسفم، مولکولی با این نام پیدا نکردم.",
     "املا را بررسی کنید یا به‌جای نام، SMILES بنویسید. جست‌وجوی نام به اینترنت نیاز دارد."),
    (r"could not be processed", "این مولکول قابل پردازش نبود.", ""),
]


def _error(err: InputError, query: str, lang: str = "en") -> HTTPException:
    message, hint = err.message, err.hint
    if lang == "fa":
        for pattern, fa_message, fa_hint in _FA_ERRORS:
            if re.search(pattern, err.message):
                message = fa_message
                hint = hint.replace("Did you mean:", "منظورتان این بود:") if hint.startswith("Did you mean") else fa_hint
                break
    return HTTPException(status_code=422, detail={"message": message, "hint": hint, "query": query})


def predict_one(query: str, lang: str = "en") -> dict:
    """Resolve + analyse one query; raises HTTPException with a friendly message.

    Salts (NaCl, silver nitrate ...) are answered by the textbook solubility rules;
    everything else goes to the machine-learning model.
    """
    salt = ionic.analyze_text(query, lang)
    if salt:
        return {"query": query, **salt}
    try:
        r = resolve(query)
    except InputError as err:
        raise _error(err, query, lang) from None
    result = analyze(predictor(), r.mol, lang)
    name = (fa_name(r.name) if lang == "fa" else None) or r.name
    return {"query": query, "name": name, "name_en": r.name, "smiles": r.smiles, "source": r.source, **result}


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health() -> dict:
    """Used by Render to check the service is up."""
    return {"status": "ok", "model_version": predictor().version}


@app.get("/api/molecules")
def molecules(lang: Lang = "en") -> dict:
    """The offline list of common molecules, grouped by category (plus every name for autocomplete)."""
    groups: dict[str, list[dict]] = {}
    for name, (smiles, cat) in COMMON.items():
        title = FA_CATEGORIES.get(cat, cat) if lang == "fa" else cat
        label = (fa_name(name) if lang == "fa" else None) or name
        groups.setdefault(title, []).append({"name": name, "label": label, "smiles": smiles})
    return {"categories": groups, "names": sorted(set(COMMON) | set(SYNONYMS)),
            "solubility_classes": [{"min_logS": c, "label": l} for c, l in CATEGORIES]}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    """Predict the solubility of one molecule and explain it (salts: solubility rules)."""
    return predict_one(req.query, req.lang)


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    """Predict several molecules; failures are reported per molecule."""
    out = []
    for q in req.queries:
        try:
            out.append(predict_one(q, req.lang))
        except HTTPException as exc:
            out.append({"query": q, "error": exc.detail})
    return {"results": out}


@app.post("/api/suggest")
def suggest(req: PredictRequest) -> dict:
    """Small changes to the molecule that the model expects to raise its solubility."""
    try:
        r = resolve(req.query)
    except InputError as err:
        raise _error(err, req.query, req.lang) from None
    return {"query": req.query, "name": r.name, "smiles": r.smiles, **lab.suggest(predictor(), r.mol, req.lang)}


@app.get("/api/series")
def series(lang: Lang = "en") -> dict:
    """Homologous series (alkanes, alcohols, acids ...): solubility against chain length."""
    return _series(lang)


@lru_cache(maxsize=2)
def _series(lang: str) -> dict:
    return lab.series(predictor(), lang)


@app.get("/api/quiz")
def quiz(n: int = 8, level: Literal["easy", "hard"] = "easy", lang: Lang = "en") -> dict:
    """Rounds of "which one dissolves better?" built from measured values."""
    return lab.quiz(predictor(), max(1, min(n, 15)), level, lang)


@app.post("/api/batch")
def batch(req: BatchRequest) -> dict:
    """Predict up to 200 molecules at once (offline names and SMILES only)."""
    return lab.batch(predictor(), req.queries, req.lang)


@app.get("/api/household")
def household(lang: Lang = "en") -> dict:
    """Everyday substances: measured value vs SoluAI vs the classic formula."""
    return lab.household(predictor(), lang)


@app.get("/api/salts")
def salts(lang: Lang = "en") -> dict:
    """The solubility-rules table for ionic compounds."""
    return ionic.table(lang)


@app.get("/api/salts/{cation}/{anion}")
def salt(cation: str, anion: str, lang: Lang = "en") -> dict:
    """One salt, by ion keys (e.g. /api/salts/Ag/Cl)."""
    if cation not in ionic.CATION or anion not in ionic.ANION:
        raise HTTPException(404, "Unknown ion")
    return ionic.describe(ionic.CATION[cation], ionic.ANION[anion], lang)


@app.post("/api/salts/mix")
def salts_mix(req: MixRequest) -> dict:
    """Mix two salt solutions: does a precipitate form?"""
    try:
        ions = (ionic.CATION[req.a[0]], ionic.ANION[req.a[1]], ionic.CATION[req.b[0]], ionic.ANION[req.b[1]])
    except KeyError:
        raise HTTPException(404, "Unknown ion") from None
    return ionic.mix(*ions, lang=req.lang)


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
        msg = ("Please send between 1 and 200 molecules." if request.url.path.endswith("/batch")
               else "Please compare between 2 and 4 molecules.")
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
    """Serve one of the HTML pages."""
    if page not in PAGES:
        raise HTTPException(404, "Page not found")
    return FileResponse(STATIC / PAGES[page])
