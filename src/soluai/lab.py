"""Extra tools built on top of the trained model.

* :func:`series`    - homologous series: solubility as a carbon chain grows;
* :func:`suggest`   - small edits that should make a molecule more soluble;
* :func:`quiz`      - "which one dissolves better?" rounds from measured data;
* :func:`batch`     - many molecules at once;
* :func:`household` - everyday substances, measured vs predicted.

None of these retrain anything: they only call the saved model.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from . import ionic
from .baselines import delaney_one
from .common_molecules import COMMON, fa_name
from .data import DEFAULT_ELEMENTS, standardize
from .explain import draw_svg, usp_term
from .predict import category
from .resolve import InputError, resolve

if TYPE_CHECKING:
    from .predict import Predictor

RDLogger.DisableLog("rdApp.*")
T = tuple[str, str]  # (English, Persian)


def _t(lang: str, pair: T) -> str:
    return pair[1] if lang == "fa" else pair[0]


# ----------------------------------------------------------------------------- homologous series
STEMS = ["meth", "eth", "prop", "but", "pent", "hex", "hept", "oct", "non", "dec", "undec", "dodec"]

#: key -> (title, head group, first n, last n, SMILES builder, name builder)
SERIES: dict[str, dict[str, Any]] = {
    "alkanes": {"title": ("Alkanes", "آلکان‌ها"), "group": "–H", "start": 1, "end": 12,
                "smiles": lambda n: "C" * n, "name": lambda n: f"{STEMS[n - 1]}ane"},
    "alcohols": {"title": ("Alcohols (1-alkanols)", "الکل‌ها"), "group": "–OH", "start": 1, "end": 12,
                 "smiles": lambda n: "C" * n + "O",
                 "name": lambda n: f"{STEMS[n - 1]}anol" if n < 3 else f"1-{STEMS[n - 1]}anol"},
    "acids": {"title": ("Carboxylic acids", "کربوکسیلیک اسیدها"), "group": "–COOH", "start": 1, "end": 12,
              "smiles": lambda n: "C" * (n - 1) + "C(=O)O", "name": lambda n: f"{STEMS[n - 1]}anoic acid"},
    "amines": {"title": ("Amines (1-alkylamines)", "آمین‌ها"), "group": "–NH₂", "start": 1, "end": 12,
               "smiles": lambda n: "C" * n + "N", "name": lambda n: f"{STEMS[n - 1]}ylamine"},
    "aldehydes": {"title": ("Aldehydes", "آلدهیدها"), "group": "–CHO", "start": 1, "end": 12,
                  "smiles": lambda n: "C" * (n - 1) + "C=O", "name": lambda n: f"{STEMS[n - 1]}anal"},
    "ketones": {"title": ("Ketones (2-alkanones)", "کتون‌ها"), "group": "C=O", "start": 3, "end": 12,
                "smiles": lambda n: "CC(=O)" + "C" * (n - 2),
                "name": lambda n: "propanone" if n == 3 else f"2-{STEMS[n - 1]}anone"},
    "chlorides": {"title": ("Chloroalkanes", "کلروآلکان‌ها"), "group": "–Cl", "start": 1, "end": 12,
                  "smiles": lambda n: "C" * n + "Cl",
                  "name": lambda n: f"chloro{STEMS[n - 1]}ane" if n < 3 else f"1-chloro{STEMS[n - 1]}ane"},
}


def _slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope (change in logS per extra carbon)."""
    return round(float(np.polyfit(xs, ys, 1)[0]), 2) if len(xs) >= 3 else None


def series(predictor: "Predictor", lang: str = "en") -> dict:
    """Predicted, measured and classic-formula logS for every member of every series."""
    out = []
    for key, s in SERIES.items():
        ns = list(range(s["start"], s["end"] + 1))
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s["smiles"](n))) for n in ns]
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        pred = predictor.predict_many(smiles)
        points = []
        for n, smi, mol, p in zip(ns, smiles, mols, pred):
            m = predictor.measured(mol)
            points.append({"n": n, "name": s["name"](n), "smiles": smi, "predicted": round(float(p), 2),
                           "measured": m["logS"] if m else None, "classic": round(delaney_one(mol), 2)})
        # The trend only starts once the chain is long enough to matter (n >= 4):
        # the first members mix with water completely, so their numbers hit a ceiling.
        tail = [pt for pt in points if pt["n"] >= 4]
        meas = [pt for pt in tail if pt["measured"] is not None]
        out.append({"key": key, "title": _t(lang, s["title"]), "group": s["group"], "points": points,
                    "slope_predicted": _slope([pt["n"] for pt in tail], [pt["predicted"] for pt in tail]),
                    "slope_measured": _slope([pt["n"] for pt in meas], [pt["measured"] for pt in meas])})
    return {"series": out}


# ----------------------------------------------------------------------------- "make it more soluble"
#: (key, reaction SMARTS, label, chemistry reason)
EDITS: list[tuple[str, str, T, T]] = [
    ("add_oh", "[cH:1]>>[c:1][OH]", ("Add an –OH to the ring", "افزودن ‎–OH‎ به حلقه"),
     ("a new group that gives and accepts hydrogen bonds with water",
      "یک گروه تازه که با آب هم پیوند هیدروژنی می‌دهد و هم می‌پذیرد")),
    ("add_nh2", "[cH:1]>>[c:1][NH2]", ("Add an –NH₂ to the ring", "افزودن ‎–NH₂‎ به حلقه"),
     ("a new hydrogen-bond donor", "یک دهندهٔ تازهٔ پیوند هیدروژنی")),
    ("ch3_oh", "[CH3:1]>>[CH2:1][OH]", ("Turn a –CH₃ into –CH₂OH", "تبدیل ‎–CH₃‎ به ‎–CH₂OH‎"),
     ("an oily end becomes a hydrogen-bonding end", "یک سرِ چرب به یک سرِ پیونددهندهٔ هیدروژنی تبدیل می‌شود")),
    ("ch3_cooh", "[CH3:1]>>[C:1](=O)[OH]", ("Turn a –CH₃ into –COOH", "تبدیل ‎–CH₃‎ به ‎–COOH‎"),
     ("a carboxylic acid is very polar (and ionises in water)", "کربوکسیلیک اسید بسیار قطبی است (و در آب یونیده می‌شود)")),
    ("hal_oh", "[#6;!$([#6]([Cl,Br,I,F])[Cl,Br,I,F]):1][Cl,Br,I]>>[#6:1][OH]", ("Swap a halogen for –OH", "جایگزینی هالوژن با ‎–OH‎"),
     ("a big oily atom is replaced by a hydrogen-bonding group", "یک اتم بزرگ و چرب جای خود را به گروهی با پیوند هیدروژنی می‌دهد")),
    ("hal_h", "[#6:1][Cl,Br,I]>>[#6:1]", ("Remove a halogen", "حذف یک هالوژن"),
     ("the molecule becomes smaller and less oily", "مولکول کوچک‌تر و کمتر چرب می‌شود")),
    ("ch_n", "[cH:1]>>[n:1]", ("Swap a ring CH for N (as in pyridine)", "جایگزینی ‎CH‎ حلقه با ‎N‎ (مثل پیریدین)"),
     ("the ring nitrogen accepts hydrogen bonds from water", "نیتروژن حلقه از آب پیوند هیدروژنی می‌پذیرد")),
    ("shorten", "[CH3][CH2:1]>>[CH3:1]", ("Shorten a carbon chain by one CH₂", "کوتاه کردن زنجیر کربنی به اندازهٔ یک ‎CH₂‎"),
     ("less non-polar surface for water to make room for", "سطح ناقطبی کمتری که آب باید برایش جا باز کند")),
    ("ester_acid", "[C:1](=[O:2])[O:3][CX4]>>[C:1](=[O:2])[OH:3]", ("Turn an ester into a free acid", "تبدیل استر به اسید آزاد"),
     ("the free –COOH can donate a hydrogen bond", "‎–COOH‎ آزاد می‌تواند پیوند هیدروژنی بدهد")),
    ("ome_oh", "[c:1][O:2][CH3]>>[c:1][OH:2]", ("Turn –OCH₃ into –OH", "تبدیل ‎–OCH₃‎ به ‎–OH‎"),
     ("the oxygen gains an H it can donate to water", "اکسیژن یک ‎H‎ به دست می‌آورد که می‌تواند به آب بدهد")),
    ("ring_o", "[#6:2][CH2;R:1][#6:3]>>[#6:2][O:1][#6:3]", ("Swap a ring CH₂ for O", "جایگزینی ‎CH₂‎ حلقه با ‎O‎"),
     ("an ether oxygen accepts hydrogen bonds", "اکسیژن اتری پیوند هیدروژنی می‌پذیرد")),
]
_REACTIONS = {key: AllChem.ReactionFromSmarts(smarts) for key, smarts, _, _ in EDITS}


def _variants(mol: Chem.Mol, key: str, limit: int = 12) -> list[str]:
    """Canonical SMILES of the molecules one edit can produce (one per position)."""
    out: list[str] = []
    for products in _REACTIONS[key].RunReactants((mol,), maxProducts=40):
        prod = products[0]
        try:
            Chem.SanitizeMol(prod)
        except Exception:  # noqa: BLE001 - an edit that makes an impossible molecule is simply skipped
            continue
        clean, _reason = standardize(Chem.MolToSmiles(prod), DEFAULT_ELEMENTS)
        if clean is None:
            continue
        smi = Chem.MolToSmiles(clean)
        if smi not in out:
            out.append(smi)
        if len(out) >= limit:
            break
    return out


def suggest(predictor: "Predictor", mol: Chem.Mol, lang: str = "en", top: int = 6) -> dict:
    """Try each edit at every position, predict, and keep the best result of each edit."""
    base = predictor.predict_mol(mol)
    start = Chem.MolToSmiles(mol)
    candidates: list[tuple[str, str]] = [(key, smi) for key, *_ in EDITS for smi in _variants(mol, key) if smi != start]
    ideas: dict[str, dict] = {}
    if candidates:
        preds = predictor.predict_many([smi for _, smi in candidates])
        meta = {key: (label, why) for key, _, label, why in EDITS}
        for (key, smi), p in zip(candidates, preds):
            delta = float(p) - base
            if delta <= 0.05 or (key in ideas and ideas[key]["delta"] >= delta):
                continue
            m = Chem.MolFromSmiles(smi)
            ideas[key] = {"key": key, "label": _t(lang, meta[key][0]), "why": _t(lang, meta[key][1]), "smiles": smi,
                          "logS": round(float(p), 2), "delta": round(delta, 2), "fold": float(f"{10 ** delta:.2g}"),
                          "category": category(float(p)), "formula": rdMolDescriptors.CalcMolFormula(m)}
    best = sorted(ideas.values(), key=lambda d: -d["delta"])[:top]
    for d in best:
        d["svg"] = draw_svg(Chem.MolFromSmiles(d["smiles"]), size=(260, 200))
    return {"base_logS": round(base, 2), "tried": len(candidates), "ideas": best}


# ----------------------------------------------------------------------------- quiz
#: Quiz levels: which molecules, the smallest gap between the two measured values (log units),
#: and for "expert" a minimum similarity, so the two molecules look alike.
LEVELS: dict[str, dict] = {
    "easy":   {"pool": "familiar", "gap": 2.0, "similar": None},
    "medium": {"pool": "familiar", "gap": 1.0, "similar": None},
    "hard":   {"pool": "any", "gap": 1.0, "similar": None},
    "expert": {"pool": "any", "gap": 0.7, "similar": 0.45},
}


@lru_cache(maxsize=4)
def _quiz_pool(predictor: "Predictor", pool: str) -> list[dict]:
    """Molecules with a measured value: the familiar list, or the whole dataset."""
    out = []
    if pool == "familiar":
        for name, (smi, _cat) in COMMON.items():
            mol = Chem.MolFromSmiles(smi)
            m = predictor.measured(mol)
            if m and m["exact"]:
                out.append({"name": name, "smiles": Chem.MolToSmiles(mol), "measured": m["logS"], "row": None})
        return out
    ts = predictor.training_set
    for i, (name, smi, logS) in enumerate(zip(ts["name"], ts["smiles"], ts["logS"])):
        name = str(name).strip()
        if 3 <= len(name) <= 28 and ";" not in name and 5 <= sum(c.isupper() or c in "cnos" for c in smi) <= 20:
            out.append({"name": name.replace("_", " "), "smiles": smi, "measured": round(float(logS), 2), "row": i})
    return out


def _pairs(pool: list[dict], n: int, gap: float, rng: random.Random) -> list[list[dict]]:
    """Random pairs whose measured values differ by at least ``gap``."""
    rounds, used, tries = [], set(), 0
    while len(rounds) < n and tries < 4000:
        tries += 1
        a, b = rng.sample(pool, 2)
        if abs(a["measured"] - b["measured"]) < gap or a["name"] in used or b["name"] in used:
            continue
        used.update([a["name"], b["name"]])
        rounds.append([a, b])
    return rounds


def _look_alike_pairs(predictor: "Predictor", pool: list[dict], n: int, gap: float, min_sim: float,
                      rng: random.Random) -> list[list[dict]]:
    """Pairs of molecules that look alike (fingerprint similarity >= ``min_sim``) but dissolve differently."""
    from rdkit import DataStructs

    fps = [predictor.training_fps[m["row"]] for m in pool]
    order = list(range(len(pool)))
    rng.shuffle(order)
    rounds, used = [], set()
    for i in order:
        if len(rounds) >= n:
            break
        a = pool[i]
        if a["name"] in used:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        cands = [j for j, s in enumerate(sims) if j != i and min_sim <= s < 0.999
                 and abs(pool[j]["measured"] - a["measured"]) >= gap and pool[j]["name"] not in used
                 and pool[j]["name"].lower() != a["name"].lower()]
        if not cands:
            continue
        b = pool[rng.choice(cands)]
        used.update([a["name"], b["name"]])
        rounds.append([a, b] if rng.random() < 0.5 else [b, a])
    return rounds


def quiz(predictor: "Predictor", n: int = 8, level: str = "easy", lang: str = "en", seed: int | None = None,
         zoo=None) -> dict:
    """``n`` rounds of two molecules whose *measured* solubilities differ clearly.

    ``predictor`` is the opponent. If a ``zoo`` is given, every other model's answer is
    included too, so the website can show a leaderboard of all models.
    """
    level = level if level in LEVELS else "easy"
    cfg = LEVELS[level]
    rng = random.Random(seed)
    base = getattr(predictor, "base", predictor)
    pool = _quiz_pool(base, cfg["pool"])
    if cfg["similar"]:
        rounds = _look_alike_pairs(base, pool, n, cfg["gap"], cfg["similar"], rng)
    else:
        rounds = _pairs(pool, n, cfg["gap"], rng)
    flat = [m for pair in rounds for m in pair]
    mols = [Chem.MolFromSmiles(m["smiles"]) for m in flat]
    if zoo is not None:
        every = zoo.predict_all(mols)
        preds = every[predictor.model_id]
    else:
        every = {}
        preds = predictor.predict_many([m["smiles"] for m in flat]) if flat else []
    out = []
    for i, pair in enumerate(rounds):
        items = []
        for j, m in enumerate(pair):
            mol = mols[2 * i + j]
            display = (fa_name(m["name"]) if lang == "fa" else None) or m["name"]
            items.append({"name": display, "query": m["name"] if cfg["pool"] == "familiar" else m["smiles"],
                          "smiles": m["smiles"], "measured": m["measured"], "predicted": round(float(preds[2 * i + j]), 2),
                          "formula": rdMolDescriptors.CalcMolFormula(mol), "svg": draw_svg(mol, size=(300, 230))})
        answer = int(items[1]["measured"] > items[0]["measured"])
        r = {"molecules": items, "answer": answer, "model_answer": int(items[1]["predicted"] > items[0]["predicted"]),
             "gap": round(abs(items[0]["measured"] - items[1]["measured"]), 2)}
        if every:
            r["model_answers"] = {k: int(v[2 * i + 1] > v[2 * i]) for k, v in every.items()}
        out.append(r)
    return {"level": level, "levels": list(LEVELS), "rounds": out}


def warm_up(zoo) -> None:
    """Build the quiz pools in advance."""
    for pool in ("familiar", "any"):
        _quiz_pool(zoo.base, pool)


# ----------------------------------------------------------------------------- batch
def quick(predictor: "Predictor", mol: Chem.Mol, logS: float | None = None) -> dict:
    """The key numbers for one molecule, without the (slower) explanation (``logS``: already predicted)."""
    logS = predictor.predict_mol(mol) if logS is None else float(logS)
    dom = predictor.domain(mol, k=1, pred=logS)
    mw = Descriptors.MolWt(mol)
    hw = dom["half_width"]
    m = predictor.measured(mol)
    g = (10 ** logS) * mw
    return {"logS": round(logS, 2), "low": round(logS - hw, 2), "high": round(logS + hw, 2),
            "g_per_L": float(f"{g:.3g}"), "category": category(logS), "usp": usp_term(g),
            "in_domain": dom["in_domain"], "similarity": dom["max_similarity"], "confidence": dom.get("confidence"),
            "measured": m["logS"] if m else None, "classic": round(delaney_one(mol), 2),
            "formula": rdMolDescriptors.CalcMolFormula(mol), "mol_weight": round(mw, 2)}


def batch(predictor: "Predictor", queries: list[str], lang: str = "en") -> dict:
    """Predict many names/SMILES. PubChem is not used, so a batch never waits on the internet."""
    rows = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        salt = ionic.analyze_text(q, lang)
        if salt:
            rows.append({"query": q, "kind": "ionic", "name": salt["name"], "formula": salt["formula"],
                         "status": salt["status"], "status_label": salt["status_label"],
                         "measured_g_per_100mL": salt["measured"]["g_per_100mL"] if salt["measured"] else None})
            continue
        try:
            r = resolve(q, use_pubchem=False)
        except InputError as err:
            rows.append({"query": q, "kind": "error", "error": err.message})
            continue
        rows.append({"query": q, "kind": "organic", "name": r.name, "smiles": r.smiles, "_mol": r.mol})
    ok = [r for r in rows if r["kind"] == "organic"]
    # One call for all molecules: much faster than one call each.
    preds = predictor.predict_many([r["smiles"] for r in ok]) if ok else []
    for r, p in zip(ok, preds):
        r.update(quick(predictor, r.pop("_mol"), p))
    return {"rows": rows, "n_ok": len(ok), "n_salts": sum(r["kind"] == "ionic" for r in rows),
            "n_failed": sum(r["kind"] == "error" for r in rows)}


# ----------------------------------------------------------------------------- household check
#: (where you meet it, molecule name in the offline list or salt formula)
HOUSEHOLD: list[tuple[T, str]] = [
    (("Table sugar", "شکر"), "sucrose"), (("Honey and fruit", "عسل و میوه"), "fructose"),
    (("Vinegar", "سرکه"), "acetic acid"), (("Lemon juice", "آب‌لیمو"), "citric acid"),
    (("Vitamin C tablet", "قرص ویتامین ث"), "ascorbic acid"), (("Coffee and tea", "قهوه و چای"), "caffeine"),
    (("Aspirin tablet", "قرص آسپرین"), "aspirin"), (("Paracetamol tablet", "قرص استامینوفن"), "paracetamol"),
    (("Ibuprofen tablet", "قرص ایبوپروفن"), "ibuprofen"), (("Rubbing alcohol", "الکل ضدعفونی"), "2-propanol"),
    (("Nail polish remover", "لاک‌پاک‌کن"), "acetone"), (("Car antifreeze", "ضدیخ خودرو"), "ethylene glycol"),
    (("Hand cream (glycerin)", "کرم دست (گلیسیرین)"), "glycerol"), (("Vanilla", "وانیل"), "vanillin"),
    (("Mint", "نعنا"), "menthol"), (("Chili pepper", "فلفل تند"), "capsaicin"),
    (("Cinnamon", "دارچین"), "cinnamaldehyde"), (("Mothballs", "نفتالین"), "naphthalene"),
    (("Candle wax / fat", "چربی و شمع"), "stearic acid"), (("Egg yolk", "زردهٔ تخم‌مرغ"), "cholesterol"),
    (("Table salt", "نمک خوراکی"), "NaCl"), (("Baking soda", "جوش شیرین"), "NaHCO3"),
    (("Chalk / eggshell", "گچ تحریر / پوست تخم‌مرغ"), "CaCO3"), (("Plaster (gypsum)", "گچ ساختمانی"), "CaSO4"),
]


@lru_cache(maxsize=32)
def household(predictor: "Predictor", lang: str = "en") -> dict:
    """Everyday substances: what the lab measured vs what SoluAI (or the rules) says."""
    rows = []
    for where, query in HOUSEHOLD:
        salt = ionic.analyze_text(query, lang)
        if salt:
            rows.append({"where": _t(lang, where), "kind": "ionic", "name": salt["name"], "formula": salt["formula"],
                         "status": salt["status"], "status_label": salt["status_label"],
                         "measured_logS": salt["measured"]["logS"] if salt["measured"] else None,
                         "measured_g_per_L": salt["measured"]["g_per_L"] if salt["measured"] else None,
                         "agrees": salt["agrees_with_measurement"]})
            continue
        mol = Chem.MolFromSmiles(COMMON[query][0])
        q = quick(predictor, mol)
        rows.append({"where": _t(lang, where), "kind": "organic", "query": query,
                     "name": (fa_name(query) if lang == "fa" else None) or query, **q})
    organic = [r for r in rows if r["kind"] == "organic" and r["measured"] is not None]
    err = lambda k: round(float(np.mean([abs(r[k] - r["measured"]) for r in organic])), 2)  # noqa: E731
    return {"rows": rows, "mean_abs_error_model": err("logS"), "mean_abs_error_classic": err("classic"),
            "n_compared": len(organic)}
