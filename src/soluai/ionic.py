"""Ionic compounds (salts): the textbook solubility rules.

The machine-learning model only knows single organic molecules. Salts such as
NaCl dissolve in a different way - the crystal splits into ions - so for them
SoluAI uses the **solubility rules** taught in high-school chemistry instead:

1. Salts of group 1 metals (Li+, Na+, K+) and ammonium (NH4+) are soluble.
2. Nitrates, acetates and hydrogen carbonates are soluble.
3. Chlorides, bromides and iodides are soluble - except with Ag+ and Pb2+.
4. Sulfates are soluble - except with Ba2+, Sr2+, Pb2+ (Ca2+ and Ag+: slightly).
5. Hydroxides are insoluble - except group 1, NH4+, Ba2+ (Ca2+, Sr2+: slightly).
6. Carbonates, phosphates and sulfides are insoluble - except group 1 and NH4+.

This is a rule engine, **not** a trained model, and the website says so.
"Soluble" here means more than about 1 g per 100 mL of water, "slightly
soluble" 0.1-1 g, "insoluble" less than 0.1 g.

Measured values (g per 100 mL of water at 20 degC) come from the Wikipedia
"Solubility table" (https://en.wikipedia.org/wiki/Solubility_table), which
collects them from the CRC Handbook and similar references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from math import gcd, log10

ATOMIC_MASS = {
    "H": 1.008, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078,
    "Mn": 54.938, "Fe": 55.845, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904,
    "Sr": 87.62, "Ag": 107.868, "I": 126.904, "Ba": 137.327, "Pb": 207.2,
}


@dataclass(frozen=True)
class Ion:
    """One ion: its formula, charge, names and what it is made of."""

    key: str
    formula: str               # as written inside a compound, e.g. "NH4"
    charge: int                # +1, +2, ... or -1, -2, ...
    en: str
    fa: str
    atoms: tuple[tuple[str, int], ...]
    aliases: tuple[str, ...] = ()       # other English names
    formula_alt: tuple[str, ...] = ()   # other ways to write the formula
    poly: bool = False                  # polyatomic -> needs brackets when there is more than one

    @property
    def mass(self) -> float:
        return sum(ATOMIC_MASS[e] * n for e, n in self.atoms)


CATIONS: list[Ion] = [
    Ion("Li", "Li", 1, "lithium", "لیتیم", (("Li", 1),)),
    Ion("Na", "Na", 1, "sodium", "سدیم", (("Na", 1),)),
    Ion("K", "K", 1, "potassium", "پتاسیم", (("K", 1),)),
    Ion("NH4", "NH4", 1, "ammonium", "آمونیوم", (("N", 1), ("H", 4)), poly=True),
    Ion("Mg", "Mg", 2, "magnesium", "منیزیم", (("Mg", 1),)),
    Ion("Ca", "Ca", 2, "calcium", "کلسیم", (("Ca", 1),)),
    Ion("Sr", "Sr", 2, "strontium", "استرانسیم", (("Sr", 1),)),
    Ion("Ba", "Ba", 2, "barium", "باریم", (("Ba", 1),)),
    Ion("Al", "Al", 3, "aluminium", "آلومینیم", (("Al", 1),), aliases=("aluminum",)),
    Ion("Mn", "Mn", 2, "manganese(II)", "منگنز(II)", (("Mn", 1),), aliases=("manganese",)),
    Ion("Fe2", "Fe", 2, "iron(II)", "آهن(II)", (("Fe", 1),), aliases=("ferrous",)),
    Ion("Fe3", "Fe", 3, "iron(III)", "آهن(III)", (("Fe", 1),), aliases=("ferric",)),
    Ion("Ni", "Ni", 2, "nickel(II)", "نیکل(II)", (("Ni", 1),), aliases=("nickel",)),
    Ion("Cu", "Cu", 2, "copper(II)", "مس(II)", (("Cu", 1),), aliases=("copper", "cupric")),
    Ion("Zn", "Zn", 2, "zinc", "روی", (("Zn", 1),)),
    Ion("Ag", "Ag", 1, "silver", "نقره", (("Ag", 1),)),
    Ion("Pb", "Pb", 2, "lead(II)", "سرب(II)", (("Pb", 1),), aliases=("lead",)),
]

ANIONS: list[Ion] = [
    Ion("NO3", "NO3", -1, "nitrate", "نیترات", (("N", 1), ("O", 3)), poly=True),
    Ion("CH3COO", "CH3COO", -1, "acetate", "استات", (("C", 2), ("H", 3), ("O", 2)),
        aliases=("ethanoate",), formula_alt=("C2H3O2",), poly=True),
    Ion("HCO3", "HCO3", -1, "hydrogen carbonate", "هیدروژن کربنات", (("H", 1), ("C", 1), ("O", 3)),
        aliases=("bicarbonate", "hydrogencarbonate"), poly=True),
    Ion("F", "F", -1, "fluoride", "فلوئورید", (("F", 1),)),
    Ion("Cl", "Cl", -1, "chloride", "کلرید", (("Cl", 1),)),
    Ion("Br", "Br", -1, "bromide", "برمید", (("Br", 1),)),
    Ion("I", "I", -1, "iodide", "یدید", (("I", 1),)),
    Ion("SO4", "SO4", -2, "sulfate", "سولفات", (("S", 1), ("O", 4)), aliases=("sulphate",), poly=True),
    Ion("OH", "OH", -1, "hydroxide", "هیدروکسید", (("O", 1), ("H", 1)), poly=True),
    Ion("CO3", "CO3", -2, "carbonate", "کربنات", (("C", 1), ("O", 3)), poly=True),
    Ion("PO4", "PO4", -3, "phosphate", "فسفات", (("P", 1), ("O", 4)), poly=True),
    Ion("S", "S", -2, "sulfide", "سولفید", (("S", 1),), aliases=("sulphide",)),
]

CATION = {c.key: c for c in CATIONS}
ANION = {a.key: a for a in ANIONS}
GROUP1 = {"Li", "Na", "K", "NH4"}

#: The six rules, in the order they are applied (English, Persian).
RULES: dict[str, tuple[str, str]] = {
    "alkali": ("Salts of group 1 metals (Li⁺, Na⁺, K⁺) and ammonium (NH₄⁺) are soluble.",
               "نمک‌های فلزهای گروه ۱ (‎Li⁺، Na⁺، K⁺‎) و آمونیوم (‎NH₄⁺‎) محلول‌اند."),
    "nitrate": ("Nitrates, acetates and hydrogen carbonates are soluble.",
                "نیترات‌ها، استات‌ها و هیدروژن کربنات‌ها محلول‌اند."),
    "halide": ("Chlorides, bromides and iodides are soluble, except with Ag⁺ and Pb²⁺.",
               "کلریدها، برمیدها و یدیدها محلول‌اند؛ به‌جز با ‎Ag⁺‎ و ‎Pb²⁺‎."),
    "fluoride": ("Fluorides are soluble, except with Mg²⁺, Ca²⁺, Sr²⁺, Ba²⁺ and Pb²⁺.",
                 "فلوئوریدها محلول‌اند؛ به‌جز با ‎Mg²⁺، Ca²⁺، Sr²⁺، Ba²⁺‎ و ‎Pb²⁺‎."),
    "sulfate": ("Sulfates are soluble, except with Ba²⁺, Sr²⁺ and Pb²⁺ (Ca²⁺ and Ag⁺ are only slightly soluble).",
                "سولفات‌ها محلول‌اند؛ به‌جز با ‎Ba²⁺، Sr²⁺‎ و ‎Pb²⁺‎ (‎Ca²⁺‎ و ‎Ag⁺‎ فقط کم‌محلول‌اند)."),
    "hydroxide": ("Hydroxides are insoluble, except with group 1 metals, NH₄⁺ and Ba²⁺ (Ca²⁺ and Sr²⁺ are slightly soluble).",
                  "هیدروکسیدها نامحلول‌اند؛ به‌جز با فلزهای گروه ۱، ‎NH₄⁺‎ و ‎Ba²⁺‎ (‎Ca²⁺‎ و ‎Sr²⁺‎ کم‌محلول‌اند)."),
    "carbonate": ("Carbonates and phosphates are insoluble, except with group 1 metals and NH₄⁺.",
                  "کربنات‌ها و فسفات‌ها نامحلول‌اند؛ به‌جز با فلزهای گروه ۱ و ‎NH₄⁺‎."),
    "sulfide": ("Sulfides are insoluble, except with group 1 metals, NH₄⁺, Ca²⁺, Sr²⁺ and Ba²⁺.",
                "سولفیدها نامحلول‌اند؛ به‌جز با فلزهای گروه ۱، ‎NH₄⁺، Ca²⁺، Sr²⁺‎ و ‎Ba²⁺‎."),
}
RULE_ORDER = list(RULES)

#: Pairs that do not simply dissolve: they react with water or do not exist as a solid.
UNSTABLE = {("Al", "CO3"), ("Fe3", "CO3"), ("Al", "S"), ("Fe3", "S"), ("Mg", "S"), ("Fe3", "I"),
            ("Cu", "I"), ("Ag", "OH"), ("NH4", "OH"), ("Al", "HCO3"), ("Fe3", "HCO3"), ("Cu", "CO3")}

#: Measured solubility, g per 100 mL water at 20 degC (Wikipedia "Solubility table").
MEASURED: dict[tuple[str, str], float] = {
    ("Na", "Cl"): 35.89, ("K", "Cl"): 34.2, ("K", "NO3"): 31.6, ("Na", "NO3"): 87.6, ("Ag", "NO3"): 216,
    ("Ag", "Cl"): 1.923e-4, ("Ag", "Br"): 1.328e-5, ("Pb", "Cl"): 1.08, ("Pb", "I"): 0.076,
    ("Pb", "SO4"): 0.00443, ("Pb", "NO3"): 54.3, ("Ba", "SO4"): 2.448e-4, ("Ba", "Cl"): 35.8,
    ("Ba", "OH"): 3.89, ("Ba", "NO3"): 9.02, ("Ca", "CO3"): 6.17e-4, ("Ca", "SO4"): 0.255,
    ("Ca", "OH"): 0.173, ("Ca", "Cl"): 74.5, ("Mg", "OH"): 9.628e-4, ("Mg", "SO4"): 35.1,
    ("Mg", "Cl"): 54.6, ("Na", "CO3"): 21.5, ("Na", "HCO3"): 9.6, ("Na", "OH"): 109, ("K", "OH"): 112,
    ("Na", "SO4"): 19.5, ("Cu", "SO4"): 32, ("Cu", "OH"): 1.722e-6, ("Fe3", "OH"): 2.097e-9,
    ("Fe2", "SO4"): 28.8, ("Zn", "SO4"): 53.8, ("NH4", "Cl"): 37.2, ("NH4", "NO3"): 192,
    ("K", "I"): 144, ("K", "Br"): 65.3, ("Li", "Cl"): 83.5, ("Sr", "SO4"): 0.0132, ("Ag", "SO4"): 0.8,
    ("Al", "OH"): 2.262e-8, ("Al", "SO4"): 36.4, ("Ca", "PO4"): 0.002, ("Ba", "CO3"): 2.4e-3,
    ("Mg", "CO3"): 0.039, ("Ag", "CH3COO"): 1.05, ("Na", "CH3COO"): 46.4, ("NH4", "SO4"): 75.4,
    ("K", "SO4"): 11.1, ("Na", "F"): 4.06, ("Ca", "NO3"): 121.2,
}

#: Everyday names (English and Persian) -> (cation, anion).
COMMON_NAMES: dict[str, tuple[str, str]] = {
    "salt": ("Na", "Cl"), "table salt": ("Na", "Cl"), "rock salt": ("Na", "Cl"), "baking soda": ("Na", "HCO3"),
    "washing soda": ("Na", "CO3"), "soda ash": ("Na", "CO3"), "chalk": ("Ca", "CO3"), "limestone": ("Ca", "CO3"),
    "marble": ("Ca", "CO3"), "lye": ("Na", "OH"), "caustic soda": ("Na", "OH"), "caustic potash": ("K", "OH"),
    "gypsum": ("Ca", "SO4"), "slaked lime": ("Ca", "OH"), "limewater": ("Ca", "OH"),
    "milk of magnesia": ("Mg", "OH"), "epsom salt": ("Mg", "SO4"), "saltpeter": ("K", "NO3"),
    "saltpetre": ("K", "NO3"), "potash": ("K", "CO3"), "blue vitriol": ("Cu", "SO4"), "rust": ("Fe3", "OH"),
    "نمک": ("Na", "Cl"), "نمک خوراکی": ("Na", "Cl"), "نمک طعام": ("Na", "Cl"), "جوش شیرین": ("Na", "HCO3"),
    "گچ": ("Ca", "SO4"), "سنگ آهک": ("Ca", "CO3"), "آهک مرده": ("Ca", "OH"), "آب آهک": ("Ca", "OH"),
    "سود سوزآور": ("Na", "OH"), "شوره": ("K", "NO3"), "کات کبود": ("Cu", "SO4"), "شیر منیزی": ("Mg", "OH"),
}

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_PLAIN = str.maketrans("₀₁₂₃₄₅₆₇₈₉۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


# ----------------------------------------------------------------------------- formulas
def _counts(c: Ion, a: Ion) -> tuple[int, int]:
    """How many cations and anions make a neutral compound (e.g. Al2(SO4)3 -> 2, 3)."""
    g = gcd(c.charge, -a.charge)
    return -a.charge // g, c.charge // g


def _part(formula: str, n: int, poly: bool) -> str:
    if n == 1:
        return formula
    return f"({formula}){n}" if poly else f"{formula}{n}"


def formula(c: Ion, a: Ion, pretty: bool = True) -> str:
    """Formula of the neutral compound, e.g. ``Ca(OH)₂``."""
    nc, na = _counts(c, a)
    text = _part(c.formula, nc, c.poly) + _part(a.formula, na, a.poly)
    return re.sub(r"\d+", lambda m: m.group().translate(_SUB), text) if pretty else text


def ion_label(ion: Ion) -> str:
    """An ion with its charge, e.g. ``SO₄²⁻``."""
    n = abs(ion.charge)
    sign = "+" if ion.charge > 0 else "-"
    return ion.formula.translate(_SUB) + (("" if n == 1 else str(n)) + sign).translate(_SUP)


def molar_mass(c: Ion, a: Ion) -> float:
    nc, na = _counts(c, a)
    return nc * c.mass + na * a.mass


# ----------------------------------------------------------------------------- the rules
def verdict(c: Ion, a: Ion) -> tuple[str, str, bool]:
    """Apply the rules. Returns ``(status, rule_key, is_exception)``.

    ``status`` is "soluble", "slightly" or "insoluble".
    """
    ck, ak = c.key, a.key
    if ck in GROUP1:
        if (ck, ak) in {("Li", "CO3"), ("Li", "F")}:
            return "slightly", "alkali", True
        if (ck, ak) == ("Li", "PO4"):
            return "insoluble", "alkali", True
        return "soluble", "alkali", False
    if ak in ("NO3", "CH3COO", "HCO3"):
        if (ck, ak) == ("Ag", "CH3COO"):
            return "slightly", "nitrate", True
        return "soluble", "nitrate", False
    if ak in ("Cl", "Br", "I"):
        if ck == "Ag" or (ck, ak) == ("Pb", "I"):
            return "insoluble", "halide", True
        if ck == "Pb":
            return "slightly", "halide", True
        return "soluble", "halide", False
    if ak == "F":
        if ck in ("Mg", "Ca", "Sr", "Pb"):
            return "insoluble", "fluoride", True
        if ck == "Ba":
            return "slightly", "fluoride", True
        return "soluble", "fluoride", False
    if ak == "SO4":
        if ck in ("Ba", "Sr", "Pb"):
            return "insoluble", "sulfate", True
        if ck in ("Ca", "Ag"):
            return "slightly", "sulfate", True
        return "soluble", "sulfate", False
    if ak == "OH":
        if ck == "Ba":
            return "soluble", "hydroxide", True
        if ck in ("Ca", "Sr"):
            return "slightly", "hydroxide", True
        return "insoluble", "hydroxide", False
    if ak == "S":
        if ck in ("Ca", "Sr", "Ba"):
            return "soluble", "sulfide", True
        return "insoluble", "sulfide", False
    return "insoluble", "carbonate", False   # carbonates and phosphates


# ----------------------------------------------------------------------------- reading the input
def _norm(text: str) -> str:
    text = text.strip().translate(_PLAIN).replace("‌", " ").replace("·", " ")
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=1)
def _lookup_tables() -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """``(formulas, lower-case formulas, names)`` -> (cation key, anion key)."""
    formulas: dict[str, tuple[str, str]] = {}
    names: dict[str, tuple[str, str]] = dict(COMMON_NAMES)
    for c in CATIONS:
        for a in ANIONS:
            pair = (c.key, a.key)
            nc, na = _counts(c, a)
            for af in (a.formula, *a.formula_alt):
                cat, an = _part(c.formula, nc, c.poly), _part(af, na, a.poly or len(af) > 2)
                formulas.setdefault(cat + an, pair)
                if a.key == "CH3COO":                      # CH3COONa, (CH3COO)2Ca
                    formulas.setdefault(an + cat, pair)
            for cn in (c.en, *c.aliases):
                for an_ in (a.en, *a.aliases):
                    names.setdefault(f"{cn} {an_}", pair)
                    names.setdefault(re.sub(r"\((i+)\)", r" (\1)", f"{cn} {an_}"), pair)
            names.setdefault(f"{c.fa} {a.fa}", pair)
            names.setdefault(f"{re.sub(r'\(.*\)', '', c.fa)} {a.fa}", pair)
    # Fe2+/Fe3+ share formulas such as "FeCl2"/"FeCl3" - those are distinct, but drop true clashes.
    lower: dict[str, tuple[str, str]] = {}
    clash = set()
    for f, pair in formulas.items():
        k = f.lower()
        if k in lower and lower[k] != pair:
            clash.add(k)
        lower[k] = pair
    for k in clash:
        lower.pop(k, None)
    return formulas, lower, {k.lower(): v for k, v in names.items()}


def parse(text: str) -> tuple[Ion, Ion] | None:
    """Recognise an ionic compound from its formula or name (English or Persian)."""
    if not text:
        return None
    t = _norm(text)
    formulas, lower, names = _lookup_tables()
    compact = t.replace(" ", "")
    pair = formulas.get(compact) or names.get(t.lower()) or names.get(t.lower().replace("(ii)", "(II)").replace("(iii)", "(III)").lower())
    if pair is None:
        key = t.lower().replace(" (", "(")
        pair = names.get(key) or names.get(key.replace("(", " ("))
    if pair is None:                                          # forgiving about capitals: "nacl"
        pair = lower.get(compact.lower())
    return (CATION[pair[0]], ANION[pair[1]]) if pair else None


# ----------------------------------------------------------------------------- results
STATUS_LABEL = {"soluble": ("Soluble", "محلول"), "slightly": ("Slightly soluble", "کم‌محلول"),
                "insoluble": ("Insoluble", "نامحلول")}


def _pick(lang: str, pair: tuple[str, str]) -> str:
    return pair[1] if lang == "fa" else pair[0]


def describe(c: Ion, a: Ion, lang: str = "en") -> dict:
    """Everything the website shows about one salt."""
    status, rule, exception = verdict(c, a)
    nc, na = _counts(c, a)
    mm = molar_mass(c, a)
    f = formula(c, a)
    coef = lambda n: "" if n == 1 else f"{n} "  # noqa: E731
    arrow = "→" if status == "soluble" else "⇌"
    equation = f"{f}(s) {arrow} {coef(nc)}{ion_label(c)}(aq) + {coef(na)}{ion_label(a)}(aq)"
    measured = None
    if (c.key, a.key) in MEASURED:
        g100 = MEASURED[(c.key, a.key)]
        mol_l = g100 * 10 / mm
        measured = {"g_per_100mL": g100, "g_per_L": float(f"{g100 * 10:.3g}"),
                    "mol_per_L": float(f"{mol_l:.3g}"), "logS": round(log10(mol_l), 2)}
    unstable = (c.key, a.key) in UNSTABLE
    if lang == "fa":
        name = f"{c.fa} {a.fa}"
        why = ("این ترکیب از قاعده پیروی می‌کند." if not exception else "این ترکیب یکی از استثناهای قاعده است.")
        note = ("این ترکیب در آب پایدار نیست: به‌جای حل‌شدن ساده، با آب واکنش می‌دهد یا اصلاً به‌صورت جامد وجود ندارد."
                if unstable else "")
    else:
        name = f"{c.en} {a.en}"
        why = "This compound follows the rule." if not exception else "This compound is one of the exceptions to the rule."
        note = ("This compound is not stable in water: it reacts with water instead of simply dissolving, "
                "or does not exist as a solid at all." if unstable else "")
    return {
        "kind": "ionic", "name": name, "name_en": f"{c.en} {a.en}", "formula": f, "molar_mass": round(mm, 2),
        "cation": {"key": c.key, "label": ion_label(c), "name": _pick(lang, (c.en, c.fa)), "count": nc},
        "anion": {"key": a.key, "label": ion_label(a), "name": _pick(lang, (a.en, a.fa)), "count": na},
        "status": status, "status_label": _pick(lang, STATUS_LABEL[status]),
        "rule": rule, "rule_number": RULE_ORDER.index(rule) + 1, "rule_text": _pick(lang, RULES[rule]),
        "exception": exception, "why": why, "note": note, "equation": equation, "measured": measured,
        "agrees_with_measurement": None if measured is None else status in _statuses_from_grams(measured["g_per_100mL"]),
    }


def _statuses_from_grams(g100: float) -> set[str]:
    """Classes a measured value is compatible with (values near a boundary fit both sides)."""
    out = {"soluble" if g100 >= 1 else "slightly" if g100 >= 0.1 else "insoluble"}
    if 1 / 1.5 <= g100 <= 1.5:
        out |= {"soluble", "slightly"}
    if 0.1 / 1.5 <= g100 <= 0.15:
        out |= {"slightly", "insoluble"}
    return out


def analyze_text(text: str, lang: str = "en") -> dict | None:
    """Rule-based answer for a formula or name, or ``None`` if it is not a known salt."""
    pair = parse(text)
    return describe(*pair, lang=lang) if pair else None


def table(lang: str = "en") -> dict:
    """The whole solubility table (every cation with every anion) for the Salts page."""
    return {
        "cations": [{"key": c.key, "label": ion_label(c), "name": _pick(lang, (c.en, c.fa))} for c in CATIONS],
        "anions": [{"key": a.key, "label": ion_label(a), "name": _pick(lang, (a.en, a.fa))} for a in ANIONS],
        "cells": {f"{c.key}|{a.key}": {"status": verdict(c, a)[0], "formula": formula(c, a),
                                       "exception": verdict(c, a)[2], "unstable": (c.key, a.key) in UNSTABLE,
                                       "measured": MEASURED.get((c.key, a.key))}
                  for c in CATIONS for a in ANIONS},
        "rules": [{"key": k, "text": _pick(lang, RULES[k])} for k in RULE_ORDER],
        "status_labels": {k: _pick(lang, v) for k, v in STATUS_LABEL.items()},
    }


# ----------------------------------------------------------------------------- mixing two solutions
def _balance(c1: Ion, a1: Ion, c2: Ion, a2: Ion) -> tuple[int, int, int, int]:
    """Smallest whole-number coefficients for  salt1 + salt2 -> (c1 a2) + (c2 a1)."""
    p, q = _counts(c1, a1)
    r, s = _counts(c2, a2)
    u, v = _counts(c1, a2)
    w, x = _counts(c2, a1)
    for n1 in range(1, 13):
        for n2 in range(1, 13):
            if (n1 * p) % u or (n2 * r) % w:
                continue
            m1, m2 = n1 * p // u, n2 * r // w
            if n1 * q == m2 * x and n2 * s == m1 * v:
                return n1, n2, m1, m2
    return 1, 1, 1, 1


def mix(c1: Ion, a1: Ion, c2: Ion, a2: Ion, lang: str = "en") -> dict:
    """What happens when solutions of two salts are mixed? (double displacement)."""
    fa = lang == "fa"
    reactants = [describe(c1, a1, lang), describe(c2, a2, lang)]
    not_dissolved = [r for r in reactants if r["status"] != "soluble"]
    products = [] if (c1.key == c2.key or a1.key == a2.key) else [(c1, a2), (c2, a1)]
    solids = [(c, a) for c, a in products if verdict(c, a)[0] != "soluble" and (c.key, a.key) not in UNSTABLE]
    coef = lambda n: "" if n == 1 else f"{n} "  # noqa: E731
    out = {"reactants": reactants, "products": [describe(c, a, lang) for c, a in products],
           "precipitates": [describe(c, a, lang) for c, a in solids], "equation": None, "net_ionic": []}
    if not_dissolved:
        f = not_dissolved[0]["formula"]
        out["message"] = (f"{f} خودش در آب خوب حل نمی‌شود، پس نمی‌توان از آن محلول ساخت. دو نمک محلول انتخاب کنید."
                          if fa else f"{f} does not dissolve well in water, so you cannot make a solution of it. Pick two soluble salts.")
        out["precipitates"] = []
        return out
    if not products:
        out["message"] = ("این دو نمک یک یون مشترک دارند؛ جابه‌جایی یون‌ها ترکیب تازه‌ای نمی‌سازد."
                          if fa else "These two salts share an ion, so swapping partners gives nothing new.")
        return out
    n1, n2, m1, m2 = _balance(c1, a1, c2, a2)
    state = lambda c, a: "(s)" if (c, a) in solids else "(aq)"  # noqa: E731
    out["equation"] = (f"{coef(n1)}{formula(c1, a1)}(aq) + {coef(n2)}{formula(c2, a2)}(aq) → "
                       f"{coef(m1)}{formula(c1, a2)}{state(c1, a2)} + {coef(m2)}{formula(c2, a1)}{state(c2, a1)}")
    for c, a in solids:
        nc, na = _counts(c, a)
        out["net_ionic"].append(f"{coef(nc)}{ion_label(c)}(aq) + {coef(na)}{ion_label(a)}(aq) → {formula(c, a)}(s)")
    if solids:
        names = " و ".join(d["formula"] for d in out["precipitates"]) if fa else " and ".join(d["formula"] for d in out["precipitates"])
        out["message"] = (f"رسوب تشکیل می‌شود: {names}. بقیهٔ یون‌ها در محلول می‌مانند (یون‌های تماشاگر)."
                          if fa else f"A precipitate forms: {names}. The other ions stay in solution (spectator ions).")
    else:
        out["message"] = ("رسوبی تشکیل نمی‌شود: هر دو فراوردهٔ ممکن محلول‌اند، پس همهٔ یون‌ها در محلول می‌مانند."
                          if fa else "No precipitate: both possible products are soluble, so all the ions simply stay in solution.")
    return out
