"""Why did the model predict that?

SHAP gives every feature a number: how much it pushed this prediction up or
down compared with the average molecule. The SHAP values always add up exactly:

    prediction = average prediction + sum of all SHAP values

To show this *on the molecule*, each feature's push is shared among the atoms
responsible for it:

* a fingerprint fragment -> the atoms of that fragment;
* logP, polar surface area (TPSA), molar refractivity -> every atom according
  to its own contribution (RDKit computes these per atom);
* H-bond donors / acceptors -> the N and O atoms that donate / accept;
* aromatic features -> aromatic atoms; rotatable bonds -> their atoms; ...

Model v2 also uses about 200 extended descriptors. They are shared the same way
wherever a descriptor belongs to particular atoms:

* surface-area families (``SlogP_VSA3``, ``PEOE_VSA6`` ...) are "the surface of
  all atoms whose oiliness / charge / ... lies in one range" -> those atoms;
* functional-group counts (``fr_ester`` ...) -> the atoms of that group;
* shape, branching and charge-pattern indices describe the whole molecule.

Effects that belong to the whole molecule (overall size, shape, fragments that
are *absent*) are not drawn on atoms; they are listed separately.
Blue atoms raise solubility, red atoms lower it.
"""

from __future__ import annotations

import bisect
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Fragments, Lipinski, MolSurf, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.EState import EState, EState_VSA

from .features import DESCRIPTOR_NAMES, FAMILIES, PRETTY, descriptors, family
from .groups import GROUPS_FA, find_groups

if TYPE_CHECKING:
    from .predict import Predictor

BLUE = np.array([37, 99, 235]) / 255     # raises solubility
RED = np.array([220, 38, 38]) / 255      # lowers solubility
SIZE_FEATURES = {"MolWt", "HeavyAtomCount", "NumValenceElectrons"}
NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


# ----------------------------------------------------------------------------- atom attribution
def _atom_sets(mol: Chem.Mol) -> dict[str, list[int]]:
    """Which atoms are 'responsible' for each count-type descriptor."""
    atoms = list(range(mol.GetNumAtoms()))
    rot = Chem.MolFromSmarts("[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]")
    donors = [m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[N,O;!H0]"))]
    acceptors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O")]
    rot_atoms = sorted({i for m in mol.GetSubstructMatches(rot) for i in m})
    arom = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    return {
        "NumHDonors": donors,
        "NumHAcceptors": acceptors,
        "NumRotatableBonds": rot_atoms,
        "AromaticProportion": arom,
        "NumAromaticRings": arom,
        "RingCount": [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()],
        "FractionCSP3": [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"],
        "NumHeteroatoms": [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H")],
        "_all": atoms,
    }


@lru_cache(maxsize=1)
def _fragment_patterns() -> dict[str, Chem.Mol]:
    """SMARTS pattern of each RDKit functional-group descriptor (``fr_...``)."""
    out = {}
    with open(Fragments.defaultPatternFileName, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].startswith("fr_"):
                patt = Chem.MolFromSmarts(parts[2])
                if patt is not None:
                    out[parts[0]] = patt
    return out


#: Surface-area families: descriptor prefix -> (which per-atom property, bin edges).
_VSA_BINS = {"SlogP_VSA": ("logp", MolSurf.logpBins), "SMR_VSA": ("mr", MolSurf.mrBins),
             "PEOE_VSA": ("charge", MolSurf.chgBins), "EState_VSA": ("estate", EState_VSA.estateBins),
             "VSA_EState": ("area", EState_VSA.vsaBins)}


def extended_atom_sets(mol: Chem.Mol, names: list[str]) -> dict[str, list[int]]:
    """Atoms 'responsible' for each extended descriptor (missing key = whole-molecule descriptor).

    ``SlogP_VSA3`` is the surface area of the atoms whose logP contribution falls in the
    third range, so those atoms get its effect; ``fr_ester`` belongs to the ester atoms.
    An empty list means the group or range does not occur in this molecule.
    """
    out: dict[str, list[int]] = {}
    props: dict[str, np.ndarray] = {}

    def prop(kind: str) -> np.ndarray:
        if kind not in props:
            if kind in ("logp", "mr"):
                c = np.array(rdMolDescriptors._CalcCrippenContribs(mol))
                props["logp"], props["mr"] = c[:, 0], c[:, 1]
            elif kind == "charge":
                rdPartialCharges.ComputeGasteigerCharges(mol)
                props[kind] = np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()])
            elif kind == "estate":
                props[kind] = np.array(EState.EStateIndices(mol))
            else:
                props[kind] = np.array(list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0]))
        return props[kind]

    hetero_no = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O")]
    for name in names:
        if name.startswith("fr_"):
            patt = _fragment_patterns().get(name)
            if patt is not None:
                out[name] = sorted({i for m in mol.GetSubstructMatches(patt) for i in m})
        elif name == "NOCount":
            out[name] = hetero_no
        elif name == "NHOHCount":
            out[name] = [i for i in hetero_no if mol.GetAtomWithIdx(i).GetTotalNumHs() > 0]
        else:
            for prefix, (kind, bins) in _VSA_BINS.items():
                if name.startswith(prefix) and name[len(prefix):].isdigit():
                    k = int(name[len(prefix):]) - 1
                    out[name] = [i for i, v in enumerate(prop(kind)) if np.isfinite(v) and bisect.bisect_right(bins, v) == k]
                    break
    return out


def atom_weights(predictor: "Predictor", mol: Chem.Mol, shap: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Share SHAP values among atoms.

    Returns ``(weights, molecule_wide)`` where ``weights[i]`` is atom ``i``'s
    share (log units) and ``molecule_wide`` holds the effects not drawn on atoms.
    """
    n = mol.GetNumAtoms()
    w = np.zeros(n)
    wide: dict[str, float] = {"Overall size": 0.0, "Shape and charge pattern": 0.0,
                              "Compared with the average molecule": 0.0,
                              "Fragments not present": 0.0, "Other": 0.0}
    names = predictor.feature_names
    sets = _atom_sets(mol)
    extra = [n for n in names if n not in PRETTY and not n.startswith("FP_")]
    extra_sets = extended_atom_sets(mol, extra)
    crippen = np.array(rdMolDescriptors._CalcCrippenContribs(mol))
    per_atom = {"MolLogP": crippen[:, 0], "MolMR": crippen[:, 1],
                "TPSA": np.array(rdMolDescriptors._CalcTPSAContribs(mol))}

    for j, name in enumerate(names):
        s = float(shap[j])
        if name.startswith("FP_"):
            continue
        if name in SIZE_FEATURES:
            wide["Overall size"] += s
        elif name not in PRETTY:
            # Extended descriptor (model v2).
            idx = extra_sets.get(name)
            if idx:
                w[idx] += s / len(idx)
            elif idx is not None:          # a group / surface type this molecule does not have
                wide["Fragments not present"] += s
            else:
                fam = family(name)
                wide["Overall size" if fam == "size" else "Shape and charge pattern" if fam in ("shape", "charge") else "Other"] += s
        elif name in per_atom:
            # Each atom gets slope * (its own contribution), and the remainder is
            # shared evenly so the atom shares add up exactly to the SHAP value.
            # Each atom gets  slope x (its own contribution), e.g. an O-H oxygen has a
            # negative logP contribution, so it gets a positive (blue) share. What is left
            # over is "how this molecule compares with the average molecule" and goes to
            # the molecule-wide list, so everything still adds up exactly.
            c = per_atom[name]
            slope = predictor.shap_slopes.get(name, 0.0)
            w += slope * c
            wide["Compared with the average molecule"] += s - slope * c.sum()
        else:
            idx = sets.get(name) or []
            if idx:
                w[idx] += s / len(idx)
            else:
                wide["Other"] += s

    if any(n.startswith("FP_") for n in names):
        _, info = predictor.featurizer.fingerprint_with_info(mol)
        offset = names.index("FP_0")
        for bit in range(predictor.featurizer.n_bits):
            s = float(shap[offset + bit])
            if bit not in info:
                wide["Fragments not present"] += s
                continue
            envs = info[bit]
            for center, radius in envs:
                atoms = {center}
                for b in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center):
                    bond = mol.GetBondWithIdx(b)
                    atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                for a in atoms:
                    w[a] += s / len(envs) / len(atoms)
    return w, {k: round(float(v), 3) for k, v in wide.items() if abs(v) >= 0.005}


# ----------------------------------------------------------------------------- drawing
def _shade(weight: float, scale: float) -> tuple[float, float, float]:
    t = min(1.0, abs(weight) / scale)
    base = BLUE if weight > 0 else RED
    rgb = 1 - 0.85 * t * (1 - base)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def draw_svg(mol: Chem.Mol, weights: np.ndarray | None = None, size: tuple[int, int] = (420, 320)) -> str:
    """2D structure as SVG; with ``weights``, atoms are coloured blue (+) / red (-)."""
    m = rdMolDraw2D.PrepareMolForDrawing(mol)
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 2
    o.padding = 0.12
    o.fixedFontSize = 18 if size[0] >= 380 else 13
    o.additionalAtomLabelPadding = 0.12
    if weights is None:
        d.DrawMolecule(m)
    else:
        scale = max(0.15, float(np.max(np.abs(weights))) if len(weights) else 0.15)
        atoms = [i for i, x in enumerate(weights) if abs(x) > 0.02 * scale]
        colors = {i: _shade(weights[i], scale) for i in atoms}
        radii = {i: 0.35 + 0.2 * min(1.0, abs(weights[i]) / scale) for i in atoms}
        bonds, bcolors = [], {}
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in colors and j in colors and np.sign(weights[i]) == np.sign(weights[j]):
                bonds.append(b.GetIdx())
                bcolors[b.GetIdx()] = _shade((weights[i] + weights[j]) / 2, scale)
        o.fillHighlights = True
        o.highlightRadius = 0.4
        d.DrawMolecule(m, highlightAtoms=atoms, highlightAtomColors=colors, highlightAtomRadii=radii,
                       highlightBonds=bonds, highlightBondColors=bcolors)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.index("<svg"):]  # drop the XML header so it can be inlined in HTML


# ----------------------------------------------------------------------------- words
def _plural(text: str) -> str:
    """Turn a reason written for 'it' into one for 'they'."""
    for a, b in [("it is", "they are"), ("it forms", "they form"), ("it adds", "they add"),
                 ("it makes", "they make"), ("it also", "they also"), ("its ", "their "),
                 ("so it avoids", "so they avoid"), ("and packs", "and pack"), ("it avoids", "they avoid"), ("it ", "they ")]:
        text = text.replace(a, b)
    return text


def describe_groups(mol: Chem.Mol, weights: np.ndarray) -> list[dict[str, Any]]:
    """Functional groups with their summed atom effect, biggest effect first."""
    out = []
    for g, matches in find_groups(mol):
        if not matches:
            continue
        atoms = set().union(*matches)
        effect = float(weights[sorted(atoms)].sum())
        out.append({"key": g.key, "group": g, "count": len(matches), "effect": round(effect, 3),
                    "atoms": sorted(atoms)})
    return sorted(out, key=lambda x: -abs(x["effect"]))


FA_NUMBERS = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده"]


def _group_sentence_fa(item: dict[str, Any]) -> str:
    g, n, up = item["group"], item["count"], item["effect"] > 0
    name, why_up, why_down = GROUPS_FA[g.key]
    noun = name if n == 1 else f"{FA_NUMBERS[n] if n < len(FA_NUMBERS) else n} {name}"
    if up == g.polar:
        return f"{noun} انحلال‌پذیری را {'افزایش' if up else 'کاهش'} می‌دهد، چون {why_up if up else why_down}."
    return (f"برخلاف انتظار، مدل فکر می‌کند {noun} این‌جا انحلال‌پذیری را کمی {'بالا' if up else 'پایین'} می‌برد؛ "
            "اثر آن با بقیهٔ مولکول درهم شده است.")


def group_sentence(item: dict[str, Any], lang: str = "en") -> str:
    """'The two –OH groups increase solubility because they form hydrogen bonds with water.'"""
    if lang == "fa":
        return _group_sentence_fa(item)
    g, n, up = item["group"], item["count"], item["effect"] > 0
    word = NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)
    noun = g.label if n == 1 else f"{word} {g.plural}"
    if up == g.polar:  # the direction chemistry expects
        verb = ("increases" if up else "decreases") if n == 1 else ("increase" if up else "decrease")
        reason = g.why_up if up else g.why_down
        return f"The {noun} {verb} solubility because {reason if n == 1 else _plural(reason)}."
    verb = ("raises" if up else "lowers") if n == 1 else ("raise" if up else "lower")
    return (f"Unusually, the model thinks the {noun} slightly {verb} solubility here - "
            "its effect is mixed up with the rest of the molecule.")


def _ratio(r: float) -> str:
    """Readable ratio: 218 -> '220', 3129680 -> '3 million'."""
    if r < 1000:
        return f"{float(f'{r:.2g}'):,.0f}"
    if r < 1e6:
        return f"{r / 1e3:,.0f} thousand"
    if r < 1e9:
        return f"{r / 1e6:,.0f} million"
    return f"{r / 1e9:,.0f} billion"


def _ratio_fa(r: float) -> str:
    if r < 1000:
        return f"{float(f'{r:.2g}'):,.0f}"
    if r < 1e6:
        return f"{r / 1e3:,.0f} هزار"
    if r < 1e9:
        return f"{r / 1e6:,.0f} میلیون"
    return f"{r / 1e9:,.0f} میلیارد"


def _logp_sentence_fa(logp: float) -> str:
    r = 10 ** logp
    logp = round(logp, 1) + 0.0
    head = f"‎logP‎ آن ‎{logp:.1f}‎ است: "
    if abs(logp) < 0.3:
        return head + "در روغن و آب تقریباً به یک اندازه راحت است، پس نسبت به اندازه‌اش کاملاً قطبی است."
    if logp < 0:
        return head + f"آب را به روغن ترجیح می‌دهد (حدود {_ratio_fa(1 / r)} به 1)، پس آب‌دوست است."
    kind = "کمی چرب است" if logp < 2 else "نسبتاً چرب (آب‌گریز) است" if logp < 4 else "بسیار چرب (آب‌گریز) است"
    return head + f"روغن را حدود {_ratio_fa(r)} به 1 به آب ترجیح می‌دهد، پس {kind}."


def logp_sentence(logp: float, lang: str = "en") -> str:
    """Absolute chemistry reading of logP (not relative to the average molecule)."""
    if lang == "fa":
        return _logp_sentence_fa(logp)
    r = 10 ** logp
    logp = round(logp, 1) + 0.0  # "+ 0.0" turns -0.0 into 0.0 for display
    if abs(logp) < 0.3:
        return f"Its logP is {logp:.1f}: it is about equally happy in oil and in water, so it is quite polar for its size."
    if logp < 0:
        return f"Its logP is {logp:.1f}: it prefers water to oil (about {_ratio(1 / r)} to 1), so it is water-loving (hydrophilic)."
    if logp < 2:
        return f"Its logP is {logp:.1f}: it prefers oil to water about {_ratio(r)} to 1, so it is mildly oily."
    if logp < 4:
        return f"Its logP is {logp:.1f}: it prefers oil to water about {_ratio(r)} to 1, so it is quite oily (hydrophobic)."
    return f"Its logP is {logp:.1f}: it prefers oil to water about {_ratio(r)} to 1, so it is very oily (hydrophobic)."


def size_sentence(mw: float, lang: str = "en") -> str | None:
    """Mention size only when it is notable."""
    if lang == "fa":
        if mw >= 350:
            return f"مولکول بزرگی است (‎{mw:.0f} g/mol‎)، پس آب باید حفرهٔ بزرگی برای جا دادن آن باز کند."
        if mw <= 80:
            return f"مولکول کوچکی است (‎{mw:.0f} g/mol‎) و آب به‌آسانی دور آن را می‌گیرد."
        return None
    if mw >= 350:
        return f"It is a large molecule ({mw:.0f} g/mol), so water must open a big cavity to fit it in."
    if mw <= 80:
        return f"It is a small molecule ({mw:.0f} g/mol), which makes it easy for water to surround."
    return None


# ----------------------------------------------------------------------------- full analysis
#: Descriptive terms of the pharmacopoeias (USP/Ph. Eur.): mL of water needed to dissolve 1 g.
USP_TERMS = [(1, "very_soluble"), (10, "freely_soluble"), (30, "soluble"), (100, "sparingly_soluble"),
             (1000, "slightly_soluble"), (10000, "very_slightly_soluble"), (float("inf"), "practically_insoluble")]


def usp_term(g_per_L: float) -> str:
    """Pharmacopoeia word for a solubility, e.g. 'sparingly_soluble' (30-100 mL of water per gram)."""
    parts = 1000 / g_per_L if g_per_L > 0 else float("inf")
    return next(term for limit, term in USP_TERMS if parts < limit)


def analyze(predictor: "Predictor", mol: Chem.Mol, lang: str = "en") -> dict[str, Any]:
    """Everything the website shows about one molecule (``lang`` = "en" or "fa" for the sentences)."""
    from .baselines import delaney_one
    from .predict import category

    logS = predictor.predict_mol(mol)
    shap, base = predictor.shap_values(mol)
    weights, wide = atom_weights(predictor, mol, shap)
    dom = predictor.domain(mol, pred=logS)
    desc = dict(zip(DESCRIPTOR_NAMES, descriptors(mol)))
    mw = desc["MolWt"]

    # Top features for the bar chart: descriptors individually, fingerprint fragments grouped.
    names = predictor.feature_names
    contributions = [{"feature": PRETTY[n], "key": n, "value": round(float(desc[n]), 3), "shap": round(float(shap[j]), 3)}
                     for j, n in enumerate(names) if n in PRETTY]
    # Extended descriptors (model v2) are summed by family: one bar each.
    fam_sum: dict[str, float] = {}
    for j, n in enumerate(names):
        if n not in PRETTY and not n.startswith("FP_"):
            fam_sum[family(n)] = fam_sum.get(family(n), 0.0) + float(shap[j])
    contributions += [{"feature": FAMILIES[f], "key": f"family_{f}", "value": None, "shap": round(v, 3)}
                      for f, v in fam_sum.items()]
    fp_idx = [j for j, n in enumerate(names) if n.startswith("FP_")]
    if fp_idx:
        present = predictor.featurizer.fingerprint_with_info(mol)[1]
        offset = names.index("FP_0")
        s_present = float(sum(shap[offset + b] for b in present))
        s_absent = float(shap[fp_idx].sum()) - s_present
        contributions += [
            {"feature": "Fragments present", "key": "fp_present", "value": len(present), "shap": round(s_present, 3)},
            {"feature": "Fragments absent", "key": "fp_absent", "value": None, "shap": round(s_absent, 3)},
        ]
    contributions.sort(key=lambda c: -abs(c["shap"]))

    # Plain-language explanation.
    groups = describe_groups(mol, weights)
    sentences = [group_sentence(g, lang) for g in groups[:2] if abs(g["effect"]) >= 0.1]
    sentences.append(logp_sentence(desc["MolLogP"], lang))
    size = size_sentence(mw, lang)
    if size:
        sentences.append(size)

    hw = dom["half_width"]
    return {
        "kind": "organic",
        "measured": predictor.measured(mol),
        "classic": {"delaney": round(delaney_one(mol), 2)},
        "usp": usp_term((10 ** logS) * mw),
        "logS": round(logS, 2),
        "interval": [round(logS - hw, 2), round(logS + hw, 2)],
        "coverage": predictor.conformal["coverage"],
        "g_per_L": float(f"{(10 ** logS) * mw:.3g}"),
        "mg_per_L": float(f"{(10 ** logS) * mw * 1000:.3g}"),
        "category": category(logS),
        "base_value": round(base, 2),
        "mol_weight": round(mw, 2),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "descriptors": {k: round(float(v), 3) for k, v in desc.items()},
        "contributions": contributions[:10],
        "molecule_wide": wide,
        "groups": [{"name": g["group"].label if g["count"] == 1 else g["group"].plural, "count": g["count"],
                    "effect": g["effect"]} for g in groups],
        "explanation": sentences,
        "domain": dom,
        "svg": draw_svg(mol),
        "svg_explained": draw_svg(mol, weights),
        "atom_weights": [round(float(x), 3) for x in weights],
    }
