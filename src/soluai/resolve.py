"""Turning what the user typed into a molecule, with friendly error messages.

Order of attempts:
1. the offline list of common molecules (fast, works without internet);
2. a SMILES string (e.g. ``CCO`` for ethanol);
3. a PubChem name lookup over the internet;
4. otherwise: an error with "did you mean ..." suggestions.
"""

from __future__ import annotations

import difflib
import re
import urllib.parse
from dataclasses import dataclass

import requests
from rdkit import Chem, RDLogger

from .common_molecules import INORGANIC_NAMES, WATER_NAMES, all_names, lookup
from .data import DEFAULT_ELEMENTS, standardize

RDLogger.DisableLog("rdApp.*")

MAX_LEN = 300
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$/\\%.:*]+$")


class InputError(ValueError):
    """A problem with the user's input. ``str(err)`` is a friendly message."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass
class Resolved:
    """A molecule that is ready for prediction."""

    mol: Chem.Mol
    smiles: str   # canonical SMILES
    name: str     # display name (may equal the SMILES)
    source: str   # "list", "smiles" or "pubchem"


def _make(mol: Chem.Mol, name: str, source: str) -> Resolved:
    """Build a :class:`Resolved` with the canonical SMILES filled in."""
    return Resolved(mol, Chem.MolToSmiles(mol), name, source)


def looks_like_smiles(text: str) -> bool:
    """Rough check: SMILES has no spaces and only SMILES characters."""
    return " " not in text and bool(_SMILES_CHARS.match(text))


def pubchem_smiles(name: str, timeout: float = 6.0) -> tuple[str, str] | None:
    """Ask PubChem for a molecule by name. Returns ``(title, smiles)`` or ``None``.

    Returns ``None`` if the name is unknown or PubChem cannot be reached
    (e.g. no internet) - the caller then shows a friendly message.
    """
    quoted = urllib.parse.quote(name)
    for prop in ("SMILES,Title", "IsomericSMILES,Title"):  # PubChem renamed this property in 2025
        try:
            r = requests.get(PUBCHEM.format(quoted, prop), timeout=timeout)
        except requests.RequestException:
            return None
        if r.status_code == 404:
            return None
        if r.ok:
            try:
                p = r.json()["PropertyTable"]["Properties"][0]
            except (KeyError, IndexError, ValueError):
                return None
            smi = p.get("SMILES") or p.get("IsomericSMILES") or p.get("CanonicalSMILES")
            if smi:
                return p.get("Title", name), smi
    return None


def check_scope(mol: Chem.Mol, original: str) -> Chem.Mol:
    """Reject molecules the model was not built for (salts, inorganics)."""
    clean, reason = standardize(Chem.MolToSmiles(mol), DEFAULT_ELEMENTS)
    if clean is not None:
        return clean
    messages = {
        "salt_or_mixture": (
            "This looks like a salt or a mixture (several separate pieces).",
            "SoluAI only handles single organic molecules. Salts dissolve by splitting into ions, "
            "which the model was not trained on. Try the neutral organic part instead.",
        ),
        "no_carbon": (
            "This is an inorganic substance (it has no carbon).",
            "SoluAI is trained on organic molecules only; inorganic solubility follows different rules "
            "(lattice energy, ion hydration).",
        ),
        "non_organic_element": (
            "This molecule contains an element (e.g. a metal) that is outside SoluAI's training data.",
            "Allowed elements: C, H, N, O, S, P, F, Cl, Br, I, B, Si, Se.",
        ),
    }
    msg, hint = messages.get(reason, ("This molecule could not be processed.", ""))
    raise InputError(msg, hint)


def resolve(text: str, use_pubchem: bool = True) -> Resolved:
    """Turn a name or SMILES into a :class:`Resolved` molecule, or raise :class:`InputError`."""
    if text is None or not text.strip():
        raise InputError("Please type a molecule name or a SMILES string.", "Example: ethanol, or CCO")
    text = text.strip()
    if len(text) > MAX_LEN:
        raise InputError(f"That input is too long (max {MAX_LEN} characters).")
    if text.lower() in WATER_NAMES:
        raise InputError(
            "Water is the solvent itself, so its 'solubility in water' is not defined - it mixes with itself completely!",
            "Try a molecule that dissolves in water, such as ethanol or glucose.",
        )

    if text.lower() in INORGANIC_NAMES:
        raise InputError(
            f"{text} is an inorganic substance - outside what SoluAI was trained on.",
            "SoluAI predicts organic (carbon-based) molecules. Inorganic salts dissolve by splitting into ions, "
            "which follows different rules (lattice energy, ion hydration).",
        )

    hit = lookup(text)
    if hit:
        name, smi = hit
        mol = Chem.MolFromSmiles(smi)
        return _make(check_scope(mol, text), name, "list")

    if looks_like_smiles(text):
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            return _make(check_scope(mol, text), text, "smiles")

    if use_pubchem and not re.search(r"[\[\]=#()\\/]", text):
        found = pubchem_smiles(text)
        if found:
            title, smi = found
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return _make(check_scope(mol, text), title, "pubchem")

    suggestions = difflib.get_close_matches(text.lower(), all_names(), n=3, cutoff=0.6)
    # Only blame the SMILES if the text has SMILES-only characters (ring digits, brackets, bonds).
    if looks_like_smiles(text) and re.search(r"[0-9()\[\]=#@/\\+]", text) and not suggestions:
        raise InputError(
            f"'{text}' is not a valid SMILES string.",
            "Check that brackets and ring numbers are closed, e.g. benzene is c1ccccc1.",
        )
    hint = f"Did you mean: {', '.join(suggestions)}?" if suggestions else (
        "Check the spelling, or type a SMILES string instead. Name lookup needs an internet connection."
    )
    raise InputError(f"Sorry, I couldn't find a molecule called '{text}'.", hint)
