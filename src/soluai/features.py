"""Turning a molecule into numbers a model can learn from.

Three kinds of features are used:

* **Descriptors** - 14 named, chemically meaningful numbers such as logP
  (how oily the molecule is), molecular weight, or the number of hydrogen-bond
  donors. A chemist can read every one of them.
* **Morgan fingerprint** - 1024 yes/no bits. Each bit says "this small
  fragment (an atom plus its neighbours up to 2 bonds away) is present". It
  lets the model notice specific groups such as -OH or -COOH.
* **Extended descriptors** (model v2) - every other descriptor RDKit can
  compute from the 2D structure (about 200 more): surface area split by
  charge / oiliness (the "VSA" families), shape and branching indices,
  partial charges and counts of 85 functional groups. They are less readable
  one by one, so explanations group them into a few families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

#: Descriptor names in a fixed order, with a plain-language explanation of each.
DESCRIPTORS: dict[str, str] = {
    "MolLogP": "logP: how much the molecule prefers oil over water",
    "MolWt": "molecular weight (g/mol)",
    "TPSA": "polar surface area: how much of the surface is polar (N, O atoms)",
    "NumHDonors": "hydrogen-bond donors (O-H, N-H)",
    "NumHAcceptors": "hydrogen-bond acceptors (N, O lone pairs)",
    "NumRotatableBonds": "rotatable bonds: how floppy the molecule is",
    "AromaticProportion": "fraction of heavy atoms in aromatic rings",
    "FractionCSP3": "fraction of carbons that are saturated (sp3)",
    "RingCount": "number of rings",
    "NumAromaticRings": "number of aromatic rings",
    "HeavyAtomCount": "number of non-hydrogen atoms (size)",
    "NumHeteroatoms": "number of atoms that are not C or H",
    "MolMR": "molar refractivity (size + polarisability)",
    "NumValenceElectrons": "number of valence electrons",
}
DESCRIPTOR_NAMES: list[str] = list(DESCRIPTORS)

#: Short labels used in charts.
PRETTY: dict[str, str] = {
    "MolLogP": "logP (oiliness)",
    "MolWt": "Molecular weight",
    "TPSA": "Polar surface area",
    "NumHDonors": "H-bond donors",
    "NumHAcceptors": "H-bond acceptors",
    "NumRotatableBonds": "Rotatable bonds",
    "AromaticProportion": "Aromatic proportion",
    "FractionCSP3": "Fraction sp3 carbons",
    "RingCount": "Rings",
    "NumAromaticRings": "Aromatic rings",
    "HeavyAtomCount": "Heavy atoms",
    "NumHeteroatoms": "Heteroatoms",
    "MolMR": "Molar refractivity",
    "NumValenceElectrons": "Valence electrons",
}


#: Every other RDKit 2D descriptor, in RDKit's own order (the list depends on the RDKit
#: version, so a trained model stores the names it used and asks for exactly those).
_RDKIT_FUNCTIONS: dict[str, object] = dict(Descriptors._descList)
EXTRA_DESCRIPTOR_NAMES: list[str] = [n for n in _RDKIT_FUNCTIONS if n not in DESCRIPTORS]

#: Families used to summarise the extended descriptors in explanations.
FAMILIES: dict[str, str] = {
    "surface": "Surface area by polarity",
    "groups": "Functional-group counts",
    "shape": "Shape and branching",
    "charge": "Charges and electron states",
    "size": "Other size measures",
    "other": "Other descriptors",
}
_SHAPE = {"BalabanJ", "BertzCT", "Ipc", "AvgIpc", "HallKierAlpha", "Phi", "SPS"}
_SIZE = {"ExactMolWt", "HeavyAtomMolWt", "LabuteASA"}


def family(name: str) -> str:
    """Which family an extended descriptor belongs to (a key of ``FAMILIES``)."""
    if "VSA" in name:
        return "surface"
    if name.startswith("fr_") or name in {"NHOHCount", "NOCount"} or (name.startswith("Num") and name not in _SIZE):
        return "groups"
    if name.startswith(("Chi", "Kappa")) or name in _SHAPE:
        return "shape"
    if "PartialCharge" in name or "EState" in name or name.startswith("BCUT2D"):
        return "charge"
    if name in _SIZE:
        return "size"
    return "other"


def aromatic_proportion(mol: Chem.Mol) -> float:
    """Aromatic atoms divided by heavy atoms (one of Delaney's four ESOL variables)."""
    heavy = mol.GetNumHeavyAtoms()
    return sum(a.GetIsAromatic() for a in mol.GetAtoms()) / heavy if heavy else 0.0


def descriptors(mol: Chem.Mol) -> np.ndarray:
    """The 14 named descriptors of one molecule, in ``DESCRIPTOR_NAMES`` order."""
    values = {
        "MolLogP": Crippen.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "AromaticProportion": aromatic_proportion(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "MolMR": Crippen.MolMR(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
    }
    return np.array([values[n] for n in DESCRIPTOR_NAMES], dtype=float)


def extended_descriptors(mol: Chem.Mol, names: list[str] | None = None) -> np.ndarray:
    """The extended RDKit descriptors of one molecule (``EXTRA_DESCRIPTOR_NAMES`` order by default).

    Values RDKit cannot compute become NaN (LightGBM treats NaN as "missing"), and the
    few descriptors that can explode for big molecules (e.g. ``Ipc``) are clipped.
    """
    out = np.empty(len(names or EXTRA_DESCRIPTOR_NAMES))
    for i, n in enumerate(names or EXTRA_DESCRIPTOR_NAMES):
        try:
            out[i] = _RDKIT_FUNCTIONS[n](mol)
        except Exception:  # noqa: BLE001 (one failing descriptor must not break a prediction)
            out[i] = np.nan
    out[~np.isfinite(out)] = np.nan
    return np.clip(out, -1e6, 1e6)


@dataclass
class Featurizer:
    """Builds the feature vector for a molecule.

    ``kind`` is ``"desc"`` (descriptors only), ``"fp"`` (fingerprint only),
    ``"desc+fp"`` (both, descriptors first) or ``"rdkit+fp"`` (the 14 descriptors,
    then the extended RDKit descriptors, then the fingerprint; used by model v2).
    """

    kind: str = "desc+fp"
    radius: int = 2
    n_bits: int = 1024
    extra_names: list[str] | None = None   # extended descriptors to compute (default: all of them)

    def __post_init__(self) -> None:
        if self.kind not in {"desc", "fp", "desc+fp", "rdkit+fp"}:
            raise ValueError(f"unknown feature kind {self.kind!r}")
        if self.kind == "rdkit+fp" and self.extra_names is None:
            self.extra_names = EXTRA_DESCRIPTOR_NAMES.copy()
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)

    # -- fingerprints -----------------------------------------------------------
    def fingerprint(self, mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
        """RDKit bit-vector Morgan fingerprint (used for similarity searches)."""
        return self._gen.GetFingerprint(mol)

    def fingerprint_with_info(self, mol: Chem.Mol) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
        """Fingerprint as a 0/1 array plus ``bit -> [(center_atom, radius), ...]``."""
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        fp = self._gen.GetFingerprintAsNumPy(mol, additionalOutput=ao).astype(float)
        return fp, dict(ao.GetBitInfoMap())

    # -- feature vectors --------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """Column names of the feature vector."""
        fp_names = [f"FP_{i}" for i in range(self.n_bits)]
        if self.kind == "desc":
            return DESCRIPTOR_NAMES.copy()
        if self.kind == "fp":
            return fp_names
        if self.kind == "rdkit+fp":
            return DESCRIPTOR_NAMES + list(self.extra_names) + fp_names
        return DESCRIPTOR_NAMES + fp_names

    def transform_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Feature vector of one RDKit molecule."""
        parts = []
        if "desc" in self.kind or self.kind == "rdkit+fp":
            parts.append(descriptors(mol))
        if self.kind == "rdkit+fp":
            parts.append(extended_descriptors(mol, self.extra_names))
        if "fp" in self.kind:
            parts.append(self._gen.GetFingerprintAsNumPy(mol).astype(float))
        return np.concatenate(parts)

    def transform(self, smiles: list[str] | pd.Series) -> np.ndarray:
        """Feature matrix (one row per SMILES). Raises ``ValueError`` on bad SMILES."""
        rows = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"invalid SMILES: {s!r}")
            rows.append(self.transform_mol(mol))
        return np.vstack(rows)
