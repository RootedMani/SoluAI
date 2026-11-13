"""Turning a molecule into numbers a model can learn from.

Two kinds of features are used:

* **Descriptors** - 14 named, chemically meaningful numbers such as logP
  (how oily the molecule is), molecular weight, or the number of hydrogen-bond
  donors. A chemist can read every one of them.
* **Morgan fingerprint** - 1024 yes/no bits. Each bit says "this small
  fragment (an atom plus its neighbours up to 2 bonds away) is present". It
  lets the model notice specific groups such as -OH or -COOH.
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


@dataclass
class Featurizer:
    """Builds the feature vector for a molecule.

    ``kind`` is ``"desc"`` (descriptors only), ``"fp"`` (fingerprint only)
    or ``"desc+fp"`` (both, descriptors first).
    """

    kind: str = "desc+fp"
    radius: int = 2
    n_bits: int = 1024

    def __post_init__(self) -> None:
        if self.kind not in {"desc", "fp", "desc+fp"}:
            raise ValueError(f"unknown feature kind {self.kind!r}")
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
        return DESCRIPTOR_NAMES + fp_names

    def transform_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Feature vector of one RDKit molecule."""
        parts = []
        if "desc" in self.kind:
            parts.append(descriptors(mol))
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
