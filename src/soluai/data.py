"""Loading and cleaning the solubility datasets.

Two public datasets are used:

* **AqSolDB** (Sorkun et al., 2019) - about 10,000 compounds merged from 9 sources.
* **ESOL** (Delaney, 2004) - about 1,100 compounds, the classic benchmark.

Cleaning rules (applied in this order, each one counted in a report):

1. the SMILES must be readable by RDKit;
2. only one fragment: salts and mixtures ("A.B") are removed, because the
   solubility of a salt is not the solubility of its organic part;
3. organic only: must contain carbon and only "organic" elements;
4. charges are neutralised where possible and the SMILES is canonicalised
   (written in one standard way, so the same molecule always looks the same);
5. duplicates (same InChIKey) are averaged if they agree within 0.5 log units,
   otherwise dropped because we cannot know which measurement is right.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from . import DATA_DIR

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"

DEFAULT_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}


def download_if_missing(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` unless the file is already there (offline-friendly)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, path)  # noqa: S310 (trusted, fixed URL)
    return path


def load_raw_aqsoldb() -> pd.DataFrame:
    """Raw AqSolDB as a table with columns name, smiles, logS, group."""
    df = pd.read_csv(RAW / "aqsoldb_curated.csv")
    return pd.DataFrame(
        {"name": df["Name"], "smiles": df["SMILES"], "logS": df["Solubility"], "group": df["Group"]}
    )


def load_raw_esol() -> pd.DataFrame:
    """Raw ESOL (Delaney) as a table with columns name, smiles, logS."""
    df = pd.read_csv(RAW / "esol_delaney.csv")
    return pd.DataFrame(
        {
            "name": df["Compound ID"],
            "smiles": df["smiles"].str.strip(),
            "logS": df["measured log solubility in mols per litre"],
            "group": "ESOL",
        }
    )


_uncharger = rdMolStandardize.Uncharger()


def standardize(smiles: str, allowed: set[str] = DEFAULT_ELEMENTS) -> tuple[Chem.Mol | None, str]:
    """Clean one SMILES string.

    Returns ``(mol, reason)``. ``mol`` is ``None`` when the molecule is rejected,
    and ``reason`` then says which rule rejected it.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None, "invalid_smiles"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "salt_or_mixture"
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in symbols:
        return None, "no_carbon"
    if not symbols <= allowed:
        return None, "non_organic_element"
    mol = _uncharger.uncharge(mol)
    return mol, "ok"


def clean(df: pd.DataFrame, allowed: set[str] = DEFAULT_ELEMENTS, max_spread: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Apply all cleaning rules and return ``(clean_df, report)``."""
    report: dict[str, int] = {"input_rows": len(df)}
    rows = []
    for rec in df.itertuples(index=False):
        mol, reason = standardize(rec.smiles, allowed)
        report[reason] = report.get(reason, 0) + 1
        if mol is None or not np.isfinite(rec.logS):
            continue
        rows.append(
            {
                "name": rec.name,
                "smiles": Chem.MolToSmiles(mol),
                "inchikey": Chem.MolToInchiKey(mol),
                "logS": float(rec.logS),
                "group": rec.group,
            }
        )
    out = pd.DataFrame(rows)

    # Duplicates: average if they agree, drop if they disagree.
    agg = out.groupby("inchikey").agg(
        name=("name", "first"),
        smiles=("smiles", "first"),
        group=("group", "first"),
        logS=("logS", "mean"),
        spread=("logS", lambda s: s.max() - s.min()),
        n=("logS", "size"),
    )
    report["duplicate_groups"] = int((agg["n"] > 1).sum())
    report["duplicates_conflicting_dropped"] = int((agg["spread"] > max_spread).sum())
    agg = agg[agg["spread"] <= max_spread].reset_index()
    report["output_rows"] = len(agg)
    return agg[["name", "smiles", "inchikey", "logS", "group"]], report


def build_clean_datasets(cfg: dict) -> dict[str, pd.DataFrame]:
    """Download (if needed), clean and save both datasets. Returns them by name."""
    download_if_missing(cfg["data"]["aqsoldb_url"], RAW / "aqsoldb_curated.csv")
    download_if_missing(cfg["data"]["esol_url"], RAW / "esol_delaney.csv")
    allowed = set(cfg["data"]["allowed_elements"])
    spread = cfg["data"]["duplicate_max_spread"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out, reports = {}, {}
    for name, loader in [("aqsoldb", load_raw_aqsoldb), ("esol", load_raw_esol)]:
        df, rep = clean(loader(), allowed, spread)
        df.to_csv(PROCESSED / f"{name}_clean.csv", index=False)
        out[name], reports[name] = df, rep
    (PROCESSED / "cleaning_report.json").write_text(json.dumps(reports, indent=2))
    return out


def load_clean(name: str) -> pd.DataFrame:
    """Load a cleaned dataset (``'aqsoldb'`` or ``'esol'``) from ``data/processed``."""
    return pd.read_csv(PROCESSED / f"{name}_clean.csv")
