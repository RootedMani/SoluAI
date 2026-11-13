"""Offline list of common molecules (works without internet).

Each entry: name -> (SMILES, category). Synonyms point to the same molecule.
"""

from __future__ import annotations

COMMON: dict[str, tuple[str, str]] = {
    # Alcohols: a classic trend - longer chain, lower solubility
    "methanol": ("CO", "Alcohols"),
    "ethanol": ("CCO", "Alcohols"),
    "1-propanol": ("CCCO", "Alcohols"),
    "2-propanol": ("CC(C)O", "Alcohols"),
    "1-butanol": ("CCCCO", "Alcohols"),
    "1-pentanol": ("CCCCCO", "Alcohols"),
    "1-hexanol": ("CCCCCCO", "Alcohols"),
    "1-octanol": ("CCCCCCCCO", "Alcohols"),
    "ethylene glycol": ("OCCO", "Alcohols"),
    "glycerol": ("OCC(O)CO", "Alcohols"),
    "menthol": ("CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", "Alcohols"),
    "cholesterol": ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C", "Alcohols"),
    # Sugars and very polar molecules
    "glucose": ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Sugars & polar"),
    "sucrose": ("OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O", "Sugars & polar"),
    "urea": ("NC(N)=O", "Sugars & polar"),
    "glycine": ("NCC(=O)O", "Sugars & polar"),
    "alanine": ("C[C@H](N)C(=O)O", "Sugars & polar"),
    "acetone": ("CC(C)=O", "Sugars & polar"),
    "acetonitrile": ("CC#N", "Sugars & polar"),
    "dimethyl sulfoxide": ("CS(C)=O", "Sugars & polar"),
    # Acids
    "acetic acid": ("CC(=O)O", "Acids"),
    "lactic acid": ("CC(O)C(=O)O", "Acids"),
    "citric acid": ("OC(=O)CC(O)(CC(=O)O)C(=O)O", "Acids"),
    "hexanoic acid": ("CCCCCC(=O)O", "Acids"),
    "benzoic acid": ("OC(=O)c1ccccc1", "Acids"),
    "salicylic acid": ("OC(=O)c1ccccc1O", "Acids"),
    "ascorbic acid": ("OC[C@H](O)[C@H]1OC(=O)C(O)=C1O", "Acids"),
    # Hydrocarbons and halogenated solvents
    "hexane": ("CCCCCC", "Hydrocarbons"),
    "octane": ("CCCCCCCC", "Hydrocarbons"),
    "cyclohexane": ("C1CCCCC1", "Hydrocarbons"),
    "diethyl ether": ("CCOCC", "Hydrocarbons"),
    "ethyl acetate": ("CCOC(C)=O", "Hydrocarbons"),
    "dichloromethane": ("ClCCl", "Hydrocarbons"),
    "chloroform": ("ClC(Cl)Cl", "Hydrocarbons"),
    "carbon tetrachloride": ("ClC(Cl)(Cl)Cl", "Hydrocarbons"),
    # Aromatics
    "benzene": ("c1ccccc1", "Aromatics"),
    "toluene": ("Cc1ccccc1", "Aromatics"),
    "ethylbenzene": ("CCc1ccccc1", "Aromatics"),
    "styrene": ("C=Cc1ccccc1", "Aromatics"),
    "phenol": ("Oc1ccccc1", "Aromatics"),
    "aniline": ("Nc1ccccc1", "Aromatics"),
    "pyridine": ("c1ccncc1", "Aromatics"),
    "benzaldehyde": ("O=Cc1ccccc1", "Aromatics"),
    "chlorobenzene": ("Clc1ccccc1", "Aromatics"),
    "nitrobenzene": ("O=[N+]([O-])c1ccccc1", "Aromatics"),
    "naphthalene": ("c1ccc2ccccc2c1", "Aromatics"),
    "biphenyl": ("c1ccc(-c2ccccc2)cc1", "Aromatics"),
    "anthracene": ("c1ccc2cc3ccccc3cc2c1", "Aromatics"),
    "pyrene": ("c1cc2ccc3cccc4ccc(c1)c2c34", "Aromatics"),
    # Drugs and everyday molecules
    "caffeine": ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Everyday & drugs"),
    "theobromine": ("Cn1cnc2c1c(=O)[nH]c(=O)n2C", "Everyday & drugs"),
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "Everyday & drugs"),
    "paracetamol": ("CC(=O)Nc1ccc(O)cc1", "Everyday & drugs"),
    "ibuprofen": ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Everyday & drugs"),
    "nicotine": ("CN1CCC[C@H]1c1cccnc1", "Everyday & drugs"),
    "vanillin": ("COc1cc(C=O)ccc1O", "Everyday & drugs"),
    "thymol": ("Cc1ccc(C(C)C)c(O)c1", "Everyday & drugs"),
    "camphor": ("CC1(C)C2CCC1(C)C(=O)C2", "Everyday & drugs"),
    "capsaicin": ("COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O", "Everyday & drugs"),
    "DDT": ("ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1", "Everyday & drugs"),
}

SYNONYMS: dict[str, str] = {
    "ethyl alcohol": "ethanol", "alcohol": "ethanol", "methyl alcohol": "methanol",
    "propanol": "1-propanol", "n-propanol": "1-propanol", "isopropanol": "2-propanol",
    "isopropyl alcohol": "2-propanol", "rubbing alcohol": "2-propanol", "butanol": "1-butanol",
    "pentanol": "1-pentanol", "hexanol": "1-hexanol", "octanol": "1-octanol", "n-octanol": "1-octanol",
    "glycerin": "glycerol", "glycerine": "glycerol", "dextrose": "glucose", "sugar": "sucrose",
    "table sugar": "sucrose", "vinegar acid": "acetic acid", "ethanoic acid": "acetic acid",
    "vitamin c": "ascorbic acid", "acetaminophen": "paracetamol", "tylenol": "paracetamol",
    "acetylsalicylic acid": "aspirin", "dmso": "dimethyl sulfoxide", "ether": "diethyl ether",
    "methylbenzene": "toluene", "trichloromethane": "chloroform", "tetrachloromethane": "carbon tetrachloride",
    "methylene chloride": "dichloromethane", "propanone": "acetone", "carbamide": "urea",
}

#: Water is not predicted (it is the solvent), but users often try it.
WATER_NAMES = {"water", "h2o", "o", "dihydrogen monoxide"}


def lookup(name: str) -> tuple[str, str] | None:
    """Return ``(official_name, smiles)`` for a known name or synonym, else ``None``."""
    key = name.strip().lower()
    key = SYNONYMS.get(key, key)
    for official, (smiles, _cat) in COMMON.items():
        if official.lower() == key:
            return official, smiles
    return None


def all_names() -> list[str]:
    """Every name and synonym (for suggestions)."""
    return list(COMMON) + list(SYNONYMS)

#: Common inorganic substances people try; they get a clear "out of scope" message.
INORGANIC_NAMES = {
    "nacl", "sodium chloride", "salt", "table salt", "kcl", "potassium chloride", "caco3",
    "calcium carbonate", "chalk", "nahco3", "baking soda", "sodium bicarbonate", "naoh",
    "sodium hydroxide", "hcl", "hydrochloric acid", "h2so4", "sulfuric acid", "cuso4",
    "copper sulfate", "agcl", "silver chloride", "baso4", "barium sulfate", "nh3", "ammonia",
    "co2", "carbon dioxide", "o2", "oxygen", "n2", "nitrogen", "kno3", "potassium nitrate",
}
