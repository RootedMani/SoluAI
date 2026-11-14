"""Step 1: compute and cache candidate feature blocks (RDKit descriptors, count fingerprints, MACCS)
and the reliability information (how often each AqSolDB value was measured).

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import CACHE
import numpy as np, pandas as pd, warnings, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from soluai.data import RAW, load_clean, standardize
from soluai.features import Featurizer
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
for name in ["aqsoldb","esol"]:
    df = load_clean(name)
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    f = Featurizer("desc+fp")
    X0 = f.transform(df.smiles)
    names = [d[0] for d in Descriptors._descList]
    R = np.array([[fn(m) for _, fn in Descriptors._descList] for m in mols], dtype=float)
    R[~np.isfinite(R)] = np.nan
    R = np.clip(R, -1e6, 1e6)
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    C = np.array([g.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    M = np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols], dtype=np.float32)
    np.savez_compressed(str(CACHE / f"{name}.npz"), X0=X0, R=R, C=C, M=M, rnames=np.array(names), y=df.logS.values)
    print(name, X0.shape, R.shape, C.shape, M.shape)
# reliability info for aqsoldb
raw = pd.read_csv(RAW / "aqsoldb_curated.csv")
rows=[]
for r in raw.itertuples():
    mol, reason = standardize(r.SMILES)
    if mol is None: continue
    rows.append((Chem.MolToInchiKey(mol), r.SD, r.Occurrences, r.Group))
rel = pd.DataFrame(rows, columns=["inchikey","sd","occ","group"]).groupby("inchikey").agg(sd=("sd","max"),occ=("occ","sum"),group=("group","first")).reset_index()
df = load_clean("aqsoldb").merge(rel, on="inchikey", how="left", suffixes=("","_r"))
df[["inchikey","sd","occ","group_r"]].to_csv(str(CACHE / "rel.csv"), index=False)
print(df.group_r.value_counts(), df.sd.describe(), df.occ.describe())
