"""Question: is the error the same for all nine original data sources inside AqSolDB?

AqSolDB IDs start with a letter A-I that names the dataset each value came from.
Exploratory code: see research/README.md. Run from the research/ folder, after 01 and 02.
"""
from common import *
import pickle
from rdkit import Chem
from soluai.data import RAW, standardize

raw = pd.read_csv(RAW / "aqsoldb_curated.csv")
rows = []
for r in raw.itertuples():
    mol, reason = standardize(r.SMILES)
    if mol is not None:
        rows.append((Chem.MolToInchiKey(mol), r.ID.split("-")[0]))
src = pd.DataFrame(rows, columns=["inchikey", "src"]).groupby("inchikey").src.first().reset_index()
df, d = load("aqsoldb"); y = d["y"]
df = df.merge(src, on="inchikey", how="left")
rel = pd.read_csv(str(CACHE / "rel.csv")); e1 = pickle.load(open(str(CACHE / "e1.pkl"), "rb"))
print(pd.crosstab(df.src, rel.group_r))
for sp in ["random", "scaffold"]:
    tr, te, _ = split(df, sp); oof, p = e1[(sp, "rdkit217+fp1024")]; s = df.src.values[te]
    for k in sorted(set(s)):
        m = s == k
        print(sp, k, m.sum(), "rmse %.3f r2 %.3f within1 %.3f bias %.2f" % (
            rmse(p[m], y[te][m]), r2(y[te][m], p[m]), np.mean(np.abs(p[m] - y[te][m]) <= 1), np.mean(p[m] - y[te][m])))
