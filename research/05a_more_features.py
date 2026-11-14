"""Step 5a: more feature blocks (E-state atom types, Crippen atom types, atom pairs, torsions).

Exploratory code: see research/README.md. Run from the research/ folder, in order.
"""
from common import CACHE
import numpy as np, warnings
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator, Crippen
from rdkit.Chem.EState import Fingerprinter
from soluai.data import load_clean
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
# Crippen atom types: assign via _CalcCrippenContribs? need type ids -> use Crippen._pyGetAtomContribs patterns
from rdkit.Chem import Crippen as C
C._Init() if hasattr(C, "_Init") else None
order, patts = C._ReadPatts(C.defaultPatternFileName)
def crippen_counts(m):
    mh = m
    done = [False]*m.GetNumAtoms(); out = np.zeros(len(order))
    for j, cha in enumerate(order):
        for sma, p, logp, mr in patts[cha]:
            for (a,) in set((x[0],) for x in mh.GetSubstructMatches(p, False, False)):
                if not done[a]: done[a]=True; out[j]+=1
    return out
ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
for name in ["aqsoldb","esol"]:
    df = load_clean(name); mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    E = np.array([np.concatenate(Fingerprinter.FingerprintMol(m)) for m in mols], dtype=np.float32)
    CR = np.array([crippen_counts(m) for m in mols], dtype=np.float32)
    AP = np.array([ap.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    TT = np.array([tt.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    np.savez_compressed(str(CACHE / f"{name}2.npz"), E=E, CR=CR, AP=AP, TT=TT); print(name, E.shape, CR.shape, AP.shape, TT.shape, len(order))
