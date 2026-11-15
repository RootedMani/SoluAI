import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/soluai/src')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator as G
from soluai.features import descriptors as d14
RDLogger.DisableLog('rdApp.*')
NAMES=[n for n,_ in Descriptors._descList]
mg=G.GetMorganGenerator(radius=2,fpSize=2048); mg1=G.GetMorganGenerator(radius=2,fpSize=1024)
ap=G.GetAtomPairGenerator(fpSize=1024); tt=G.GetTopologicalTorsionGenerator(fpSize=1024)
def one(smi):
    m=Chem.MolFromSmiles(smi)
    full=np.array([f(m) for _,f in Descriptors._descList],dtype=float)
    return (d14(m), mg1.GetFingerprintAsNumPy(m), full, mg.GetCountFingerprintAsNumPy(m),
            np.array(MACCSkeys.GenMACCSKeys(m)), ap.GetCountFingerprintAsNumPy(m), tt.GetCountFingerprintAsNumPy(m))
if __name__=='__main__':
    c=pd.read_csv('merged.csv'); parts=[one(s) for s in c.smiles]
    arr=[np.array([p[i] for p in parts]) for i in range(7)]
    full=np.nan_to_num(np.clip(arr[2],-1e6,1e6))
    np.savez_compressed('X.npz',d14=arr[0],fp1024=arr[1].astype(np.uint8),full=full.astype(np.float32),mcount=np.clip(arr[3],0,255).astype(np.uint8),
        maccs=arr[4].astype(np.uint8),ap=np.clip(arr[5],0,255).astype(np.uint8),tt=np.clip(arr[6],0,255).astype(np.uint8))
    print({k:v.shape for k,v in np.load('X.npz').items()})
