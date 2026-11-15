"""Score PR #2's "chemist" recipe on the same merged-data splits as the model zoo."""
import sys, json, time; sys.path.insert(0,'../../src')
from common import *
import joblib, lightgbm as lgb
from rdkit import Chem
from soluai.splits import cv_splitter
b=joblib.load('../../models/soluai-v2.0.0/model.joblib')
from soluai.features import Featurizer
fz=Featurizer(b["feature_kind"], b["radius"], b["n_bits"], b.get("extra_names"))
t=time.time(); F=np.array([fz.transform_mol(Chem.MolFromSmiles(s)) for s in c.smiles],dtype=np.float32); print('features',F.shape,round(time.time()-t))
params=dict(json.load(open('../../models/soluai-v2.0.0/metadata.json'))['params'])
mk=lambda: lgb.LGBMRegressor(**params,subsample_freq=1,verbose=-1,n_jobs=2,random_state=42)
out={}
for kind in ('random','scaffold'):
    tr,te=splits(kind); folds=cv_splitter(kind,5,42,list(c.smiles.values[tr]))
    oof=np.zeros(len(tr))
    for a,bb in folds: oof[bb]=mk().fit(F[tr[a]],y[tr[a]]).predict(F[tr[bb]])
    p=mk().fit(F[tr],y[tr]).predict(F[te])
    m=metrics(y[te],p); m['q90_cv']=float(np.quantile(np.abs(oof-y[tr]),0.9))
    out[kind]={k:round(v,4) if isinstance(v,float) else v for k,v in m.items()}
    np.savez(f'p_{kind}_chemist.npz',oof=oof,test=p); print(kind,out[kind],flush=True)
json.dump(out,open('chemist_metrics.json','w'),indent=1)
