"""Honest held-out scores for every selectable model + 90% error widths from cross-validation (training part only)."""
from common import *
from scipy.optimize import nnls
import json
sys.path.insert(0,'/home/claude/soluai/src')
from rdkit import Chem
from soluai.baselines import delaney_predict
import lightgbm as lgb
members=['lgbm','huber','cat','svr','mlp']
out={'members':members,'splits':{}}
dela=delaney_predict(list(c.smiles))
for kind in ('random','scaffold'):
    tr,te=splits(kind); P={n:np.load(f'p_{kind}_{n}.npz') for n in members}
    O=np.column_stack([P[n]['oof'] for n in members]); T=np.column_stack([P[n]['test'] for n in members])
    w,_=nnls(O,y[tr]); 
    res={'n_train':len(tr),'n_test':len(te),'weights':dict(zip(members,w.tolist()))}
    preds={n:(P[n]['oof'],P[n]['test']) for n in members}; preds['consensus']=(O@w,T@w)
    # v1.0 recipe (14 descriptors + 1024-bit fingerprint, same LightGBM settings), same split
    Xa=np.hstack([X['d14'],X['fp1024']]).astype(np.float32)
    from soluai.splits import cv_splitter
    folds=cv_splitter(kind,5,42,list(c.smiles.values[tr]))
    mk=lambda: lgb.LGBMRegressor(n_estimators=400,learning_rate=0.02,num_leaves=63,subsample=0.7,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.0,min_child_samples=10,verbose=-1,n_jobs=2,random_state=42)
    oof=np.zeros(len(tr))
    for a,b in folds: oof[b]=mk().fit(Xa[tr[a]],y[tr[a]]).predict(Xa[tr[b]])
    preds['v1']=(oof,mk().fit(Xa[tr],y[tr]).predict(Xa[te]))
    preds['delaney']=(dela[tr],dela[te])
    res['models']={}
    for n,(o,t) in preds.items():
        m=metrics(y[te],t); m['q90_cv']=float(np.quantile(np.abs(o-y[tr]),0.9))
        res['models'][n]={k:round(v,4) if isinstance(v,float) else v for k,v in m.items()}
    out['splits'][kind]=res
    print(kind,{n:(v['r2'],v['rmse']) for n,v in res['models'].items()})
json.dump(out,open('zoo_metrics.json','w'),indent=1)
