"""Final evaluation: stack of 6 models + confidence gate. Everything is tuned on training data only."""
from common import *
from scipy.optimize import nnls
import lightgbm as lgb, json
from feats import NAMES
names=['lgbm','huber','cat','et','svr','mlp']; K=np.load('K.npy'); H=NAMES.index('HeavyAtomCount'); out={}
for kind in ('random','scaffold'):
    tr,te=splits(kind); P={n:np.load(f'p_{kind}_{n}.npz') for n in names}
    O=np.column_stack([P[n]['oof'] for n in names]); T=np.column_stack([P[n]['test'] for n in names])
    w,_=nnls(O,y[tr]); ob=O@w; pb=T@w
    Ktt=K[np.ix_(tr,tr)].copy(); np.fill_diagonal(Ktt,0)
    def cf(Om,Km,idx):
        top=-np.sort(-Km,axis=1)[:,:5]; return np.column_stack([Om.std(1),top[:,0],top.mean(1),X['full'][idx][:,H]])
    em=lgb.LGBMRegressor(n_estimators=200,learning_rate=0.03,num_leaves=8,min_child_samples=50,verbose=-1,random_state=0).fit(cf(O,Ktt,tr),np.abs(ob-y[tr]))
    s_tr=em.predict(cf(O,Ktt,tr)); s_te=em.predict(cf(T,K[np.ix_(te,tr)],te))
    thr=np.median(s_tr)                      # "confident" = expected error below the training median
    conf=s_te<=thr; rel=reliable[te]
    print(f'\n=== {kind} split: train {len(tr)}, test {len(te)}; weights',dict(zip(names,w.round(2))))
    res={}
    for tag,mask in [('all test molecules',np.ones(len(te),bool)),('reliable labels only',rel),('confident predictions',conf),('confident + reliable labels',conf&rel)]:
        m=metrics(y[te][mask],pb[mask]); res[tag]=m
        print(f"{tag:28s} n={m['n']:5d} ({mask.mean():.0%})  RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  R2 {m['r2']:.3f}  within±1 {m['w1']:.3f}")
    res['single_models']={n:metrics(y[te],P[n]['test']) for n in names}; res['weights']=dict(zip(names,w.tolist())); out[kind]=res
    pd.DataFrame({'smiles':c.smiles.values[te],'measured_logS':y[te],'predicted_logS':pb.round(3),'expected_error':s_te.round(3),'confident':conf,'reliable_label':rel}).to_csv(f'test_predictions_{kind}.csv',index=False)
json.dump(out,open('results_v2.json','w'),indent=1)
