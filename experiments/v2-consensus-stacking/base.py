"""Train base models with 5-fold out-of-fold predictions on the training part + test predictions. Saves to preds_{kind}.npz"""
from common import *
import lightgbm as lgb, time, os
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from soluai.splits import cv_splitter
kind=sys.argv[1]; which=sys.argv[2].split(',')
tr,te=splits(kind)
folds=cv_splitter(kind,5,42,list(c.smiles.values[tr]))
DF=np.hstack([X['full'],X['mcount'].astype(np.float32),X['maccs'].astype(np.float32)])
full=X['full']; keep=full[tr].std(0)>0
def tanimoto(A,B):  # min-max kernel on count vectors
    out=np.empty((len(A),len(B)),dtype=np.float32)
    for i in range(0,len(A),256):
        a=A[i:i+256,None,:].astype(np.int16); 
        mn=np.minimum(a,B[None].astype(np.int16)).sum(-1); out[i:i+256]=mn/(A[i:i+256].sum(1)[:,None]+B.sum(1)[None]-mn+1e-9)
    return out
def run(name,fitpred):
    t=time.time(); oof=np.zeros(len(tr))
    for a,b in folds: oof[b]=fitpred(tr[a],tr[b])
    pt=fitpred(tr,te); np.savez(f'p_{kind}_{name}.npz',oof=oof,test=pt)
    show(f'{name} (oof rmse {np.sqrt(((oof-y[tr])**2).mean()):.3f})',te,pt); print('  time',round(time.time()-t))
def lgbm(a,b,obj='regression',seed=42):
    return lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.03,num_leaves=31,subsample=0.9,subsample_freq=1,colsample_bytree=0.3,min_child_samples=10,objective=obj,verbose=-1,n_jobs=2,random_state=seed).fit(DF[a],y[a]).predict(DF[b])
def cat(a,b):
    return CatBoostRegressor(iterations=3000,learning_rate=0.05,depth=6,loss_function='RMSE',verbose=0,thread_count=2,random_seed=42).fit(DF[a],y[a]).predict(DF[b])
def et(a,b):
    return ExtraTreesRegressor(500,max_features=0.33,min_samples_leaf=1,n_jobs=2,random_state=42).fit(full[a][:,keep],y[a]).predict(full[b][:,keep])
KM=np.load('K.npy')
def svr(a,b,C=5):
    return SVR(kernel='precomputed',C=C,epsilon=0.1).fit(KM[np.ix_(a,a)],y[a]).predict(KM[np.ix_(b,a)])
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
def mlp(a,b):
    q=QuantileTransformer(output_distribution='normal',n_quantiles=200,random_state=0).fit(full[a][:,keep])
    A=np.hstack([q.transform(full[a][:,keep]),X['maccs'][a]]); B=np.hstack([q.transform(full[b][:,keep]),X['maccs'][b]])
    ps=[MLPRegressor(hidden_layer_sizes=(512,256),alpha=1e-2,learning_rate_init=1e-3,max_iter=200,early_stopping=True,n_iter_no_change=15,random_state=s).fit(A,y[a]).predict(B) for s in range(3)]
    return np.mean(ps,0)
M={'lgbm':lgbm,'huber':lambda a,b:lgbm(a,b,'huber'),'cat':cat,'et':et,'svr':svr,'mlp':mlp}
for w in which: run(w,M[w])
