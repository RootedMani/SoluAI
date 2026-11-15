import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/soluai/src')
from soluai.splits import random_split, scaffold_split
from sklearn.metrics import r2_score
c=pd.read_csv('merged.csv'); X=dict(np.load('X.npz')); y=c.logS.values
reliable=((c.n_meas>=2)&(c.spread<=0.5)).values
def splits(kind):
    return random_split(len(c),0.2,42) if kind=='random' else scaffold_split(list(c.smiles),0.2,42)
def metrics(yt,p):
    e=p-yt
    return dict(rmse=float(np.sqrt((e**2).mean())),mae=float(np.abs(e).mean()),r2=float(r2_score(yt,p)),w1=float((np.abs(e)<=1).mean()),w05=float((np.abs(e)<=0.5).mean()),n=len(yt))
def show(tag,te,p):
    a=metrics(y[te],p); r=reliable[te]; b=metrics(y[te][r],p[r])
    print(f"{tag:34s} ALL n={a['n']} RMSE {a['rmse']:.3f} MAE {a['mae']:.3f} R2 {a['r2']:.3f} ±1:{a['w1']:.3f} | RELIABLE n={b['n']} RMSE {b['rmse']:.3f} R2 {b['r2']:.3f} ±1:{b['w1']:.3f}",flush=True)
    return a,b
