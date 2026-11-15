from common import *
import lightgbm as lgb
m=pd.read_csv('measurements.csv')
# lab-vs-lab: molecules with >=2 distinct measurements: pick two different values at random
rng=np.random.default_rng(0); a=[];b=[]
for k,g in m.groupby('key'):
    v=np.unique(g.logS.round(2).values)
    if len(v)>=2 and v.max()-v.min()>0.02:
        i,j=rng.choice(len(v),2,replace=False); a.append(v[i]); b.append(v[j])
a=np.array(a);b=np.array(b); print('LAB vs LAB on',len(a),'molecules:',metrics(a,b))
kind='random'; names=['lgbm','cat','et','svr']; tr,te=splits(kind)
from scipy.optimize import nnls
P={n:np.load(f'p_{kind}_{n}.npz') for n in names}
O=np.column_stack([P[n]['oof'] for n in names]); T=np.column_stack([P[n]['test'] for n in names]); w,_=nnls(O,y[tr]); ob=O@w; pb=T@w
K=np.load('K.npy'); Ktt=K[np.ix_(tr,tr)].copy(); np.fill_diagonal(Ktt,0); Kte=K[np.ix_(te,tr)]
def cf(Om,Km,idx): 
    top=-np.sort(-Km,axis=1)[:,:5]
    return np.column_stack([Om.std(1),top[:,0],top.mean(1),X['full'][idx][:,[NAMES.index('HeavyAtomCount')]] ])
from feats import NAMES
Ftr=cf(O,Ktt,tr); Fte=cf(T,Kte,te)
em=lgb.LGBMRegressor(n_estimators=200,learning_rate=0.03,num_leaves=8,min_child_samples=50,verbose=-1).fit(Ftr,np.abs(ob-y[tr]))
score=em.predict(Fte); order=np.argsort(score)
for cov in (1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3):
    s=order[:int(cov*len(te))]; a_=metrics(y[te][s],pb[s]); r=reliable[te][s]; b_=metrics(y[te][s][r],pb[s][r])
    print(f"coverage {cov:.0%}: n={a_['n']} RMSE {a_['rmse']:.3f} R2 {a_['r2']:.3f} ±1 {a_['w1']:.3f} | reliable n={b_['n']} R2 {b_['r2']:.3f} ±1 {b_['w1']:.3f}")
# classification views
yt=y[te]
for thr in (-2,-4):
    print('binary at',thr,'acc',((yt>thr)==(pb>thr)).mean().round(3))
bins=[-99,-4,-2,0,99]; print('4-class acc',(np.digitize(yt,bins)==np.digitize(pb,bins)).mean().round(3), 'within one class',(np.abs(np.digitize(yt,bins)-np.digitize(pb,bins))<=1).mean().round(3))
