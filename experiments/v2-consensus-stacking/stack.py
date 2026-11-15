from common import *
from scipy.optimize import nnls
kind=sys.argv[1]; names=sys.argv[2].split(',')
tr,te=splits(kind)
P={n:np.load(f'p_{kind}_{n}.npz') for n in names}
O=np.column_stack([P[n]['oof'] for n in names]); T=np.column_stack([P[n]['test'] for n in names])
w,_=nnls(O,y[tr]); print('weights',dict(zip(names,w.round(3))),'sum',w.sum().round(3))
show('simple mean',te,T.mean(1)); pb=T@w; ob=O@w; show('NNLS stack',te,pb)
# analog correction: add similarity-weighted mean OOF residual of nearest training neighbours
K=np.load('K.npy'); res=y[tr]-ob
Ktt=K[np.ix_(tr,tr)].copy(); np.fill_diagonal(Ktt,0)
def corr(Km,k,tau,lam):
    idx=np.argpartition(-Km,k,axis=1)[:,:k]; s=np.take_along_axis(Km,idx,1); s=np.where(s>=tau,s,0)**2
    return (s*res[idx]).sum(1)/(s.sum(1)+lam)
best=None
for k in (3,5,10):
    for tau in (0.3,0.5,0.7):
        for lam in (0.2,0.5,1.0):
            r=np.sqrt(((ob+corr(Ktt,k,tau,lam)-y[tr])**2).mean())
            if best is None or r<best[0]: best=(r,k,tau,lam)
print('oof rmse stack',np.sqrt(((ob-y[tr])**2).mean()).round(4),'with analog corr',best)
pc=pb+corr(K[np.ix_(te,tr)],*best[1:]); show('stack + analog correction',te,pc)
np.savez(f'final_{kind}.npz',pred=pc,stack=pb,te=te)
