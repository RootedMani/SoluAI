import numpy as np, time
X=dict(np.load('X.npz'))
def tan(A):
    n=len(A); s=A.sum(1).astype(np.float32); out=np.empty((n,n),dtype=np.float32)
    for i in range(0,n,24):
        mn=np.minimum(A[i:i+24,None,:],A[None]).sum(-1,dtype=np.int32); out[i:i+24]=mn/(s[i:i+24,None]+s[None]-mn+1e-9)
    return out
t=time.time(); K=0.5*tan(X['mcount'])+0.5*tan(X['ap']); np.save('K.npy',K); print(K.shape,time.time()-t)
