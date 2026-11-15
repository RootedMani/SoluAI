from common import *
import lightgbm as lgb, time
kind=sys.argv[1]
tr,te=splits(kind); print(kind,len(tr),len(te),'reliable in test',reliable[te].sum())
def fit(feat,w=None,**kw):
    Xa=np.hstack([X[f].astype(np.float32) for f in feat])
    p=dict(n_estimators=400,learning_rate=0.02,num_leaves=63,subsample=0.9,subsample_freq=1,colsample_bytree=0.3,min_child_samples=10,verbose=-1,n_jobs=2,random_state=42); p.update(kw)
    m=lgb.LGBMRegressor(**p).fit(Xa[tr],y[tr],sample_weight=w); return m.predict(Xa[te])
t=time.time()
show('v1.0 recipe (14desc+fp1024)',te,fit(['d14','fp1024']))
show('full desc',te,fit(['full'],n_estimators=1500,learning_rate=0.03,num_leaves=31))
p=fit(['full','mcount','maccs'],n_estimators=1500,learning_rate=0.03,num_leaves=31); show('full+mcount+maccs',te,p)
w=np.where(c.n_meas.values[tr]>=2, np.where(c.spread.values[tr]<=0.5,2.0,0.5),1.0)
show(' + noise weights',te,fit(['full','mcount','maccs'],w=w,n_estimators=1500,learning_rate=0.03,num_leaves=31))
print(time.time()-t)
