"""Merge every thermodynamic water-solubility measurement we could find into one table."""
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/soluai/src')
from rdkit import Chem
from soluai.data import standardize
N='/home/claude/newdata/'
rows=[]
for f in sorted(glob.glob(N+'AqSolDB/repo/data/dataset-*.csv')):
    d=pd.read_csv(f); tag='AqSolDB-'+f[-5]
    rows += [(tag,s,v,n) for s,v,n in zip(d.SMILES,d.Solubility,d.Name)]
for tag,f in [('AQUA','aqua/aqua_org.csv'),('PHYSPROP','phys/phys_org.csv'),('OCHEM','ochem/ochem_org.csv'),('ESOL','esol/esol_org.csv')]:
    d=pd.read_csv(N+'SolCuration/repo/org/'+f)
    rows += [(tag,str(s).strip(),v,None) for s,v in zip(d.smiles,d.logS)]
raw=pd.DataFrame(rows,columns=['source','smiles_raw','logS','name'])
print('raw measurements',len(raw)); print(raw.source.value_counts().to_dict())
out=[]; rej={}
cache={}
for r in raw.itertuples(index=False):
    if r.smiles_raw not in cache:
        mol,reason=standardize(r.smiles_raw)
        cache[r.smiles_raw]=(Chem.MolToSmiles(mol),Chem.MolToInchiKey(mol)) if mol is not None else reason
    c=cache[r.smiles_raw]
    if isinstance(c,str): rej[c]=rej.get(c,0)+1; continue
    if not np.isfinite(r.logS): continue
    out.append((r.source,c[0],c[1],float(r.logS),r.name))
m=pd.DataFrame(out,columns=['source','smiles','inchikey','logS','name'])
m['key']=m.inchikey.str[:14]          # ignore stereo: labs rarely separate isomers
print('kept',len(m),'rejected',rej)
m.to_csv('measurements.csv',index=False)
def agg(g):
    v=np.sort(g.logS.values)
    # values within 0.02 are the same number copied between databases -> count once
    distinct=[v[0]]
    for x in v[1:]:
        if x-distinct[-1]>0.02: distinct.append(x)
    distinct=np.array(distinct)
    nm=g['name'].dropna()
    return pd.Series({'smiles':g.smiles.iloc[0],'inchikey':g.inchikey.iloc[0],'name':nm.iloc[0] if len(nm) else '',
        'logS':float(np.median(distinct)),'n_meas':len(distinct),'spread':float(distinct.max()-distinct.min()),
        'sd':float(distinct.std(ddof=1)) if len(distinct)>1 else np.nan,'n_sources':g.source.nunique(),
        'sources':'|'.join(sorted(g.source.unique())),'in_aqsoldb':any(s.startswith('AqSolDB') for s in g.source)})
c=m.groupby('key').apply(agg,include_groups=False).reset_index()
c.to_csv('merged.csv',index=False)
print('unique molecules',len(c),'| in AqSolDB',c.in_aqsoldb.sum(),'| new',(~c.in_aqsoldb).sum())
print('n_meas dist',c.n_meas.clip(upper=5).value_counts().sort_index().to_dict())
multi=c[c.n_meas>1]
print('multi-measured',len(multi),'median spread',multi.spread.median().round(2),'mean sd',multi.sd.mean().round(2),'spread>1:',(multi.spread>1).mean().round(3))
print('logS std',c.logS.std().round(2))
