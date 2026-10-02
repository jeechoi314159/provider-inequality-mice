# Reference pipeline loader: ed_spectro CSVs (2026-09-01; sliced from Powerspect_for_foraging.mat), cohort A
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, os, pickle
from scipy.ndimage import convolve1d
D=(INTER + 'extended_data/')
REG=['PFC','NAc','BLA']; FR=np.arange(1,101)
def gauss(s):
    h=int(np.ceil(3*s)); x=np.arange(-h,h+1); k=np.exp(-x**2/(2*s**2)); return k/k.sum()
KT,KF=gauss(1.5),gauss(2.0)
def load(prefix, cache):
    if os.path.exists(cache): return pickle.load(open(cache,'rb'))
    meta=pd.read_csv(D+prefix+'trials_meta.csv'); X={}
    for r in REG:
        s=pd.read_csv(D+f'{prefix}{r}.csv'); t=np.sort(s.t.unique())
        s=s.sort_values(['idx','t']); arr=s[[f'f{k}' for k in FR]].values.reshape(len(meta),len(t),100)
        assert (s.idx.values.reshape(len(meta),len(t))[:,0]==meta.idx.values).all()
        # same smoothing as the reference script (Gaussian, sigma 1.5 bins x 2 Hz, 'same')
        arr=convolve1d(convolve1d(arr,KT,axis=1,mode='constant'),KF,axis=2,mode='constant')
        X[r]=arr.astype(np.float32)
    out=(meta,t,X); pickle.dump(out,open(cache,'wb')); return out
beh=pd.read_csv((INTER + 'follower_retrieval_latency.csv')); TMAX=beh[beh.Group=='A'].Trial.max()
def prep(meta):
    m=meta.copy(); m['logL']=np.log10(m.FollowerLatency); m['pos']=(m.Trial-1)/(TMAX-1); m['fP']=(m.Case=='NP follows P').astype(float); return m
if __name__=='__main__':
    for p,c in [('ed_spectro_','cag.pkl'),('ed_spectro_own_','own.pkl')]:
        meta,t,X=load(p,c); print(p,len(meta),t[0],t[-1],len(t),{r:X[r].shape for r in X})
