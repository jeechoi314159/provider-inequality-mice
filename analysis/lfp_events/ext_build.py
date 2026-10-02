# Extended windows from the reference source (Powerspect_for_foraging.mat): band traces aligned to the cagemate's (starter's) entry
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, pickle
P=(LFP + 'powerspect_extract/')
REG=['PFC','NAc','BLA']; BANDS=['ltheta','htheta','lbeta','hbeta','lgamma','g51_100','hgamma']
FS=8; T=np.arange(-8,3.0001,1/FS)          # s from starter entry
M=pd.read_csv('ps_entries_mapped.csv'); beh=pd.read_csv((INTER + 'follower_retrieval_latency.csv')); A=beh[beh.Group=='A']
TMAX=A.Trial.max(); BAD={64,98,118}
S={(st,r):np.load(P+f'ps_{st}_{r}.npz') for st in ['Start_act','Act_ride'] for r in ['worker','participant','freerider']}
wk=M[M.ps_role=='worker'].set_index('trial'); st=A[A.TrialRole=='Starter'].set_index('Trial')
rows=[]; X={(g,b):[] for g in REG for b in BANDS}
for _,e in M.iterrows():
    t=int(e.trial)
    if t in BAD or t not in st.index or e.Mouse_ID==st.loc[t,'Mouse_ID'] or e.Mouse_ID=='A2' and False: continue
    SL=st.loc[t,'EntryLatency']; follow=int(e.ps_role!='freerider')
    if follow:
        if abs(e.len_s-e.EntryLatency)>0.5: continue
        L=e.EntryLatency-SL; tmax=L-1.0        # exclude the last 1 s before own entry
    else:
        WL=wk.loc[t,'EntryLatency']
        if abs(e.len_s-WL)>0.5: continue
        L=np.nan; tmax=(WL-SL)-1.0 if WL-SL>=1.0 else 3.0   # stage ends at worker entry; if worker = starter, continue into Act_ride
    idx=np.round((SL+T)*FS).astype(int); valid=(idx>=0)&(T<=tmax)
    for g in REG:
        for b in BANDS:
            a=S[('Start_act',e.ps_role)][f'{g}_{b}'][int(e.ps_j)]; a=a[~np.isnan(a)]
            if not follow and WL-SL<1.0:
                c=S[('Act_ride','freerider')][f'{g}_{b}'][int(e.ps_j)]; a=np.concatenate([a,c[~np.isnan(c)]])
            v=np.full(len(T),np.nan,np.float32); ok=valid&(idx<len(a)); v[ok]=a[idx[ok]]; X[(g,b)].append(v)
    rows.append(dict(Mouse_ID=e.Mouse_ID,trial=t,day=e.day,follow=follow,L=L,logL=np.log10(L) if follow else np.nan,StarterLat=SL,
                     pos=(t-1)/(TMAX-1),role=e.ps_role,starter=st.loc[t,'Mouse_ID']))
meta=pd.DataFrame(rows); X={k:np.stack(v) for k,v in X.items()}
pickle.dump(dict(meta=meta,X=X,T=T),open('ext_data.pkl','wb'))
print(meta.groupby(['Mouse_ID','follow']).size().unstack()); print('bins',len(T))
cov=np.isfinite(X[('PFC','g51_100')]).mean(0); print('coverage at t=-8,-4,-2,0,+2,+3:',[round(cov[np.argmin(abs(T-x))],2) for x in [-8,-4,-2,0,2,3]])
