# -*- coding: utf-8 -*-
"""운동량 공변량 통제. two_signals.py 의 결합 모형을 재현한 뒤 시행별 운동에너지를 넣음."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd
KE='/'.join((FIGDATA[:-1]).split('/'))+'/figS02h_kinetic_energy.csv'
rng=np.random.default_rng(20260922); NPERM=10000
NP=['A1','A3','A4','A5','A6']
M=pd.read_csv('two_signals_trials.csv')
K=pd.read_csv(KE).rename(columns={'mouse':'Mouse_ID','ke_foraging':'ke'})
print('ke rows %d, state별 중앙값:'%len(K)); print(K.groupby('state').ke.describe()[['count','50%']])
M=M.merge(K[['Mouse_ID','trial','state','ke']],on=['Mouse_ID','trial'],how='left')
print('\n결합 실패 %d / %d'%(M.ke.isna().sum(),len(M)))
M['logke']=np.log10(M.ke.clip(lower=1e-8))
M['logL']=np.log10(M.L.where(M.L>0))

def resid(Mx,g,pos):
    out=np.empty_like(Mx,dtype=float)
    for u in np.unique(g):
        i=np.where(g==u)[0]; X=np.column_stack([np.ones(len(i)),pos[i]])
        B=np.linalg.lstsq(X,Mx[i],rcond=None)[0]; out[i]=Mx[i]-X@B
    return out

def fit(S,ycol,cols,label,nperm=NPERM):
    g=S.Mouse_ID.values; pos=S.pos.values
    Z=resid(S[cols].values.astype(float),g,pos); Z=Z/Z.std(0)
    y=resid(S[[ycol]].values.astype(float),g,pos)[:,0]
    B=np.linalg.lstsq(Z,y,rcond=None)[0]
    idx=[np.where(g==u)[0] for u in np.unique(g)]
    null=np.empty((nperm,len(cols)))
    for k in range(nperm):
        yp=y.copy()
        for i in idx: yp[i]=y[rng.permutation(i)]
        null[k]=np.linalg.lstsq(Z,yp,rcond=None)[0]
    P=(np.abs(null)>=np.abs(B)).mean(0)
    e=y-Z@B; R2=1-e@e/(y@y)
    print('  %-34s n=%-4d R2=%.4f  %s'%(label,len(S),R2,
          '  '.join('%s %+.4f (P=%.4f)'%(c,b,p) for c,b,p in zip(cols,B,P))))
    return dict(label=label,n=len(S),R2=R2,**{('b_'+c):b for c,b in zip(cols,B)},**{('P_'+c):p for c,p in zip(cols,P)})

rows=[]
for tag,sel,ycol in [('Wait (followers, L>=3 s)', lambda d: d[d.Mouse_ID.isin(NP)&(d.follow==1)&(d.L>=3)], 'logL'),
                     ('Follow vs stay out',        lambda d: d[d.Mouse_ID.isin(NP)],                      'follow')]:
    base=sel(M).dropna(subset=['S_pre','S_post']).reset_index(drop=True)
    both=sel(M).dropna(subset=['S_pre','S_post','logke']).reset_index(drop=True)
    print('\n===== %s'%tag)
    rows.append(fit(base,ycol,['S_pre','S_post'],'재현 (원 모형)'))
    if len(both)!=len(base): rows.append(fit(both,ycol,['S_pre','S_post'],'운동량 있는 시행만'))
    rows.append(fit(both,ycol,['S_pre','S_post','logke'],'+ 운동에너지 통제'))
    rows.append(fit(both,ycol,['logke'],'운동에너지 단독'))
    g=both.Mouse_ID.values; pos=both.pos.values
    Z=resid(both[['S_pre','S_post','logke']].values.astype(float),g,pos); Z=Z/Z.std(0)
    print('  개체 내 상관: r(S_pre,ke) %+.3f   r(S_post,ke) %+.3f   r(S_pre,S_post) %+.3f'%(
        np.corrcoef(Z[:,0],Z[:,2])[0,1],np.corrcoef(Z[:,1],Z[:,2])[0,1],np.corrcoef(Z[:,0],Z[:,1])[0,1]))
    per={}
    for u in np.unique(g):
        i=g==u; b=np.linalg.lstsq(Z[i],resid(both[[ycol]].values.astype(float),g,pos)[i,0],rcond=None)[0]
        per[u]=[round(float(x),3) for x in b]
    print('  개체별 [S_pre, S_post, ke]:',per)
pd.DataFrame(rows).to_csv('ke_control_summary.csv',index=False)
print('\nwrote ke_control_summary.csv')
