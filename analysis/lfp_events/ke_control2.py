# -*- coding: utf-8 -*-
"""운동량 통제 2: 시행 기저 구간 운동량(결과의 원인이 될 수 없는 사전 공변량)을 함께 씀."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, openpyxl, os
XL=(BEHAV + 'Kinetic energy /Set4_Kinetic_energy.xlsx')
rng=np.random.default_rng(20260922); NPERM=10000; NP=['A1','A3','A4','A5','A6']
wb=openpyxl.load_workbook(XL,data_only=True)
def sheet(name,col):
    ws=wb[name]; rows=[]
    for r in ws.iter_rows(min_row=2,values_only=True):
        if r[0] is None: continue
        for i in range(6):
            if r[1+i] is not None: rows.append(dict(trial=int(r[0]),Mouse_ID='A%d'%(i+1),**{col:float(r[1+i])}))
    return pd.DataFrame(rows)
F=sheet('kinetic energy group_foraging','ke_act'); B=sheet('Kinetic_energy_group_baseline','ke_base')
M=pd.read_csv('two_signals_trials.csv').merge(F,on=['Mouse_ID','trial'],how='left').merge(B,on=['Mouse_ID','trial'],how='left')
M['logL']=np.log10(M.L.where(M.L>0)); M['lke_act']=np.log10(M.ke_act.clip(lower=1e-8)); M['lke_base']=np.log10(M.ke_base.clip(lower=1e-8))
print('결측 act %d base %d / %d'%(M.ke_act.isna().sum(),M.ke_base.isna().sum(),len(M)))
print('기저 운동량 대 채집 운동량 상관(원시) r=%.3f'%M[['lke_base','lke_act']].corr().iloc[0,1])
def resid(Mx,g,pos):
    out=np.empty_like(Mx,dtype=float)
    for u in np.unique(g):
        i=np.where(g==u)[0]; X=np.column_stack([np.ones(len(i)),pos[i]])
        out[i]=Mx[i]-X@np.linalg.lstsq(X,Mx[i],rcond=None)[0]
    return out
def fit(S,ycol,cols,label):
    g=S.Mouse_ID.values; pos=S.pos.values
    Z=resid(S[cols].values.astype(float),g,pos); Z=Z/Z.std(0)
    y=resid(S[[ycol]].values.astype(float),g,pos)[:,0]
    B_=np.linalg.lstsq(Z,y,rcond=None)[0]
    idx=[np.where(g==u)[0] for u in np.unique(g)]; null=np.empty((NPERM,len(cols)))
    for k in range(NPERM):
        yp=y.copy()
        for i in idx: yp[i]=y[rng.permutation(i)]
        null[k]=np.linalg.lstsq(Z,yp,rcond=None)[0]
    P=(np.abs(null)>=np.abs(B_)).mean(0); e=y-Z@B_
    print('  %-30s n=%-4d R2=%.4f  %s'%(label,len(S),1-e@e/(y@y),'  '.join('%s %+.4f (P=%.4f)'%(c,b,p) for c,b,p in zip(cols,B_,P))))
    per={}
    for u in np.unique(g):
        i=g==u; per[u]=[round(float(x),3) for x in np.linalg.lstsq(Z[i],y[i],rcond=None)[0]]
    return per
for tag,sel,ycol in [('Wait (followers, L>=3 s)',lambda d: d[d.Mouse_ID.isin(NP)&(d.follow==1)&(d.L>=3)],'logL'),
                     ('Follow vs stay out',lambda d: d[d.Mouse_ID.isin(NP)],'follow')]:
    S=sel(M).dropna(subset=['S_pre','S_post','lke_act','lke_base']).reset_index(drop=True)
    print('\n===== %s  (n=%d)'%(tag,len(S)))
    fit(S,ycol,['S_pre','S_post'],'재현')
    per=fit(S,ycol,['S_pre','S_post','lke_base'],'+ 기저 운동량 (사전 공변량)')
    print('      개체별 [S_pre,S_post,ke_base]:',per)
    fit(S,ycol,['S_pre','S_post','lke_act'],'+ 채집 운동량 (과잉 통제)')
    fit(S,ycol,['S_pre','S_post','lke_base','lke_act'],'+ 둘 다')
    fit(S,ycol,['lke_base'],'기저 운동량 단독')
    g=S.Mouse_ID.values; pos=S.pos.values
    Z=resid(S[['S_pre','S_post','lke_base','lke_act']].values.astype(float),g,pos); Z=Z/Z.std(0)
    print('  개체 내 상관: r(S_pre,ke_base) %+.3f  r(S_post,ke_base) %+.3f  r(S_post,ke_act) %+.3f  r(ke_base,ke_act) %+.3f'%(
        np.corrcoef(Z[:,0],Z[:,2])[0,1],np.corrcoef(Z[:,1],Z[:,2])[0,1],np.corrcoef(Z[:,1],Z[:,3])[0,1],np.corrcoef(Z[:,2],Z[:,3])[0,1]))
