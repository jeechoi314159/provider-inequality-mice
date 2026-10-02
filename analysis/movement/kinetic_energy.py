# Kinetic energy (cohort A, accelerometer-derived per trial x mouse): effort cost, withdrawal ≠ inactivity, movement covariate for N13/N14
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy import stats
import sys; sys.path.insert(0,(LFPCODE))
from robust import wslope
from scan import resid_mat
P=(BEHAV + 'Kinetic energy /Set4_Kinetic_energy.xlsx')
x=pd.ExcelFile(P)
def sheet(n):
    d=pd.read_excel(P,sheet_name=n); d.columns=['trial']+[f'A{i}' for i in range(1,7)]; return d.melt('trial',var_name='Mouse_ID',value_name=n)
KE=sheet('kinetic energy group_foraging').merge(sheet('Kinetic_energy_group_baseline'),on=['trial','Mouse_ID']).merge(sheet('kinetic energy eating'),on=['trial','Mouse_ID'])
KE.columns=['trial','Mouse_ID','KE_act','KE_base','KE_eat']
for c in ['KE_act','KE_base','KE_eat']: KE['log_'+c]=np.log10(KE[c])
beh=pd.read_csv((INTER + 'follower_retrieval_latency.csv')); A=beh[beh.Group=='A'].rename(columns={'Trial':'trial'})
D=KE.merge(A[['trial','Mouse_ID','Entered','Worked','TrialRole','EntryLatency']],on=['trial','Mouse_ID'],how='inner')
D['pos']=(D.trial-1)/(D.trial.max()-1); D['state']=np.where(D.Worked==1,'retrieved',np.where(D.Entered==1,'entered only','stayed out'))
print('rows',len(D),'trials',D.trial.nunique())
# (1) effort: within trial, retriever vs others (log KE_act), paired within trial
g=D.groupby(['trial','state']).log_KE_act.mean().unstack()
for s in ['entered only','stayed out']:
    ok=g[['retrieved',s]].dropna(); t,p=stats.wilcoxon(ok['retrieved'],ok[s]); print(f'effort: retrieved vs {s}: median diff log10 KE {np.median(ok.retrieved-ok[s]):+.3f} (×{10**np.median(ok.retrieved-ok[s]):.2f}), n={len(ok)} trials, Wilcoxon P={p:.2g}')
# (2) withdrawal ≠ inactivity: within-mouse session trends of KE (NP) vs retrieval
NP=['A1','A3','A4','A5','A6']
for c in ['log_KE_act','log_KE_base']:
    line=f'{c} session trend:'
    for m in ['A2']+NP:
        s=D[D.Mouse_ID==m]; r,p=stats.spearmanr(s.pos,s[c]); line+=f' {m} ρ={r:+.2f}(P={p:.2g})'
    print(line)
line='retrieval session trend:'
for m in ['A2']+NP:
    s=D[D.Mouse_ID==m]; r,p=stats.spearmanr(s.pos,s.Worked); line+=f' {m} ρ={r:+.2f}(P={p:.2g})'
print(line)
# KE while staying out vs entering, within mouse (is staying out = resting?)
for m in NP:
    s=D[D.Mouse_ID==m]; a=s[s.Entered==1].log_KE_act; b=s[s.Entered==0].log_KE_act
    if len(b)>3: print(f'  {m}: KE_act entered vs stayed out: {a.median():.2f} vs {b.median():.2f} (MWU P={stats.mannwhitneyu(a,b).pvalue:.2g}, n={len(a)}/{len(b)})')
D.to_csv('ke_trials.csv',index=False)
# (3) movement covariates for N13/N14
T=pd.read_csv((LFPCODE + '/two_signals_trials.csv')); T['logL']=np.log10(T.L.where(T.L>0))
M=T.merge(D[['trial','Mouse_ID','log_KE_act','log_KE_base']],on=['trial','Mouse_ID'],how='left')
print('\nN13/N14 with trial-level movement covariates (KE during foraging and baseline), within-mouse, session trend removed:')
rng=np.random.default_rng(1)
def joint_cov(S,ycol,cols):
    g=S.Mouse_ID.values; pos=S.pos.values
    Z=resid_mat(S[cols].values.astype(float),g,pos); Z=Z/Z.std(0); y=resid_mat(S[[ycol]].values.astype(float),g,pos)[:,0]
    B=np.linalg.lstsq(Z,y,rcond=None)[0]; idx=[np.where(g==u)[0] for u in np.unique(g)]; null=np.empty((5000,len(cols)))
    for k in range(5000):
        yp=y.copy()
        for i in idx: yp[i]=y[rng.permutation(i)]
        null[k]=np.linalg.lstsq(Z,yp,rcond=None)[0]
    return dict(zip(cols,zip(np.round(B,4),(np.abs(null)>=np.abs(B)).mean(0))))
base=M[M.Mouse_ID.isin(NP)].replace([np.inf,-np.inf],np.nan).dropna(subset=['log_KE_act','log_KE_base'])
W=base[(base.follow==1)&(base.L>=3)].dropna(subset=['S_pre']).reset_index(drop=True)
print('  wait ~ S_pre + KE_act + KE_base:', joint_cov(W,'logL',['S_pre','log_KE_act','log_KE_base']), 'n',len(W))
F=base.dropna(subset=['S_post']).reset_index(drop=True)
print('  follow ~ S_post + KE_act + KE_base:', joint_cov(F,'follow',['S_post','log_KE_act','log_KE_base']), 'n',len(F))
W2=base[(base.follow==1)&(base.L>=3)].dropna(subset=['S_post']).reset_index(drop=True)
print('  wait ~ S_post + KE_act + KE_base:', joint_cov(W2,'logL',['S_post','log_KE_act','log_KE_base']), 'n',len(W2))
for s,c in [('S_pre','log_KE_base'),('S_post','log_KE_act'),('S_post','log_KE_base'),('S_pre','log_KE_act')]:
    S=base.dropna(subset=[s]); a=resid_mat(S[[s,c]].values.astype(float),S.Mouse_ID.values,S.pos.values); print(f'  within-mouse r({s},{c}) = {np.corrcoef(a[:,0],a[:,1])[0,1]:+.3f} n={len(S)}')
print('\nPre-outcome covariate only (KE during pre-trial baseline; KE during foraging is a consequence of entering):')
print('  wait ~ S_pre + KE_base:', joint_cov(W,'logL',['S_pre','log_KE_base']), 'n',len(W))
print('  follow ~ S_post + KE_base:', joint_cov(F,'follow',['S_post','log_KE_base']), 'n',len(F))
print('  wait ~ S_post + KE_base:', joint_cov(W2,'logL',['S_post','log_KE_base']), 'n',len(W2))
