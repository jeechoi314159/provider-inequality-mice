# Two withdrawal-linked signals together: S_pre = PFC γ (51–100 Hz; reference 2-D cluster band), −2…0 s before the cagemate's entry;
# S_post = PFC high β (24–32 Hz), 0.25–2.25 s after it. Within-mouse, mouse-specific session trend removed, within-mouse permutation.
import numpy as np, pandas as pd, pickle
from scan import resid_mat
rng=np.random.default_rng(20260922); NPERM=10000
d=pickle.load(open('ext_data_s.pkl','rb')); meta,X,T=d['meta'],d['X'],d['T']
NP=['A1','A3','A4','A5','A6']
def wmean(key,tw):
    b=(T>=tw[0])&(T<tw[1]); A=X[key][:,b]; return np.where(np.isfinite(A).all(1),A.mean(1),np.nan)
meta['S_pre']=wmean(('PFC','g51_100'),(-2,0)); meta['S_post']=wmean(('PFC','hbeta'),(0.25,2.25))
def joint(S,ycol,label):
    g=S.Mouse_ID.values; pos=S.pos.values
    Z=resid_mat(S[['S_pre','S_post']].values.astype(float),g,pos); Z=Z/Z.std(0)
    y=resid_mat(S[[ycol]].values.astype(float),g,pos)[:,0]
    B=np.linalg.lstsq(Z,y,rcond=None)[0]; r=np.corrcoef(Z[:,0],Z[:,1])[0,1]
    idx=[np.where(g==u)[0] for u in np.unique(g)]; null=np.empty((NPERM,2))
    for k in range(NPERM):
        yp=y.copy()
        for i in idx: yp[i]=y[rng.permutation(i)]
        null[k]=np.linalg.lstsq(Z,yp,rcond=None)[0]
    P=(np.abs(null)>=np.abs(B)).mean(0)
    # variance explained (within-mouse): each alone and together
    def r2(M): b=np.linalg.lstsq(M,y,rcond=None)[0]; e=y-M@b; return 1-e@e/(y@y)
    R2=[r2(Z[:,[0]]),r2(Z[:,[1]]),r2(Z)]
    per={}
    for u in np.unique(g):
        i=g==u; per[u]=np.round(np.linalg.lstsq(Z[i],y[i],rcond=None)[0],3)
    print(f'\n== {label}: n={len(S)}, mice={S.Mouse_ID.nunique()}')
    print(f'  within-mouse r(S_pre,S_post) = {r:+.3f}')
    print(f'  joint: S_pre {B[0]:+.4f}/SD (P={P[0]:.4f}) | S_post {B[1]:+.4f}/SD (P={P[1]:.4f})')
    print(f'  within-mouse R²: S_pre alone {R2[0]:.4f}, S_post alone {R2[1]:.4f}, both {R2[2]:.4f}')
    print('  per mouse [S_pre, S_post]:', {k:list(v) for k,v in per.items()})
    return dict(label=label,n=len(S),r=r,b_pre=B[0],P_pre=P[0],b_post=B[1],P_post=P[1],R2_pre=R2[0],R2_post=R2[1],R2_both=R2[2],
                pre_neg=sum(v[0]<0 for v in per.values()),post_sign=sum(np.sign(v[1])==(-1 if ycol=='logL' else 1) for v in per.values()))
out=[]
S=meta[meta.Mouse_ID.isin(NP)&(meta.follow==1)&(meta.L>=3)].dropna(subset=['S_pre','S_post']).reset_index(drop=True)
out.append(joint(S,'logL','Wait (followers, L ≥ 3 s)'))
S2=meta[meta.Mouse_ID.isin(NP)].dropna(subset=['S_pre','S_post']).reset_index(drop=True)
out.append(joint(S2,'follow','Follow vs stay out'))
# per-mouse trial counts
print('\ntrials per mouse (wait / follow-model):', S.Mouse_ID.value_counts().to_dict(), S2.groupby('Mouse_ID').follow.agg(['sum','size']).to_dict('index'))
pd.DataFrame(out).to_csv('two_signals_joint.csv',index=False)
meta[['Mouse_ID','trial','follow','L','pos','StarterLat','S_pre','S_post']].to_csv('two_signals_trials.csv',index=False)
