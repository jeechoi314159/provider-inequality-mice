# Robustness of the scan clusters: LOMO cross-validated cluster, per-mouse signs, session control, latency cut-offs, provider comparison
import numpy as np, pandas as pd, pickle
from scipy.ndimage import label
from scipy import stats
from ref_load import load, prep
from scan import resid_mat
rng=np.random.default_rng(7)
def cl_mean(A,t,tw,fw): 
    ti=(t>=tw[0])&(t<tw[1]); return A[:,ti,:][:,:,fw[0]-1:fw[1]].mean((1,2))
def wslope(x,y,g,pos,ctrl=True,nperm=5000):
    if ctrl: x=resid_mat(x[:,None],g,pos)[:,0]; y=resid_mat(y[:,None],g,pos)[:,0]
    else:
        x=x-pd.Series(x).groupby(g).transform('mean').values; y=y-pd.Series(y).groupby(g).transform('mean').values
    x=x/x.std(); b=(x@y)/(x@x); idx=[np.where(g==u)[0] for u in np.unique(g)]; null=np.empty(nperm)
    for k in range(nperm):
        yp=y.copy()
        for i in idx: yp[i]=y[rng.permutation(i)]
        null[k]=(x@yp)/(x@x)
    per={u:float(np.sign(x[g==u]@y[g==u])) for u in np.unique(g)}
    return b,np.mean(np.abs(null)>=abs(b)),per
def lomo(block,outcome,region,var,minL,sign,CF=0.10):
    """Define the cluster on 4 NP mice (largest same-sign cluster), score the held-out mouse; pool held-out scores."""
    meta,t,X=load(*{'cag':('ed_spectro_','cag.pkl'),'own':('ed_spectro_own_','own.pkl')}[block]); m=prep(meta)
    sel=((m.Role=='NP')&(m.FollowerLatency>=minL)).values; m=m[sel].reset_index(drop=True); A=X[region][sel].astype(float)
    if var=='bl':
        bl=(t>=-2)&(t<=-0.5); A=A-A[:,bl,:].mean(1,keepdims=True); keep=t>=-0.5; A=A[:,keep,:]; t=t[keep]
    g=m.Mouse_ID.values; pos=m.pos.values; y=m[outcome].values; score=np.full(len(m),np.nan); defs={}
    for hold in np.unique(g):
        tr=g!=hold; gt=g[tr]; n=tr.sum(); df=n-2*len(np.unique(gt))-1
        M=resid_mat(A[tr].reshape(n,-1),gt,pos[tr]); M=(M-M.mean(0))/M.std(0)
        yy=resid_mat(y[tr][:,None],gt,pos[tr])[:,0]; yy=(yy-yy.mean())/yy.std()
        R=(M.T@yy)/n; T=(R*np.sqrt(df/(1-R**2))).reshape(len(t),100)
        lab,k=label(sign*T>stats.t.ppf(1-CF/2,df))
        if k==0: continue
        masses=[np.abs(T[lab==c]).sum() for c in range(1,k+1)]; mk=lab==(1+int(np.argmax(masses)))
        ti,fi=np.where(mk); defs[hold]=(t[ti.min()],t[ti.max()]+0.125,fi.min()+1,fi.max()+1)
        score[g==hold]=A[g==hold][:,mk].mean(1)
    ok=~np.isnan(score)
    b,p,per=wslope(score[ok],y[ok],g[ok],pos[ok])
    return b,p,per,defs
if __name__=='__main__':
    out=[]
    print('=== Cluster 1: cagemate block, wait, PFC raw, 51–100 Hz, −2.0–0.625 s (negative)')
    meta,t,X=load('ed_spectro_','cag.pkl'); m=prep(meta)
    for minL in [0,3,4,5]:
        s=((m.Role=='NP')&(m.FollowerLatency>=minL)).values; mm=m[s]; x=cl_mean(X['PFC'][s].astype(float),t,(-2,0.625),(51,100))
        for ctrl in [True,False]:
            b,p,per=wslope(x,mm.logL.values,mm.Mouse_ID.values,mm.pos.values,ctrl)
            print(f'  L≥{minL}s n={s.sum()} session ctrl={ctrl}: slope/sd {b:+.4f} P={p:.4f} per-mouse signs {per}')
    # windows sweep inside the cluster band: pre-entry only vs post-entry only
    s=((m.Role=='NP')&(m.FollowerLatency>=3)).values; mm=m[s]
    for tw in [(-2,-1),(-1,0),(0,1),(1,2),(2,2.875),(-2,0),(0,2)]:
        x=cl_mean(X['PFC'][s].astype(float),t,tw,(51,100)); b,p,per=wslope(x,mm.logL.values,mm.Mouse_ID.values,mm.pos.values)
        print(f'  window {tw}: slope/sd {b:+.4f} P={p:.4f} mice negative {sum(v<0 for v in per.values())}/5')
    # provider (A2) following an NP: single-mouse regression, session-controlled
    s=((m.Role=='P')&(m.FollowerLatency>=3)).values; mm=m[s]; x=cl_mean(X['PFC'][s].astype(float),t,(-2,0.625),(51,100))
    Xd=np.column_stack([np.ones(s.sum()),mm.pos]); xr=x-Xd@np.linalg.lstsq(Xd,x,rcond=None)[0]; yr=mm.logL.values-Xd@np.linalg.lstsq(Xd,mm.logL.values,rcond=None)[0]
    r,p=stats.pearsonr(xr,yr); print(f'  provider A2 (n={s.sum()}): r={r:+.3f} P={p:.3f}')
    b,p,per,defs=lomo('cag','logL','PFC','raw',3,-1)
    print(f'  LOMO cross-validated (cluster defined without the tested mouse): slope/sd {b:+.4f} P={p:.4f} signs {per}\n   cluster defs {defs}')
    print('\n=== Cluster 2: own-entry block, provider vs NP starter, PFC baseline-subtracted, 27–73 Hz, −0.25–2 s (negative)')
    b,p,per,defs=lomo('own','fP','PFC','bl',2.5,-1)
    print(f'  LOMO cross-validated: slope/sd {b:+.4f} P={p:.4f} signs {per}\n   cluster defs {defs}')
    print('\n=== Cluster 3: own-entry block, wait, NAc raw, 44–100 Hz, −0.125–2 s (positive)')
    b,p,per,defs=lomo('own','logL','NAc','raw',2.5,+1)
    print(f'  LOMO cross-validated: slope/sd {b:+.4f} P={p:.4f} signs {per}\n   cluster defs {defs}')
