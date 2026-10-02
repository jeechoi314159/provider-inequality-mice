# 1-D time scan (−8…+3 s from the cagemate's entry), 7 bands x 3 regions, within-mouse + session control,
# cluster-based permutation over time with family-wise max across the 21 maps (Maris & Oostenveld 2007)
import numpy as np, pandas as pd, pickle, sys
from scipy.ndimage import label
from scipy import stats
d=pickle.load(open('ext_data_s.pkl','rb')); meta,X,T=d['meta'],d['X'],d['T']
NP=['A1','A3','A4','A5','A6']; NPERM=2000; CF=0.10
def design(g,pos):
    u=np.unique(g); D=np.column_stack([(g==v).astype(float) for v in u]+[pos-pos.mean()]+[ (g==v)*(pos-pos[g==v].mean()) for v in u])
    return D   # mouse intercepts + mouse-specific session slopes
def run(name,sel,ycol,minN=40,seed=1):
    rng=np.random.default_rng(seed)
    m=meta[sel].reset_index(drop=True); g=m.Mouse_ID.values; pos=m.pos.values; y=m[ycol].values.astype(float); n=len(m)
    idx=[np.where(g==u)[0] for u in np.unique(g)]
    Yp=np.empty((n,NPERM+1)); Yp[:,0]=y
    for k in range(1,NPERM+1):
        yp=y.copy()
        for i in idx: yp[i]=y[rng.permutation(i)]
        Yp[:,k]=yp
    keys=list(X.keys()); Tm=np.full((len(keys),len(T),NPERM+1),np.nan,np.float32); nb=np.zeros(len(T),int)
    for b in range(len(T)):
        av=np.all([np.isfinite(X[k][sel][:,b]) for k in keys],axis=0)
        if av.sum()<minN or len(np.unique(g[av]))<len(np.unique(g)): continue
        nb[b]=av.sum(); Dm=design(g[av],pos[av]); H=Dm@np.linalg.pinv(Dm); R=np.eye(av.sum())-H
        Yr=R@Yp[av]; Yr/=np.linalg.norm(Yr,axis=0)
        df=av.sum()-np.linalg.matrix_rank(Dm)-1
        for ki,k in enumerate(keys):
            x=R@X[k][sel][av,b].astype(float); x/=np.linalg.norm(x); r=x@Yr; Tm[ki,b]=r*np.sqrt(df/(1-r**2))
    thr=stats.t.ppf(1-CF/2,max(nb[nb>0].min()-12,10))
    def cl(v):
        out=[]
        for s in [1,-1]:
            lab,kk=label(np.nan_to_num(s*v)>thr)
            for c in range(1,kk+1): out.append((np.abs(v[lab==c]).sum(),s,lab==c))
        return out
    null=np.zeros(NPERM)
    for p in range(1,NPERM+1):
        null[p-1]=max([c[0] for ki in range(len(keys)) for c in cl(Tm[ki,:,p])] or [0])
    rows=[]
    for ki,k in enumerate(keys):
        for mass,s,mk in cl(Tm[ki,:,0]):
            tt=T[mk]; rows.append(dict(analysis=name,region=k[0],band=k[1],t_start=tt.min(),t_end=tt.max()+0.125,sign='+' if s>0 else '−',
                                     mass=round(mass,1),peak_t=round(float(np.nanmax(np.abs(Tm[ki,mk,0]))*s),2),P_family=np.mean(null>=mass),n_min=int(nb[mk].min()),n_max=int(nb[mk].max())))
    res=pd.DataFrame(rows).sort_values('P_family')
    pickle.dump(dict(keys=keys,T0=Tm[:,:,0],nb=nb,thr=thr),open(f'ext_s_{name}.pkl','wb'))
    return res
if __name__=='__main__':
    NPm=meta.Mouse_ID.isin(NP)
    out=[]
    r=run('wait_NP',(NPm&(meta.follow==1)&(meta.L>=3)).values,'logL'); out.append(r); print('\n### E1 wait (NP followers, L≥3 s)'); print(r[r.P_family<0.5].to_string(index=False))
    r=run('follow_NP',NPm.values,'follow'); out.append(r); print('\n### E2 follow vs stay out (NP)'); print(r[r.P_family<0.5].to_string(index=False))
    pd.concat(out).to_csv('ext_clusters_smoothed.csv',index=False)
