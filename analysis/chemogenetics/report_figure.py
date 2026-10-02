# -*- coding: utf-8 -*-
"""보고용 요약표와 그림. 채널 대응은 확정값(ch1=PFC, ch2=NAc, ch3=BLA)을 씀."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, re, csv, glob, math, collections, statistics as st
import numpy as np, openpyxl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
BASE = (CHEMO[:-1])
D, OUT = BASE+'/data', BASE+'/out'; os.makedirs(OUT, exist_ok=True)
COL2REAL = {'BLA':'PFC','NAc':'NAc','PFC':'BLA'}      # 추출 열 이름 → 실제 영역
REAL2COL = {v:k for k,v in COL2REAL.items()}
BANDS = ['theta1','theta2','beta1','beta2','gamma1','gamma2']
LBL = {'theta1':r'$\theta_{low}$','theta2':r'$\theta_{high}$','beta1':r'$\beta_{low}$',
       'beta2':r'$\beta_{high}$','gamma1':r'$\gamma_{low}$','gamma2':r'$\gamma_{high}$'}
B = list(csv.DictReader(open(D+'/bandpower_whole.csv')))
rng = np.random.default_rng(20260930); NPERM = 10000

def onef(R,c):
    v=[math.log10(float(r[c+'_theta1']))-math.log10(float(r[c+'_gamma2'])) for r in R
       if float(r[c+'_theta1'])>0 and float(r[c+'_gamma2'])>0]
    return st.median(v) if v else float('nan')
g=collections.defaultdict(list)
for r in B: g[(r['cohort'],r['condition'],int(r['device']))].append(r)
ok={}
for k,R in g.items():
    o=[onef(R,c) for c in ('BLA','NAc','PFC')]
    sd=[st.median(float(r['sd_ch%d'%i]) for r in R) for i in (1,2,3)]
    synth=(max(o)-min(o)<0.03) and (max(sd)-min(sd))/max(sd)<0.02
    for i,c in enumerate(('BLA','NAc','PFC')): ok[(k[0],k[1],k[2],c)]=(not synth) and (0.01<sd[i]<0.5)
use={}
for coh in sorted({k[0] for k in g}):
    for dev in sorted({k[2] for k in g if k[0]==coh}):
        if not all((coh,c,dev) in g for c in ('CNO','Saline')): continue
        keep=[c for c in ('BLA','NAc','PFC') if ok.get((coh,'CNO',dev,c)) and ok.get((coh,'Saline',dev,c))]
        if keep: use[(coh,dev)]=keep
def perm(y,x,grp,n=NPERM):
    y=y.copy();x=x.copy()
    for u in np.unique(grp):
        i=grp==u; y[i]-=y[i].mean(); x[i]-=x[i].mean()
    b=(x@y)/(x@x); c=0
    for _ in range(n):
        xp=x.copy()
        for u in np.unique(grp):
            i=np.where(grp==u)[0]; xp[i]=x[rng.permutation(i)]
        if abs((xp@y)/(xp@xp))>=abs(b): c+=1
    return b,(c+1)/(n+1)
rows=[]; res={}
for seg in ['Baseline','Foraging']:
    for real in ['PFC','NAc','BLA']:
        col=REAL2COL[real]
        for bn in BANDS:
            y=[];x=[];gg=[];cohs=set()
            for r in B:
                if r['segment']!=seg: continue
                k=(r['cohort'],int(r['device']))
                if k not in use or col not in use[k]: continue
                if float(r['artifact_frac'])>0.01: continue
                v=float(r[col+'_'+bn])
                if v<=0: continue
                y.append(math.log10(v)); x.append(1.0 if r['condition']=='CNO' else 0.0)
                gg.append(r['cohort']); cohs.add(r['cohort'])
            if len(y)<12: continue
            b,P=perm(np.array(y),np.array(x),np.array(gg))
            res[(seg,real,bn)]=(b,P)
            rows.append(dict(analysis='manipulation',segment=seg,region=real,band=bn,
                             n_segments=len(y),cohorts=''.join(sorted(cohs)),
                             effect_log10=round(b,4),P=round(P,4)))
print('조작 검증 완료', flush=True)
# 행동
BEH=[]
for f in sorted(glob.glob(BASE+'/behavior/*timestamp_saline_CNO_condition.xlsx')):
    coh=re.search(r'Group(\w)',f).group(1); ws=openpyxl.load_workbook(f,data_only=True)['Sheet1']
    hdr=list(next(ws.iter_rows(min_row=1,max_row=1,values_only=True)))
    for r in ws.iter_rows(min_row=2,values_only=True):
        if r[0] is None: continue
        d=dict(zip(hdr,r)); d['cohort']=coh
        d['condition']='CNO' if str(d['Condition']).upper()=='CNO' else 'Saline'; BEH.append(d)
def num(v):
    try:
        x=float(v); return None if x!=x else x
    except: return None
beh={}
for coh in ['E','F','G','H']:
    R=[b for b in BEH if b['cohort']==coh]; o={}
    for cond in ['Saline','CNO']:
        S=[b for b in R if b['condition']==cond]
        got=[1 if num(b['get']) is not None else 0 for b in S]
        tg=[num(b['Tau_get']) for b in S if num(b['Tau_get']) is not None]
        o[cond]=(sum(got)/len(S), st.median(tg) if tg else float('nan'), len(S))
    beh[coh]=o
    rows.append(dict(analysis='behaviour',segment='provider',region='',band='retrieval_rate',
                     n_segments=o['Saline'][2]+o['CNO'][2],cohorts=coh,
                     effect_log10=round(o['CNO'][0]-o['Saline'][0],3),P=''))
    rows.append(dict(analysis='behaviour',segment='provider',region='',band='tau_get_median_s',
                     n_segments=o['Saline'][2]+o['CNO'][2],cohorts=coh,
                     effect_log10=round(o['CNO'][1]-o['Saline'][1],2),P=''))
with open(OUT+'/dreadd_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('요약표 저장', flush=True)

# ---- 그림
plt.rcParams.update({'font.size':7,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':7,'axes.linewidth':0.6,'xtick.major.width':0.6,
                     'ytick.major.width':0.6,'savefig.dpi':300,'pdf.fonttype':42})
INK='#1a1a1a'; GRID='#d9d6d0'; C_BASE='#8c8c8c'; C_FOR='#c1440e'; PROV='#c1440e'; OTH='#6b6b6b'
fig=plt.figure(figsize=(6.9,4.6))
L0,W,GAPX = 0.115, 0.245, 0.072
for i,real in enumerate(['PFC','NAc','BLA']):
    ax=fig.add_axes([L0+i*(W+GAPX),0.635,W,0.285])
    xs=np.arange(len(BANDS))
    for j,(seg,c,lab) in enumerate((('Baseline',C_BASE,'Baseline'),('Foraging',C_FOR,'Foraging'))):
        v=[res.get((seg,real,b),(np.nan,1))[0] for b in BANDS]
        p_=[res.get((seg,real,b),(np.nan,1))[1] for b in BANDS]
        ax.bar(xs+(j-0.5)*0.38,v,0.34,color=c,lw=0,label=lab)
        for k,(vv,pp) in enumerate(zip(v,p_)):
            if pp<0.05 and np.isfinite(vv):
                ax.text(xs[k]+(j-0.5)*0.38, 0.232,'*',ha='center',va='center',fontsize=8,color=c)
    ax.axhline(0,color=INK,lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([LBL[b] for b in BANDS])
    ax.set_ylim(-0.215,0.265); ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_title('%s  (%s)'%(real,'cohorts E, F, H' if real!='BLA' else 'cohorts E, F, G, H'),
                 fontsize=8,pad=4)
    if i==0: ax.set_ylabel('CNO − Saline\n(log$_{10}$ power)',labelpad=2)
    else: ax.set_yticklabels([])
    for s_ in ('top','right'): ax.spines[s_].set_visible(False)
    ax.tick_params(length=2.5,colors=INK); ax.set_axisbelow(True)
    ax.yaxis.grid(True,color=GRID,lw=0.5)
h,l=fig.axes[0].get_legend_handles_labels()
fig.legend(h[:2],l[:2],frameon=False,ncol=2,loc='lower center',
           bbox_to_anchor=(0.5,0.545),handlelength=1.0,columnspacing=1.6)
fig.text(0.5,0.515,'* permutation $P$ < 0.05 (within-cohort, 10,000 permutations)',
         ha='center',fontsize=6.5,color='#555555')
def labelled(ax,pairs,xr=1.06,minsep=0.045):
    ys=sorted(pairs,key=lambda t:t[1]); span=ax.get_ylim()[1]-ax.get_ylim()[0]; last=None
    for coh,y in ys:
        yy=y if last is None or y-last>minsep*span else last+minsep*span
        ax.text(xr,yy,coh,fontsize=7,color=(PROV if coh=='F' else OTH),va='center'); last=yy
for i,(key,ylab,lo,hi) in enumerate((('rate','Retrieved by provider\n(fraction of trials)',-0.06,1.12),
                                     ('tau','Retrieval time\n(median, s)',0,34))):
    ax=fig.add_axes([L0+i*(W+GAPX),0.095,W,0.315])
    pts=[]
    for coh in ['E','F','G','H']:
        o=beh[coh]; y=[o['Saline'][0 if key=='rate' else 1],o['CNO'][0 if key=='rate' else 1]]
        col=PROV if coh=='F' else OTH
        ax.plot([0,1],y,'-o',color=col,lw=1.0,ms=4,mfc=col,mec='none',zorder=3 if coh=='F' else 2)
        pts.append((coh,y[1]))
    ax.set_xlim(-0.25,1.35); ax.set_ylim(lo,hi)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Saline','CNO'])
    ax.set_ylabel(ylab,labelpad=2)
    for s_ in ('top','right'): ax.spines[s_].set_visible(False)
    ax.tick_params(length=2.5,colors=INK); ax.set_axisbelow(True); ax.yaxis.grid(True,color=GRID,lw=0.5)
    labelled(ax,pts)
ax=fig.add_axes([L0+2*(W+GAPX),0.095,W,0.315]); ax.axis('off')
ax.text(0,1.0,'Provider of each cohort\n(E1, F4, G2, H4);\ncohort F in orange\n\n'
              'Retrieved by provider\n23/24 → 18/23\nFisher $P$ = 0.097\n\n'
              'Retrieval time × 1.56\ntrial-level $P$ = 0.089',
        fontsize=7,va='top',color=INK,linespacing=1.6)
for lab,x,y in (('A',0.012,0.955),('B',0.012,0.445)):
    fig.text(x,y,lab,fontsize=10,fontweight='bold',color=INK)
fig.savefig(OUT+'/dreadd_report_fig.png',dpi=300,facecolor='white')
print('그림 저장', flush=True)
