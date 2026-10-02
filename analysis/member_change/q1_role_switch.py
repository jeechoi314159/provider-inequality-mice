# -*- coding: utf-8 -*-
"""1차 질문 + 진단. 정렬 3종(동료 진입·자기 진입·자기 회수) x 영역 3 x 대역 6 x (실제/순환이동)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, csv, os, re, glob, collections

D   = (LFP + 'member_change')
OUT = (WORK + 'member_change')
GRID= (FIGDATA[:-1] + '/fig3a_daily_grid.csv')
STEP=0.125
REGION=['BLA','NAc','PFC']; BANDS=['theta1','theta2','beta1','beta2','gamma1','gamma2']
WIN={'gamma2':(-3.0,1.6),'gamma1':(-3.0,1.6),'beta2':(-0.75,4.0),'beta1':(-0.75,4.0),
     'theta1':(-3.0,1.6),'theta2':(-3.0,1.6)}
BASE=(-8.0,-5.0)
RIG={'A1':'1_2_3_7','A2':'1_2_3_7','A3':'1_2_3_7','A4':'4_5_0_8','A5':'4_5_0_8','A6':'4_5_0_8'}
SUBG={'active':{'A2','A5','A6'},'passive':{'A1','A3','A4'}}
TARGET=['A1','A2','A3','A4','A5','A6']
SHIFT=60.0; EXCL={(29,7)}

grid=collections.defaultdict(dict)
for r in csv.DictReader(open(GRID)):
    if r['cohort']=='A': grid[int(r['day'])][r['mouse']]=(int(r['retrievals']),int(r['n_trials']))
state={}
for ex,d in grid.items():
    for mo,(ret,n) in d.items():
        if n:
            rate=ret/float(n)
            state[(ex+5,mo)]=('provider' if rate>=0.4 else 'withdrawn' if rate<=0.1 else 'mid',rate)

EV=[r for r in csv.DictReader(open(os.path.join(D,'timing_record_events.csv'))) if r['dataset']=='member_change']
trials=collections.defaultdict(list)
for r in EV: trials[(int(r['day']),int(r['lfp_trial']),r['subgroup'])].append((r['mouse'],r['event'],float(r['time_s'])))
qual={}
for r in csv.DictReader(open(os.path.join(D,'bandpower_whole.csv'))):
    qual[(int(r['day']),r['segment'],int(r['trial']),r['mouse'])]=(int(float(r['frozen'])),float(r['artifact_frac']),float(r['seconds']))
index={}
for f in sorted(glob.glob(os.path.join(D,'spec_Day*_*.npz'))):
    m=re.match(r'spec_Day(\d+)_(.+)_p(\d+)\.npz$',os.path.basename(f))
    with np.load(f,allow_pickle=True) as z:
        for k in z.files:
            if k.endswith('_band'):
                mm=re.match(r'(\w+?)_T(-?\d+)_(A\d)_band$',k)
                index[(int(m.group(1)),m.group(2),mm.group(1),int(mm.group(2)),mm.group(3))]=(f,k)
print('npz entries',len(index),flush=True)
cache={}
def series(day,mouse,seg,trial):
    key=(day,RIG[mouse],seg,trial,mouse)
    if key not in index: return None
    f,k=index[key]
    if f not in cache: cache.clear(); cache[f]=np.load(f,allow_pickle=True)
    return cache[f][k]
def wm(s,t0,t1):
    i0=max(0,int(round(t0/STEP))); i1=min(len(s),int(round(t1/STEP)))
    if i1-i0<8: return np.nan
    v=s[i0:i1]; v=v[np.isfinite(v)&(v>0)]
    return np.log10(v).mean() if v.size else np.nan

rows=[]
for (day,lt,sub),evs in sorted(trials.items()):
    if (day,lt) in EXCL: continue
    present=SUBG[sub] if sub else None
    ent=[(mo,t) for mo,ev,t in evs if ev=='entry']
    for M in TARGET:
        st=state.get((day,M))
        if st is None: continue
        if present is not None and M not in present: continue
        q=qual.get((day,'Foraging',lt,M))
        if q is None or q[0]==1 or q[1]>0.01: continue
        dur=q[2]
        a=series(day,M,'Foraging',lt)
        if a is None: continue
        evlist=[]
        for mo,ev,t in evs:
            if present is not None and mo not in present: continue
            if ev=='entry' and mo!=M: evlist.append(('mate_entry',mo,t))
            elif ev=='entry' and mo==M: evlist.append(('own_entry',mo,t))
            elif ev=='retrieval' and mo==M: evlist.append(('own_retrieval',mo,t))
        for align,mo,te in evlist:
            nnear=sum(1 for _m,_t in ent if _m!=mo and te-8.0<=_t<=te+4.0)
            for kind,toff in (('real',0.0),('shift',SHIFT)):
                tt=(te+toff)%dur
                if tt+BASE[0]<0 or tt+max(w[1] for w in WIN.values())>dur: continue
                for ri,rg in enumerate(REGION):
                    for bi,bn in enumerate(BANDS):
                        s=a[ri,bi,:]
                        w0,w1=WIN[bn]
                        v=wm(s,tt+w0,tt+w1); b=wm(s,tt+BASE[0],tt+BASE[1])
                        if not (np.isfinite(v) and np.isfinite(b)): continue
                        rows.append((day,lt,sub,M,st[0],round(st[1],3),align,mo,nnear,kind,rg,bn,
                                     round(te,1),round(v-b,5),round(v,5)))
with open(os.path.join(OUT,'q1_events.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['day','trial','subgroup','mouse','state','work_rate','align','event_mouse',
                                 'n_other_entries_in_window','kind','region','band','event_t','delta','win_log10'])
    w.writerows(rows)
print('WROTE',len(rows),flush=True)
