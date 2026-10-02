# -*- coding: utf-8 -*-
"""Fig.4E 의 γ(51-100 Hz)를 원시 스펙트로그램에서 직접 계산해 동료 진입 정렬 분석을 다시 함."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, csv, os, re, glob, collections
D=(LFP + 'member_change')
OUT=(WORK + 'member_change')
GRID=(FIGDATA[:-1] + '/fig3a_daily_grid.csv')
STEP=0.125; PFC=2
FB={'g51_100':(50,100),'g35_50':(34,50),'hb24_32':(23,32)}   # 1 Hz 격자 인덱스(중심 ≈ 1+i)
WIN={'g51_100':(-3.0,1.6),'g35_50':(-3.0,1.6),'hb24_32':(-0.75,4.0)}
BASE=(-8.0,-5.0)
RIG={'A1':'1_2_3_7','A2':'1_2_3_7','A3':'1_2_3_7','A4':'4_5_0_8','A5':'4_5_0_8','A6':'4_5_0_8'}
SUBG={'active':{'A2','A5','A6'},'passive':{'A1','A3','A4'}}
SHIFT=60.0; EXCL={(29,7)}
grid=collections.defaultdict(dict)
for r in csv.DictReader(open(GRID)):
    if r['cohort']=='A': grid[int(r['day'])][r['mouse']]=(int(r['retrievals']),int(r['n_trials']))
state={}
for ex,d in grid.items():
    for mo,(ret,n) in d.items():
        if n:
            rate=ret/float(n); state[(ex+5,mo)]=('provider' if rate>=0.4 else 'withdrawn' if rate<=0.1 else 'mid',rate)
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
            if k.endswith('_spec'):
                mm=re.match(r'(\w+?)_T(-?\d+)_(A\d)_spec$',k)
                index[(int(m.group(1)),m.group(2),mm.group(1),int(mm.group(2)),mm.group(3))]=(f,k)
print('spec entries',len(index),flush=True)
cache={}
def spec(day,mouse,trial):
    key=(day,RIG[mouse],'Foraging',trial,mouse)
    if key not in index: return None
    f,k=index[key]
    if f not in cache: cache.clear(); cache[f]=np.load(f,allow_pickle=True)
    return cache[f][k]
def wm(s,t0,t1):
    i0=max(0,int(round(t0/STEP))); i1=min(len(s),int(round(t1/STEP)))
    if i1-i0<8: return np.nan
    v=s[i0:i1]; v=v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan
rows=[]; chk=True
for (day,lt,sub),evs in sorted(trials.items()):
    if (day,lt) in EXCL: continue
    present=SUBG[sub] if sub else None
    for M in ['A1','A2','A3','A4','A5','A6']:
        st=state.get((day,M))
        if st is None: continue
        if present is not None and M not in present: continue
        q=qual.get((day,'Foraging',lt,M))
        if q is None or q[0]==1 or q[1]>0.01: continue
        dur=q[2]; a=spec(day,M,lt)
        if a is None: continue
        if chk:
            chk=False
            prof=np.asarray(a[PFC],dtype=np.float32).mean(1)
            print('freq profile check: bin0 %.2f bin20 %.2f bin60 %.2f bin110 %.2f'%(prof[0],prof[20],prof[60],prof[110]),flush=True)
        S={bn:np.asarray(a[PFC,lo:hi,:],dtype=np.float32).mean(0) for bn,(lo,hi) in FB.items()}
        for mo,ev,t in evs:
            if ev!='entry' or mo==M: continue
            if present is not None and mo not in present: continue
            for kind,toff in (('real',0.0),('shift',SHIFT)):
                tt=(t+toff)%dur
                if tt+BASE[0]<0 or tt+4.0>dur: continue
                for bn,(w0,w1) in WIN.items():
                    v=wm(S[bn],tt+w0,tt+w1); b=wm(S[bn],tt+BASE[0],tt+BASE[1])
                    if np.isfinite(v) and np.isfinite(b):
                        rows.append((day,lt,sub,M,st[0],mo,kind,bn,round(t,1),round(v-b,5)))
with open(os.path.join(OUT,'q1_gamma51_100.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['day','trial','subgroup','mouse','state','event_mouse','kind','band','event_t','delta'])
    w.writerows(rows)
print('WROTE',len(rows),flush=True)
