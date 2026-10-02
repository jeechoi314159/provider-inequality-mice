# -*- coding: utf-8 -*-
"""시행 수준 tonic 비교. 출판 Fig.4C 의 elastic-net 가중치 방향을 그대로 쓰고,
척도는 구성 변경 자료 안에서 개체별 표준화. 사건 정밀도에 의존하지 않음."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, csv, os, re, glob, collections
D=(LFP + 'member_change')
OUT=(WORK + 'member_change')
SUB=(FIGDATA[:-1])
STEP=0.125
REGION=['BLA','NAc','PFC']; BANDS=['theta1','theta2','beta1','beta2','gamma1','gamma2']
PUBBAND={'low_theta':'theta1','high_theta':'theta2','low_beta':'beta1',
         'high_beta':'beta2','low_gamma':'gamma1','high_gamma':'gamma2'}
RIG={'A1':'1_2_3_7','A2':'1_2_3_7','A3':'1_2_3_7','A4':'4_5_0_8','A5':'4_5_0_8','A6':'4_5_0_8'}
SUBG={'active':{'A2','A5','A6'},'passive':{'A1','A3','A4'}}
EXCL={(29,7)}; MINPRE=4.0

# 가중치
Wt={}
for r in csv.DictReader(open(os.path.join(SUB,'fig4c_enet_weights.csv'))):
    Wt[(r['region'],PUBBAND[r['band']])]=float(r['weight'])
FEAT=[(g,b) for g in REGION for b in BANDS]
print('weights nonzero %d/18'%sum(1 for k in FEAT if Wt[k]!=0),flush=True)

# 역할 상태
grid=collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(SUB,'fig3a_daily_grid.csv'))):
    if r['cohort']=='A': grid[int(r['day'])+5][r['mouse']]=(int(r['retrievals']),int(r['n_trials']))
state={}
for dy,d in grid.items():
    for mo,(ret,n) in d.items():
        if n:
            rate=ret/float(n)
            state[(dy,mo)]=('provider' if rate>=0.4 else 'withdrawn' if rate<=0.1 else 'mid',rate)

# 전체 시행 power + 그날 기저
BP=list(csv.DictReader(open(os.path.join(D,'bandpower_whole.csv'))))
def cols(r): return {(g,b):float(r['%s_%s'%(g,b)]) for g in REGION for b in BANDS}
whole={}; base=collections.defaultdict(lambda: collections.defaultdict(list)); qual={}
for r in BP:
    day=int(r['day']); mo=r['mouse']; tr=int(r['trial'])
    v=cols(r)
    if r['segment']=='Baseline':
        for k,x in v.items():
            if x>0: base[(day,mo)][k].append(np.log10(x))
    else:
        whole[(day,tr,mo)]=v
        qual[(day,tr,mo)]=(int(float(r['frozen'])),float(r['artifact_frac']),float(r['seconds']))
dayref={k:{kk:float(np.median(vv)) for kk,vv in d.items()} for k,d in base.items()}
print('days with baseline:',len({k[0] for k in dayref}),flush=True)

# 사건 (진입 시각·시행 시작)
EV=[r for r in csv.DictReader(open(os.path.join(D,'timing_record_events.csv'))) if r['dataset']=='member_change']
tinfo=collections.defaultdict(lambda: dict(start=None,ent=[],sub=''))
for r in EV:
    k=(int(r['day']),int(r['lfp_trial']))
    tinfo[k]['sub']=r['subgroup']
    if r['trial_start_s']: tinfo[k]['start']=float(r['trial_start_s'])
    if r['event']=='entry': tinfo[k]['ent'].append((r['mouse'],float(r['time_s'])))

# npz 색인
index={}
for f in sorted(glob.glob(os.path.join(D,'spec_Day*_*.npz'))):
    m=re.match(r'spec_Day(\d+)_(.+)_p(\d+)\.npz$',os.path.basename(f))
    with np.load(f,allow_pickle=True) as z:
        for k in z.files:
            if k.endswith('_band'):
                mm=re.match(r'(\w+?)_T(-?\d+)_(A\d)_band$',k)
                index[(int(m.group(1)),m.group(2),mm.group(1),int(mm.group(2)),mm.group(3))]=(f,k)
cache={}
def band(day,mouse,trial):
    key=(day,RIG[mouse],'Foraging',trial,mouse)
    if key not in index: return None
    f,k=index[key]
    if f not in cache: cache.clear(); cache[f]=np.load(f,allow_pickle=True)
    return cache[f][k]

rows=[]
for (day,tr),info in sorted(tinfo.items()):
    if (day,tr) in EXCL: continue
    present=SUBG[info['sub']] if info['sub'] else None
    t0=info['start'] if info['start'] is not None else 0.0
    ent=sorted(info['ent'],key=lambda x:x[1])
    for mo in ['A1','A2','A3','A4','A5','A6']:
        st=state.get((day,mo))
        if st is None: continue
        if present is not None and mo not in present: continue
        q=qual.get((day,tr,mo))
        if q is None or q[0]==1 or q[1]>0.01: continue
        own=[t for m_,t in ent if m_==mo]
        t1=own[0] if own else (ent[0][1] if ent else None)
        rec=dict(day=day,trial=tr,subgroup=info['sub'],mouse=mo,state=st[0],work_rate=round(st[1],3),
                 entered=int(bool(own)),pre_s='')
        # 전체 시행
        w=whole.get((day,tr,mo)); ref=dayref.get((day,mo))
        for g,b in FEAT:
            x=w.get((g,b)) if w else None
            rec['whole_%s_%s'%(g,b)]=round(np.log10(x)-ref[(g,b)],5) if (x and x>0 and ref) else ''
        # 진입 전
        a=band(day,mo,tr)
        if a is not None and t1 is not None and t1-t0>=MINPRE:
            i0=max(0,int(round(t0/STEP))); i1=min(a.shape[-1],int(round(t1/STEP)))
            rec['pre_s']=round((i1-i0)*STEP,2)
            for gi,g in enumerate(REGION):
                for bi,b in enumerate(BANDS):
                    v=a[gi,bi,i0:i1]; v=v[np.isfinite(v)&(v>0)]
                    rec['pre_%s_%s'%(g,b)]=round(float(np.log10(v).mean())-ref[(g,b)],5) if (v.size>=32 and ref) else ''
        else:
            for g,b in FEAT: rec['pre_%s_%s'%(g,b)]=''
        rows.append(rec)
hdr=['day','trial','subgroup','mouse','state','work_rate','entered','pre_s']+ \
    ['whole_%s_%s'%(g,b) for g,b in FEAT]+['pre_%s_%s'%(g,b) for g,b in FEAT]
with open(os.path.join(OUT,'q2_tonic_features.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=hdr); w.writeheader(); w.writerows(rows)
print('WROTE',len(rows),flush=True)
