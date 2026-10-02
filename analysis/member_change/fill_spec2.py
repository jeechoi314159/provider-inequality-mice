import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, re, sys, glob, numpy as np, pandas as pd
from scipy.signal import butter, iirnotch, filtfilt, spectrogram
sys.path.insert(0, (WORK[:-1]))
from solitary_lfp import read_fields, device_signals, FS
SRC=(RAW_LFP + 'member_perturbation')
OUT=(LFP + 'member_change')
BANDS=[('theta1',4,8),('theta2',8,12),('beta1',18,24),('beta2',24,32),('gamma1',35,50),('gamma2',70,90)]
RIGMAP={'1_2_3_7':{1:'A3',2:'A1',3:'A2'},'4_5_0_8':{1:'A4',2:'A5',3:'A6'}}
PAT=re.compile(r'Day(\d+)_.*?_(Baseline|Foraging)_(?:(?:(Active|Passive)_)?Trial(\d+)_CBRAIN_[\d_]+|Individual_#(\d)(?:_CBRAIN_[\d_]+)?)_file(\d+)\.txt$')
SUB={None:0,'Active':100,'Passive':200}
bh,ah=butter(4,1/(FS/2),'highpass'); bl,al=butter(4,300/(FS/2),'lowpass'); bn,an=iirnotch(60,30,FS)
prep=lambda x: filtfilt(bn,an,filtfilt(bl,al,filtfilt(bh,ah,x.astype(np.float64))))
groups={}
for d in sorted(os.listdir(SRC)):
    if not d.startswith('Day'): continue
    rig=d.split('CBRAIN_')[-1]
    for f in sorted(os.listdir(os.path.join(SRC,d))):
        if '_Food_' in f or f.endswith('_para.txt'): continue
        m=PAT.search(f)
        if m:
            day,seg,sub,tri,ind,part=m.groups()
            groups.setdefault((int(day),rig,seg,int(tri)+SUB[sub] if tri else -int(ind)-1),[]).append((int(part),os.path.join(SRC,d,f)))
have=set()
for f in glob.glob(os.path.join(OUT,'spec_*.npz')):
    bn_=os.path.basename(f); m=re.match(r'spec_Day(\d+)_([\d_]+)_p\d+\.npz', bn_)
    for k in np.load(f).files:
        if not k.endswith('_spec'): continue
        if m:
            seg,t,mo,_=k.split('_'); have.add((int(m.group(1)),m.group(2),seg,int(t[1:])))
        else:                                     # 채움 파일: Day<N>_<rig>_<seg>_T..
            p=k.split('_'); have.add((int(p[0][3:]), '_'.join(p[1:5]), p[5], int(p[6][1:])))
b=pd.read_csv(os.path.join(OUT,'bandpower_whole.csv'))
need=sorted({(r.day,r.rig,r.segment,r.trial) for r in b.itertuples()}-have)
print('채울 구간 %d'%len(need), flush=True)
store={}
for k in need:
    if k not in groups: print('원시 없음', k, flush=True); continue
    day,rig,seg,tri=k
    F=np.vstack([read_fields(p)[0] for _,p in sorted(groups[k])])
    for dev in range(3):
        lfp,_d,_i=device_signals(F,dev); specs=[]; b8s=[]
        for c in range(3):
            x=prep(lfp[c])
            f_,t_,S=spectrogram(x,fs=FS,window='hamming',nperseg=256,noverlap=128,detrend=False,scaling='density',mode='psd')
            e=np.searchsorted(f_,np.arange(0.5,121,1.0))
            Sb=np.add.reduceat(S,e[:-1],axis=0)/np.maximum(np.diff(e),1)[:,None]
            b8=np.stack([S[(f_>=lo)&(f_<hi)].mean(0) for _,lo,hi in BANDS])
            if seg=='Baseline':
                m_=(Sb.shape[1]//8)*8
                Sb=Sb[:,:m_].reshape(Sb.shape[0],-1,8).mean(2); b8=b8[:,:m_].reshape(b8.shape[0],-1,8).mean(2)
            specs.append(np.log10(np.maximum(Sb,1e-20)).astype(np.float16)); b8s.append(b8.astype(np.float32))
        mo=RIGMAP[rig][dev+1]
        store['Day%d_%s_%s_T%d_%s_spec'%(day,rig,seg,tri,mo)]=np.stack(specs)
        store['Day%d_%s_%s_T%d_%s_band'%(day,rig,seg,tri,mo)]=np.stack(b8s)
    print('ok',k,flush=True); del F
if store:
    i=1
    while os.path.exists(os.path.join(OUT,'spec_fill_%d.npz'%i)): i+=1
    np.savez_compressed(os.path.join(OUT,'spec_fill_%d.npz'%i),**store)
    print('저장 spec_fill_%d.npz  %d키'%(i,len(store)),flush=True)
print('DONE',flush=True)
