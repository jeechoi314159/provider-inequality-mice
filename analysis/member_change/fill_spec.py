import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, re, sys, numpy as np
from scipy.signal import butter, iirnotch, filtfilt, spectrogram
sys.path.insert(0, (WORK[:-1]))
from solitary_lfp import read_fields, device_signals, FS
SRC=(RAW_LFP + 'member_perturbation')
OUT=(LFP + 'member_change')
BANDS=[('theta1',4,8),('theta2',8,12),('beta1',18,24),('beta2',24,32),('gamma1',35,50),('gamma2',70,90)]
RIGMAP={'1_2_3_7':{1:'A3',2:'A1',3:'A2'},'4_5_0_8':{1:'A4',2:'A5',3:'A6'}}
NEED=[(22,'1_2_3_7','Baseline',-4),(22,'1_2_3_7','Baseline',-3),(22,'1_2_3_7','Baseline',-2),
      (22,'1_2_3_7','Baseline',1),(23,'4_5_0_8','Baseline',1),(23,'4_5_0_8','Baseline',2),
      (23,'4_5_0_8','Baseline',3),(23,'4_5_0_8','Baseline',4),(23,'4_5_0_8','Baseline',5)]
PAT=re.compile(r'Day(\d+)_Absence_1st_[Ww]orker_(Baseline|Foraging)_'
               r'(?:Trial(\d+)_CBRAIN_([\d_]+)|Individual_#(\d))_file(\d+)\.txt$')
bh,ah=butter(4,1/(FS/2),'highpass'); bl,al=butter(4,300/(FS/2),'lowpass'); bn,an=iirnotch(60,30,FS)
prep=lambda x: filtfilt(bn,an,filtfilt(bl,al,filtfilt(bh,ah,x.astype(np.float64))))
groups={}
for d in sorted(os.listdir(SRC)):
    if not d.startswith('Day'): continue
    rig=d.split('CBRAIN_')[-1]
    for f in sorted(os.listdir(os.path.join(SRC,d))):
        m=PAT.search(f)
        if m:
            day,seg,tri,_rg,ind,part=m.groups()
            groups.setdefault((int(day),rig,seg,int(tri) if tri else -int(ind)-1),[]).append((int(part),os.path.join(SRC,d,f)))
store={}
for k in NEED:
    F=np.vstack([read_fields(p)[0] for _,p in sorted(groups[k])])
    day,rig,seg,tri=k
    for dev in range(3):
        lfp,_dg,_i=device_signals(F,dev); specs=[]; b8s=[]
        for c in range(3):
            x=prep(lfp[c])
            f_,t_,S=spectrogram(x,fs=FS,window='hamming',nperseg=256,noverlap=128,detrend=False,scaling='density',mode='psd')
            e=np.searchsorted(f_,np.arange(0.5,121,1.0))
            Sb=np.add.reduceat(S,e[:-1],axis=0)/np.maximum(np.diff(e),1)[:,None]
            b8=np.stack([S[(f_>=lo)&(f_<hi)].mean(0) for _,lo,hi in BANDS])
            m=(Sb.shape[1]//8)*8
            Sb=Sb[:,:m].reshape(Sb.shape[0],-1,8).mean(2); b8=b8[:,:m].reshape(b8.shape[0],-1,8).mean(2)
            specs.append(np.log10(np.maximum(Sb,1e-20)).astype(np.float16)); b8s.append(b8.astype(np.float32))
        mo=RIGMAP[rig][dev+1]
        store['Day%d_%s_%s_T%d_%s_spec'%(day,rig,seg,tri,mo)]=np.stack(specs)
        store['Day%d_%s_%s_T%d_%s_band'%(day,rig,seg,tri,mo)]=np.stack(b8s)
    print('ok',k,flush=True); del F
np.savez_compressed(os.path.join(OUT,'spec_fill_baselines.npz'),**store)
print('DONE %d키'%len(store),flush=True)
