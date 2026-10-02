# -*- coding: utf-8 -*-
"""구성 변경 단계(Set 4 Day21–45) CBRAIN 원시 → 시행 × 개체 대역 파워.

드라이브에서 스테이징된 파일만 처리하고, 결과는 누적 CSV 한 장에 append 한다.
전처리는 투고본 Methods 와 같음: 1 Hz 고역 · 300 Hz 저역 · 60 Hz 대역저지.
대역: theta1 4–8, theta2 8–12, beta1 18–24, beta2 24–32, gamma1 35–50, gamma2 70–90 Hz.
창 전체(채집 구간)의 절대 파워를 Welch 로 구한다 (시행 내 z 는 사건 시각이 있어야 하므로 보류).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, re, sys, glob, csv
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, welch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solitary_lfp import read_fields, device_signals, FS

UP  = (RAW_LFP[:-1])
OUT = (WORK + 'group_cbrain_bandpower.csv')
REGION = ['BLA', 'NAc', 'PFC']
BANDS = [('theta1', 4, 8), ('theta2', 8, 12), ('beta1', 18, 24),
         ('beta2', 24, 32), ('gamma1', 35, 50), ('gamma2', 70, 90)]
RIG = {'1_2_3_7': ['A1', 'A2', 'A3'], '4_5_0_8': ['A4', 'A5', 'A6']}
PAT = re.compile(r'Day(\d+)_(.+?)_(Baseline|Foraging)_Trial(\d+)_CBRAIN_([\d_]+)_file(\d+)\.txt$')

bh, ah = butter(4, 1/(FS/2), 'highpass')
bl, al = butter(4, 300/(FS/2), 'lowpass')
bn, an = iirnotch(60, 30, FS)


def prep(x):
    y = filtfilt(bh, ah, x.astype(np.float64))
    y = filtfilt(bl, al, y)
    return filtfilt(bn, an, y)


def bandpower(x):
    f, P = welch(x, fs=FS, nperseg=FS*2, noverlap=FS, detrend='constant')
    return [float(np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))
            for _, lo, hi in BANDS]


hdr = (['day', 'phase', 'segment', 'trial', 'rig', 'device', 'mouse', 'n', 'seconds',
        'frozen', 'sd_ch1', 'sd_ch2', 'sd_ch3']
       + ['%s_%s' % (g, b) for g in REGION for b, _, _ in BANDS])
new = not os.path.exists(OUT)
fh = open(OUT, 'a', newline=''); wr = csv.writer(fh)
if new:
    wr.writerow(hdr)
done = set()
if not new:
    import pandas as pd
    p = pd.read_csv(OUT)
    done = set(map(tuple, p[['day', 'segment', 'trial', 'rig']].drop_duplicates().values))

groups = {}
for p in glob.glob(os.path.join(UP, '*', '*', '*.txt')):
    m = PAT.search(os.path.basename(p))
    if m:
        day, phase, seg, tri, rig, part = m.groups()
        groups.setdefault((int(day), phase, seg, int(tri), rig), []).append((int(part), p))

keys = sorted(groups)
print('스테이징된 시행 %d' % len(keys), flush=True)
for k in keys:
    day, phase, seg, tri, rig = k
    if (day, seg, tri, rig) in done:
        continue
    try:
        F = np.vstack([read_fields(p)[0] for _, p in sorted(groups[k])])
        for dev in range(3):
            lfp, ttl, idx = device_signals(F, dev)
            frozen = float((np.diff(idx) < 0).mean())
            row = [day, phase, seg, tri, rig, dev+1, RIG[rig][dev], len(idx),
                   round(len(idx)/FS, 1), round(frozen, 4)]
            row += [round(float(lfp[c].std()), 5) for c in range(3)]
            for c in range(3):
                row += ['%.6g' % v for v in bandpower(prep(lfp[c]))]
            wr.writerow(row)
        fh.flush()
        print('ok %s  %.0f s' % (str(k), len(F)/FS), flush=True)
    except Exception as e:
        print('FAIL %s : %s' % (k, e), flush=True)
fh.close()
print('DONE', flush=True)
