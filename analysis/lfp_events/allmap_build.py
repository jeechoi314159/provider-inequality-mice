# -*- coding: utf-8 -*-
"""코호트 A 의 시행 전체 연속 대역파워 궤적을 만든다(시행 시작 = 0 s, 8 Hz).
Start_act(시행 시작 → 본인/작업자 진입) + Act_ride(그 이후) 를 이어 붙인다.
사건 시각(초, 시행 시작 기준): 본인 진입, 최초 진입자(starter) 진입, 회수."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, pickle, os

P = (LFP + 'powerspect_extract/')
REG = ['PFC', 'NAc', 'BLA']
BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
FS = 8
M = pd.read_csv('ps_entries_mapped.csv')
beh = pd.read_csv((INTER + 'follower_retrieval_latency.csv'))
A = beh[beh.Group == 'A']
RET = A.dropna(subset=['RetrievalLatency']).set_index('Trial').RetrievalLatency.to_dict()
WORKER = A[A.Worked == 1].set_index('Trial').Mouse_ID.to_dict()
STARTER = A[A.TrialRole == 'Starter'].set_index('Trial').Mouse_ID.to_dict()

# ---- 궤적 이어 붙이기 -------------------------------------------------------
LEN = {}
for role in ['worker', 'participant', 'freerider']:
    d = np.load(P + f'ps_Start_act_{role}.npz'); e = np.load(P + f'ps_Act_ride_{role}.npz')
    LEN[role] = (np.isfinite(d['PFC_ltheta']).sum(1), np.isfinite(e['PFC_ltheta']).sum(1))
n1 = np.array([LEN[r.ps_role][0][int(r.ps_j)] for _, r in M.iterrows()])
n2 = np.array([LEN[r.ps_role][1][int(r.ps_j)] for _, r in M.iterrows()])
M = M.assign(n_start=n1, n_ride=n2, n_tot=n1 + n2)
err = np.abs(M.n_start / FS - M.len_s)
print('Start_act 길이와 len_s 불일치: median %.3f s, q99 %.3f s' % (err.median(), err.quantile(.99)))

off = np.concatenate(([0], np.cumsum(M.n_tot.values)))
TOT = int(off[-1]); print('mouse-trials %d, 총 bin %d (%.1f MB/map)' % (len(M), TOT, TOT * 4 / 1e6))
X = {}
for g in REG:
    for b in BANDS:
        buf = np.full(TOT, np.nan, np.float32)
        for role in ['worker', 'participant', 'freerider']:
            d = np.load(P + f'ps_Start_act_{role}.npz')[f'{g}_{b}']
            e = np.load(P + f'ps_Act_ride_{role}.npz')[f'{g}_{b}']
            sel = np.flatnonzero((M.ps_role == role).values)
            for i in sel:
                j = int(M.ps_j.values[i]); a, c = int(M.n_start.values[i]), int(M.n_ride.values[i])
                buf[off[i]:off[i] + a] = d[j, :a]
                buf[off[i] + a:off[i] + a + c] = e[j, :c]
        X[(g, b)] = buf
    print(' ', g, 'done')

meta = M[['ps_role', 'ps_j', 'trial', 'day', 'Mouse_ID', 'EntryLatency', 'StarterLat',
          'Entered', 'Worked', 'n_start', 'n_ride', 'n_tot']].copy()
meta['start'] = off[:-1]
meta['retrieval'] = meta.trial.map(RET)
meta['starter_id'] = meta.trial.map(STARTER)
meta['worker_id'] = meta.trial.map(WORKER)
meta['is_starter'] = meta.Mouse_ID == meta.starter_id
meta['is_worker'] = meta.Mouse_ID == meta.worker_id
meta['starter_is_P'] = meta.starter_id == 'A2'
pickle.dump(dict(meta=meta, X=X, FS=FS, off=off), open('allmap_traces.pkl', 'wb'))

cov = lambda ev, lo, hi: int(((meta[ev] >= -lo) & (meta[ev] + hi <= meta.n_tot / FS)).sum())
print('\n표본 수(창 확보 기준)')
for ev, lab in (('EntryLatency', '본인 진입'), ('retrieval', '회수'), ('StarterLat', '동료 진입')):
    print('  %-10s  ±2s %3d   ±3s %3d   −3/+4 %3d   ±4s %3d'
          % (lab, cov(ev, -2, 2), cov(ev, -3, 3), cov(ev, -3, 4), cov(ev, -4, 4)))
print('\n관찰자(=starter 아님) 행 수 %d' % int((~meta.is_starter).sum()))
print('회수자 %d / 비회수자 %d' % (int(meta.is_worker.sum()), int((~meta.is_worker).sum())))
