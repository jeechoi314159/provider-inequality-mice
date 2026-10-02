# -*- coding: utf-8 -*-
"""진입 반응을 시행별 AUC 로 요약하고 (1) 조건에 따른 차이, (2) 다른 개체의 '물러남'
(뒤이은 진입 지연 · 끝내 안 들어간 개체 수) 과의 관계를 본다.
AUC = 기저(−2…−1 s) 대비 변화의 0…2 s 적분 (z·s)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, pickle
d = pickle.load(open('allmap_traces.pkl', 'rb'))
meta, X, FS = d['meta'].copy(), d['X'], d['FS']
REG = ['PFC', 'NAc', 'BLA']; BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
KEYS = [(g, b) for g in REG for b in BANDS]
PRE, POST = 2.0, 3.0
W = np.arange(-int(PRE * FS), int(POST * FS) + 1); T = W / FS
BASE = (T >= -2.0) & (T <= -1.0)
AUCW = (T >= 0.0) & (T < 2.0)

beh = pd.read_csv((INTER + 'follower_retrieval_latency.csv'))
A = beh[beh.Group == 'A'][['Mouse_ID', 'Trial', 'EntryLatency', 'Entered', 'Worked', 'RetrievalLatency']]
# 시행별 진입 순서와 '남들의 물러남'
rows = []
for t, g in A.groupby('Trial'):
    ent = g[(g.Entered == 1) & g.EntryLatency.notna()].sort_values('EntryLatency')
    n_never = int((g.Entered == 0).sum())
    for k, (_, r) in enumerate(ent.iterrows()):
        nxt = ent.EntryLatency.values[k + 1] if k + 1 < len(ent) else np.nan
        rows.append(dict(trial=t, Mouse_ID=r.Mouse_ID, entry=r.EntryLatency, rank=k + 1,
                         n_before=k, n_after=len(ent) - k - 1, n_never=n_never,
                         next_gap=nxt - r.EntryLatency if np.isfinite(nxt) else np.nan,
                         worked=int(r.Worked)))
ORD = pd.DataFrame(rows)

ev = meta.EntryLatency.values
ok = (meta.Entered == 1).values & np.isfinite(ev) & (ev >= PRE) & (ev + POST <= meta.n_tot.values / FS)
S = meta[ok].copy()
IDX = S.start.values[:, None] + np.round(ev[ok] * FS).astype(int)[:, None] + W[None, :]
for (g, b) in KEYS:
    V = X[(g, b)][IDX]; D = V - V[:, BASE].mean(1, keepdims=True)
    S['%s_%s' % (g, b)] = D[:, AUCW].sum(1) / FS          # z·s
S = S.merge(ORD, on=['trial', 'Mouse_ID'], how='left', suffixes=('', '_b'))
S['is_starter_entry'] = S['rank'] == 1
S['P'] = S.Mouse_ID == 'A2'
S['log_next'] = np.log10(S.next_gap.where(S.next_gap > 0))
S['pos'] = (S.trial - 1) / (S.trial.max() - 1)
S.to_csv('entry_auc_trials.csv', index=False)
FE = ['%s_%s' % (g, b) for g, b in KEYS]
print('시행 수 %d, 개체 %d' % (len(S), S.Mouse_ID.nunique()))
print('진입 순위 분포', S['rank'].value_counts().sort_index().to_dict())
print('next_gap 중앙값 %.1f s (유효 %d), 끝내 안 들어간 개체 수 분포 %s'
      % (S.next_gap.median(), int(S.next_gap.notna().sum()), S.n_never.value_counts().sort_index().to_dict()))
print('\n조건별 AUC (z·s) — 진입에서 유의했던 여섯 특징')
KEY6 = ['PFC_ltheta', 'PFC_lbeta', 'NAc_lbeta', 'BLA_lbeta', 'BLA_hbeta', 'BLA_lgamma']
def show(lab, grp):
    t = S.groupby(grp)[KEY6].mean().round(3)
    t['n'] = S.groupby(grp).size()
    print('\n[%s]' % lab); print(t.to_string())
show('제공자 여부', 'P')
show('먼저 들어갔나(순위 1)', 'is_starter_entry')
show('그 시행에서 회수했나', 'worked')
S['n_before_c'] = np.minimum(S.n_before, 2)
show('이미 들어가 있던 개체 수(0/1/2+)', 'n_before_c')
