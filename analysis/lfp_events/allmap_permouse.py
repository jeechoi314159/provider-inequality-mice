# -*- coding: utf-8 -*-
"""본인 진입에서 유의했던 대역의 개체별 효과 크기(기저 대비 평균 변화, z 단위).
제공자(A2)가 유의하지 않은 것이 검정력 문제인지 실제 차이인지 눈으로 볼 자료."""
import numpy as np, pandas as pd, pickle
d = pickle.load(open('allmap_traces.pkl', 'rb')); meta, X, FS = d['meta'], d['X'], d['FS']
CL = pd.read_csv('allmap_clusters.csv')
key = CL[(CL.array == 'own_entry') & (CL.row == 'All mice') & (CL.P_family < 0.05)]
PRE, POST = 2.0, 3.0
W = np.arange(-int(PRE * FS), int(POST * FS) + 1); T = W / FS
BASE = (T >= -2.0) & (T <= -1.0)
ev = meta.EntryLatency.values
ok = (meta.Entered == 1).values & np.isfinite(ev) & (ev >= PRE) & (ev + POST <= meta.n_tot.values / FS)
S = meta[ok]; IDX = S.start.values[:, None] + np.round(ev[ok] * FS).astype(int)[:, None] + W[None, :]
rows = []
for _, c in key.iterrows():
    V = X[(c.region, c.band)][IDX]
    D = V - V[:, BASE].mean(1, keepdims=True)
    w = (T >= c.t_start) & (T < c.t_end)
    v = D[:, w].mean(1)
    r = dict(region=c.region, band=c.band, window='%.2f–%.2f s' % (c.t_start, c.t_end), sign=c.sign)
    for u in ['A2'] + [m for m in sorted(S.Mouse_ID.unique()) if m != 'A2']:
        i = (S.Mouse_ID.values == u)
        r[u] = round(float(np.nanmean(v[i])), 3); r['n_' + u] = int(i.sum())
    o = [r[m] for m in ['A1', 'A3', 'A4', 'A5', 'A6']]
    r['others_mean'] = round(float(np.mean(o)), 3)
    r['A2_rank'] = int(sum(abs(r['A2']) > abs(x) for x in o))
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv('allmap_own_entry_per_mouse.csv', index=False)
cols = ['region', 'band', 'window', 'sign', 'A2', 'A1', 'A3', 'A4', 'A5', 'A6', 'others_mean']
print(R[cols].to_string(index=False))
print('\n시행 수: A2 %d, 나머지 %s' % (R.n_A2.iloc[0], {m: int(R['n_' + m].iloc[0]) for m in ['A1','A3','A4','A5','A6']}))
