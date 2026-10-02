# -*- coding: utf-8 -*-
"""사건 직전 기저(±1 s)가 아니라 '진입 전 5 s' 를 공통 기저로 쓰는 창 분석.
기존 solitary-vs-group 보고서의 5 s 창 결과와 같은 규격이므로, 사건 정렬 검정의 귀무와
그 보고서의 θ 상승이 모순인지 아닌지를 가린다."""
import numpy as np, pandas as pd, pickle
from scipy import stats
d = pickle.load(open('allmap_traces.pkl', 'rb')); meta, X, FS = d['meta'], d['X'], d['FS']
REG = ['PFC', 'NAc', 'BLA']; BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
NPERM = 5000
rng = np.random.default_rng(1)


def seg(ev, lo, hi, sel):
    """각 시행에서 ev+lo … ev+hi 초 구간 평균. 자료가 없으면 NaN."""
    out = np.full((sel.sum(), len(REG) * len(BANDS)), np.nan)
    S = meta[sel]; st = S.start.values; nt = S.n_tot.values; e = ev[sel]
    a = np.round((e + lo) * FS).astype(int); b = np.round((e + hi) * FS).astype(int)
    for k, (g, bd) in enumerate([(g, bd) for g in REG for bd in BANDS]):
        buf = X[(g, bd)]
        for i in range(len(S)):
            i0, i1 = a[i], b[i]
            if i0 < 0 or i1 > nt[i] or i1 - i0 < 8:
                continue
            out[i, k] = np.nanmean(buf[st[i] + i0:st[i] + i1])
    return out


ent = meta.EntryLatency.values; ret = meta.retrieval.values
sel = (meta.Entered == 1).values & np.isfinite(ent) & np.isfinite(ret) & (ent >= 5.0)
BASE = seg(ent, -5.0, 0.0, sel)
ENTRY = seg(ent, 0.0, 5.0, sel)
RETR = seg(ret, -2.5, 2.5, sel)
S = meta[sel]; g = S.Mouse_ID.values
cols = ['%s_%s' % (r, b) for r in REG for b in BANDS]
rows = []
for name, Wm in (('entry (0 to +5 s)', ENTRY), ('retrieval (−2.5 to +2.5 s)', RETR)):
    D = Wm - BASE
    for k, c in enumerate(cols):
        v = D[:, k]; ok = np.isfinite(v)
        if ok.sum() < 30:
            continue
        vv, gg = v[ok], g[ok]
        m = float(np.mean(vv))
        idx = [np.where(gg == u)[0] for u in np.unique(gg)]
        null = np.empty(NPERM)
        for p in range(NPERM):
            s2 = vv.copy()
            for i in idx:
                s2[i] = vv[i] * rng.choice([-1, 1])       # 개체 단위 부호 뒤집기
            null[p] = np.mean(s2)
        rows.append(dict(window=name, feature=c, region=c.split('_')[0], band=c.split('_')[1],
                         delta=round(m, 4), n=int(ok.sum()), mice=len(idx),
                         P=float((np.abs(null) >= abs(m)).mean())))
R = pd.DataFrame(rows)
R['q'] = R.groupby('window').P.transform(lambda p: np.minimum(1, p * len(p) / stats.rankdata(p)))
R.to_csv('allmap_window5s.csv', index=False)
pd.set_option('display.width', 200)
for w in R.window.unique():
    t = R[R.window == w].sort_values('P')
    print('\n==', w, ' (기저: 진입 전 5 s), n=%d 시행, %d 마리' % (t.n.max(), t.mice.max()))
    print(t[['region', 'band', 'delta', 'P', 'q']].head(8).to_string(index=False))
