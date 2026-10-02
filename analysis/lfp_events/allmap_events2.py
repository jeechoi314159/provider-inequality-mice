# -*- coding: utf-8 -*-
"""회수 시점 보완: 추출 구간이 대체로 스낵 탈취 직후 끝나므로(worker 기준 end−retrieval 중앙값 −0.6 s)
회수 배열은 −3…+0.5 s (기저 −3…−2 s) 로 다시 검정한다. 그리고 본인 진입 군집의 개체별 효과 크기."""
import numpy as np, pandas as pd, pickle, sys
from scipy import stats
from scipy.ndimage import label

d = pickle.load(open('allmap_traces.pkl', 'rb'))
meta, X, FS = d['meta'], d['X'], d['FS']
REG = ['PFC', 'NAc', 'BLA']; BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
PRE, POST, BL = 3.0, 0.5, (-3.0, -2.0)
W = np.arange(-int(PRE * FS), int(POST * FS) + 1); T = W / FS
BASE = (T >= BL[0]) & (T <= BL[1])
NPERM, CF = 1000, 0.05
rng = np.random.default_rng(20260930)
ARRAYS = [('retrieval_pre', (np.ones(len(meta), bool)), 'retrieval', True),
          ('retrieval_pre_by_self', meta.is_worker.values, 'retrieval', False),
          ('retrieval_pre_by_other', (~meta.is_worker).values, 'retrieval', False)]
ROWS = [('All mice', lambda m: np.ones(len(m), bool)),
        ('Provider', lambda m: (m.Mouse_ID == 'A2').values),
        ('Others', lambda m: (m.Mouse_ID != 'A2').values)]
win = lambda eb, st: st[:, None] + eb[:, None] + W[None, :]


def stat_map(IDX, buf):
    V = buf[IDX]; D = V - V[:, BASE].mean(1, keepdims=True)
    m = D.mean(0); s = D.std(0, ddof=1) / np.sqrt(len(D))
    return m / s, m, s


def clusters(tm, thr):
    out = []
    for sgn in (1, -1):
        lab, k = label(sgn * tm > thr)
        for c in range(1, k + 1):
            out.append((float(np.abs(tm[lab == c]).sum()), sgn, lab == c))
    return out


rc, rt = [], []
for aname, asel, evcol, is_main in ARRAYS:
    ev = meta[evcol].values
    ok = asel & np.isfinite(ev) & (ev >= PRE) & (ev + POST <= meta.n_tot.values / FS)
    for rname, rfun in ROWS:
        sel = ok & rfun(meta); S = meta[sel]; n = len(S)
        if n < 25:
            print('skip', aname, rname, n); continue
        start = S.start.values.astype(np.int64); ntot = S.n_tot.values.astype(np.int64)
        evb = np.round(ev[sel] * FS).astype(np.int64)
        lo = int(PRE * FS); hi = ntot - int(POST * FS) - 1
        IDX = win(evb, start); thr = stats.t.ppf(1 - CF / 2, n - 1)
        obs = {}
        for g in REG:
            for b in BANDS:
                tm, mn, se = stat_map(IDX, X[(g, b)]); obs[(g, b)] = tm
                for j, tt in enumerate(T):
                    rt.append(dict(array=aname, row=rname, region=g, band=b, t=float(tt),
                                   mean=float(mn[j]), sem=float(se[j]), n=n))
        null = np.zeros(NPERM)
        for p in range(NPERM):
            eb = lo + (rng.random(n) * np.maximum(hi - lo, 1)).astype(np.int64)
            far = np.abs(eb - evb) >= 3 * FS
            eb = np.where(far | (hi - lo < 8 * FS), eb, np.minimum(evb + 3 * FS, hi))
            I2 = win(eb, start); mx = 0.0
            for g in REG:
                for b in BANDS:
                    cl = clusters(stat_map(I2, X[(g, b)])[0], thr)
                    if cl:
                        mx = max(mx, max(c[0] for c in cl))
            null[p] = mx
        for g in REG:
            for b in BANDS:
                for mass, sgn, mk in clusters(obs[(g, b)], thr):
                    tt = T[mk]; k = np.argmax(np.abs(obs[(g, b)][mk]))
                    rc.append(dict(array=aname, row=rname, region=g, band=b,
                                   t_start=float(tt.min()), t_end=float(tt.max() + 1 / FS),
                                   sign='+' if sgn > 0 else '−', mass=round(mass, 1),
                                   peak_t=float(tt[k]), peak_stat=float(obs[(g, b)][mk][k]),
                                   P_family=float((null >= mass).mean()), n=n, main=is_main))
        print('%-24s %-9s n=%4d  군집 %d, P<0.05 %d' % (aname, rname, n,
              sum(1 for r in rc if r['array'] == aname and r['row'] == rname),
              sum(1 for r in rc if r['array'] == aname and r['row'] == rname and r['P_family'] < 0.05)))
        sys.stdout.flush()
pd.DataFrame(rc).to_csv('allmap_clusters_pre.csv', index=False)
pd.DataFrame(rt).to_csv('allmap_means_pre.csv', index=False)
S = pd.DataFrame(rc)
print('\n유의 군집:')
print(S[S.P_family < 0.05][['array', 'row', 'region', 'band', 't_start', 't_end', 'sign', 'P_family', 'n']]
      .to_string(index=False) if (S.P_family < 0.05).any() else '  없음')
