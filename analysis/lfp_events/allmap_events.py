# -*- coding: utf-8 -*-
"""사건 정렬 대역파워 변화의 전면 검정 (코호트 A).
창 −2…+3 s, 기저 −2…−1 s. 시행별 one-sample t, 시간축 군집 질량,
귀무분포는 '사건 시각을 시행 안에서 무작위로 옮기기'(1000회).
family-wise 는 행(all/P/others)마다 3영역 × 6대역 = 18개 지도의 최대 군집 질량."""
import numpy as np, pandas as pd, pickle, sys
from scipy import stats
from scipy.ndimage import label

d = pickle.load(open('allmap_traces.pkl', 'rb'))
meta, X, FS = d['meta'], d['X'], d['FS']
REG = ['PFC', 'NAc', 'BLA']
BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
PRE, POST = 2.0, 3.0
BL = (-2.0, -1.0)
W = np.arange(-int(PRE * FS), int(POST * FS) + 1)
T = W / FS
BASE = (T >= BL[0]) & (T <= BL[1])
NPERM = 1000
CF = 0.05
rng = np.random.default_rng(20260930)

ARRAYS = [
    ('own_entry', 'the mouse enters', (meta.Entered == 1).values, 'EntryLatency', True),
    ('retrieval', 'the snack is taken', np.ones(len(meta), bool), 'retrieval', True),
    ('cagemate_all', 'a cagemate enters (any)', (~meta.is_starter).values, 'StarterLat', True),
    ('cagemate_P', 'the provider enters', (~meta.is_starter & meta.starter_is_P).values, 'StarterLat', True),
    ('cagemate_NP', 'another mouse enters', (~meta.is_starter & ~meta.starter_is_P).values, 'StarterLat', True),
    ('retrieval_by_self', 'the mouse itself takes the snack', meta.is_worker.values, 'retrieval', False),
    ('retrieval_by_other', 'another mouse takes the snack', (~meta.is_worker).values, 'retrieval', False),
    ('cagemate_nofollow', 'a cagemate enters, observer stays out ≥3 s',
     (~meta.is_starter & ~((meta.EntryLatency - meta.StarterLat) < POST)).values, 'StarterLat', False),
]
ROWS = [('All mice', lambda m: np.ones(len(m), bool)),
        ('Provider', lambda m: (m.Mouse_ID == 'A2').values),
        ('Others', lambda m: (m.Mouse_ID != 'A2').values)]


def windows(ev_bin, start):
    return start[:, None] + ev_bin[:, None] + W[None, :]


def stat_map(IDX, buf):
    V = buf[IDX]
    D = V - V[:, BASE].mean(1, keepdims=True)
    n = len(D)
    m = D.mean(0); s = D.std(0, ddof=1)
    return m / (s / np.sqrt(n)) , m, s / np.sqrt(n)


def clusters(tm, thr):
    out = []
    for sgn in (1, -1):
        lab, k = label(sgn * tm > thr)
        for c in range(1, k + 1):
            mk = lab == c
            out.append((float(np.abs(tm[mk]).sum()), sgn, mk))
    return out


rows_cl, rows_tr = [], []
for aname, alab, asel, evcol, is_main in ARRAYS:
    ev = meta[evcol].values
    ok = asel & np.isfinite(ev) & (ev >= PRE) & (ev + POST <= meta.n_tot.values / FS)
    for rname, rfun in ROWS:
        sel = ok & rfun(meta)
        if aname == 'cagemate_P' and rname == 'Provider':
            continue                      # 제공자는 자기 진입을 관찰할 수 없음
        S = meta[sel]
        n = len(S)
        if n < 25:
            print('skip', aname, rname, n); continue
        start = S.start.values.astype(np.int64)
        ntot = S.n_tot.values.astype(np.int64)
        evb = np.round(ev[sel] * FS).astype(np.int64)
        lo = int(PRE * FS); hi = ntot - int(POST * FS) - 1
        IDX = windows(evb, start)
        thr = stats.t.ppf(1 - CF / 2, n - 1)
        obs = {}
        for g in REG:
            for b in BANDS:
                tm, mn, se = stat_map(IDX, X[(g, b)])
                obs[(g, b)] = tm
                for j, tt in enumerate(T):
                    rows_tr.append(dict(array=aname, row=rname, region=g, band=b, t=float(tt),
                                        mean=float(mn[j]), sem=float(se[j]), n=n))
        null = np.zeros(NPERM)
        for p in range(NPERM):
            r = rng.random(n)
            eb = lo + (r * np.maximum(hi - lo, 1)).astype(np.int64)
            far = np.abs(eb - evb) >= 3 * FS
            eb = np.where(far | (hi - lo < 8 * FS), eb, np.minimum(evb + 3 * FS, hi))
            I2 = windows(eb, start)
            mx = 0.0
            for g in REG:
                for b in BANDS:
                    tm, _, _ = stat_map(I2, X[(g, b)])
                    cl = clusters(tm, thr)
                    if cl:
                        mx = max(mx, max(c[0] for c in cl))
            null[p] = mx
        for g in REG:
            for b in BANDS:
                for mass, sgn, mk in clusters(obs[(g, b)], thr):
                    tt = T[mk]
                    rows_cl.append(dict(array=aname, row=rname, region=g, band=b,
                                        t_start=float(tt.min()), t_end=float(tt.max() + 1 / FS),
                                        sign='+' if sgn > 0 else '−', mass=round(mass, 1),
                                        peak_t=float(T[mk][np.argmax(np.abs(obs[(g, b)][mk]))]),
                                        peak_stat=float(obs[(g, b)][mk][np.argmax(np.abs(obs[(g, b)][mk]))]),
                                        P_family=float((null >= mass).mean()), n=n, main=is_main))
        print('%-20s %-9s n=%4d  thr=%.2f  군집 %d개, P<0.05 %d개'
              % (aname, rname, n, thr, sum(1 for r in rows_cl if r['array'] == aname and r['row'] == rname),
                 sum(1 for r in rows_cl if r['array'] == aname and r['row'] == rname and r['P_family'] < 0.05)))
        sys.stdout.flush()

CL = pd.DataFrame(rows_cl); TRm = pd.DataFrame(rows_tr)
CL.to_csv('allmap_clusters.csv', index=False)
TRm.to_csv('allmap_means.csv', index=False)
print('\n유의 군집 (P_family < 0.05), 주요 5개 배열')
S = CL[(CL.P_family < 0.05) & CL.main].sort_values(['array', 'row', 'P_family'])
print(S[['array', 'row', 'region', 'band', 't_start', 't_end', 'sign', 'peak_t', 'P_family', 'n']]
      .to_string(index=False))
