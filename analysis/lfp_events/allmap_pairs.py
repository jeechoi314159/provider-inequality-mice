# -*- coding: utf-8 -*-
"""행동으로 정의된 starter–follower 쌍에서만 본 동료 진입 반응.
관찰자 전체 평균은 '보고 있지 않은' 시행을 섞으므로, 그 시행에서 실제로 뒤따라 들어간
개체(추종자)와 끝내 안 들어간 개체(미진입)를 갈라 따로 본다.
추종자의 창은 자기 진입 0.25 s 전에서 끊는다(자기 진입 반응 오염 방지). 그 절단 마스크는
귀무 순열에서도 상대시간 그대로 유지하므로 bin 별 표본 수 곡선이 동일하다."""
import numpy as np, pandas as pd, pickle, sys
from scipy import stats
from scipy.ndimage import label

d = pickle.load(open('allmap_traces.pkl', 'rb'))
meta, X, FS = d['meta'].copy(), d['X'], d['FS']
REG = ['PFC', 'NAc', 'BLA']; BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
PRE, POST = 2.0, 3.0
W = np.arange(-int(PRE * FS), int(POST * FS) + 1); T = W / FS
BASE = (T >= -2.0) & (T <= -1.0)
NPERM, CF, MINN = 1000, 0.05, 20
rng = np.random.default_rng(20260930)

meta['L'] = meta.EntryLatency - meta.StarterLat
obs = (~meta.is_starter).values
fol = obs & (meta.Entered == 1).values & (meta.L > 0).values
non = obs & (meta.Entered == 0).values
meta['frank'] = np.nan
for t, g in meta[fol].groupby('trial'):
    meta.loc[g.sort_values('EntryLatency').index, 'frank'] = np.arange(1, len(g) + 1)
cov = np.isfinite(meta.StarterLat.values) & (meta.StarterLat.values >= PRE) & \
      (meta.StarterLat.values + POST <= meta.n_tot.values / FS)

ARRAYS = [('follower', fol & cov), ('follower_fast', fol & cov & (meta.L <= 10).values),
          ('follower_first', fol & cov & (meta.frank == 1).values), ('nonfollower', non & cov)]
ROWS = [('All mice', lambda s: np.ones(s.sum(), bool)),
        ('Provider', lambda s: (meta.Mouse_ID.values[s] == 'A2')),
        ('Others', lambda s: (meta.Mouse_ID.values[s] != 'A2'))]


def masks(S):
    """추종자는 자기 진입 0.25 s 전까지만 유효."""
    M = np.ones((len(S), len(T)), bool)
    L = S.L.values
    for i, l in enumerate(L):
        if np.isfinite(l):
            M[i] = T < (l - 0.25)
    return M


def tmap(IDX, MK, buf):
    V = buf[IDX]
    D = V - V[:, BASE].mean(1, keepdims=True)
    D = np.where(MK & np.isfinite(D), D, np.nan)
    n = np.isfinite(D).sum(0)
    with np.errstate(invalid='ignore'):
        m = np.nanmean(D, 0); s = np.nanstd(D, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
        t = np.where(n >= MINN, m / s, 0.0)
    return np.nan_to_num(t), m, s, n


def clusters(tm, thr):
    out = []
    for sg in (1, -1):
        lab, k = label(sg * tm > thr)
        for c in range(1, k + 1):
            out.append((float(np.abs(tm[lab == c]).sum()), sg, lab == c))
    return out


rc, rt = [], []
for aname, asel in ARRAYS:
    ev = meta.StarterLat.values
    for rname, rfun in ROWS:
        sel = asel.copy(); sel[sel] = rfun(asel)
        S = meta[sel]; n = len(S)
        if n < 25:
            print('skip %-16s %-9s n=%d' % (aname, rname, n)); continue
        start = S.start.values.astype(np.int64); ntot = S.n_tot.values.astype(np.int64)
        evb = np.round(ev[sel] * FS).astype(np.int64)
        MK = masks(S)
        IDX = start[:, None] + evb[:, None] + W[None, :]
        nb = MK.sum(0)
        thr = stats.t.ppf(1 - CF / 2, max(nb[nb >= MINN].min() - 1, 5))
        obs_t = {}
        for g in REG:
            for b in BANDS:
                tm, m_, s_, nn = tmap(IDX, MK, X[(g, b)]); obs_t[(g, b)] = tm
                for j, tt in enumerate(T):
                    rt.append(dict(array=aname, row=rname, region=g, band=b, t=float(tt),
                                   mean=float(m_[j]) if nn[j] >= MINN else np.nan,
                                   sem=float(s_[j]) if nn[j] >= MINN else np.nan,
                                   n_bin=int(nn[j]), n=n))
        lo = int(PRE * FS); hi = ntot - int(POST * FS) - 1
        null = np.zeros(NPERM)
        for p in range(NPERM):
            eb = lo + (rng.random(n) * np.maximum(hi - lo, 1)).astype(np.int64)
            far = np.abs(eb - evb) >= 3 * FS
            eb = np.where(far | (hi - lo < 8 * FS), eb, np.minimum(evb + 3 * FS, hi))
            I2 = start[:, None] + eb[:, None] + W[None, :]
            mx = 0.0
            for g in REG:
                for b in BANDS:
                    cl = clusters(tmap(I2, MK, X[(g, b)])[0], thr)
                    if cl:
                        mx = max(mx, max(c[0] for c in cl))
            null[p] = mx
        for g in REG:
            for b in BANDS:
                for mass, sg, mk in clusters(obs_t[(g, b)], thr):
                    tt = T[mk]; k = np.argmax(np.abs(obs_t[(g, b)][mk]))
                    rc.append(dict(array=aname, row=rname, region=g, band=b,
                                   t_start=float(tt.min()), t_end=float(tt.max() + 1 / FS),
                                   sign='+' if sg > 0 else '−', mass=round(mass, 1),
                                   peak_t=float(tt[k]), peak_stat=float(obs_t[(g, b)][mk][k]),
                                   P_family=float((null >= mass).mean()), n=n))
        k = sum(1 for r in rc if r['array'] == aname and r['row'] == rname and r['P_family'] < 0.05)
        print('%-16s %-9s n=%4d thr=%.2f  유의 군집 %d' % (aname, rname, n, thr, k)); sys.stdout.flush()

pd.DataFrame(rc).to_csv('pair_clusters.csv', index=False)
pd.DataFrame(rt).to_csv('pair_means.csv', index=False)
C = pd.DataFrame(rc)
print('\n유의 군집 (P_family < 0.05)')
S = C[C.P_family < 0.05].sort_values(['array', 'row', 'P_family'])
print(S[['array', 'row', 'region', 'band', 't_start', 't_end', 'sign', 'peak_stat', 'P_family', 'n']]
      .to_string(index=False) if len(S) else '  없음')
print('\n0.05 ≤ P < 0.10')
S2 = C[(C.P_family >= 0.05) & (C.P_family < 0.10)]
print(S2[['array', 'row', 'region', 'band', 't_start', 't_end', 'sign', 'P_family', 'n']]
      .to_string(index=False) if len(S2) else '  없음')
