# -*- coding: utf-8 -*-
"""주의 가설의 직접 검정(벡터화): 같은 개체 안에서 '뒤따라 들어간 시행' 과 '끝내 안 들어간 시행' 의
동료 진입 정렬 반응이 다른가. 개체 내 중심화(라벨과 무관하게 선계산) 후 두 집단 평균차,
라벨을 개체 안에서 순열(2000회), 18개 지도 최대 군집 질량으로 family-wise 보정."""
import numpy as np, pandas as pd, pickle
from scipy import stats
from scipy.ndimage import label
d = pickle.load(open('allmap_traces.pkl', 'rb'))
meta, X, FS = d['meta'].copy(), d['X'], d['FS']
REG = ['PFC', 'NAc', 'BLA']; BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
KEYS = [(g, b) for g in REG for b in BANDS]
PRE, POST = 2.0, 3.0
W = np.arange(-int(PRE * FS), int(POST * FS) + 1); T = W / FS
BASE = (T >= -2.0) & (T <= -1.0)
NPERM, CF, MINN = 2000, 0.05, 15
rng = np.random.default_rng(7)
meta['L'] = meta.EntryLatency - meta.StarterLat
obs = (~meta.is_starter).values
fol = obs & (meta.Entered == 1).values & (meta.L > 0).values
non = obs & (meta.Entered == 0).values
meta['frank'] = np.nan
for t, g in meta[fol].groupby('trial'):
    meta.loc[g.sort_values('EntryLatency').index, 'frank'] = np.arange(1, len(g) + 1)
cov = np.isfinite(meta.StarterLat.values) & (meta.StarterLat.values >= PRE) & \
      (meta.StarterLat.values + POST <= meta.n_tot.values / FS)


def pull(sel):
    S = meta[sel]
    IDX = S.start.values[:, None] + np.round(meta.StarterLat.values[sel] * FS).astype(int)[:, None] + W[None, :]
    MK = np.ones((len(S), len(T)), bool)
    for i, l in enumerate(S.L.values):
        if np.isfinite(l):
            MK[i] = T < (l - 0.25)
    out = []
    for k in KEYS:
        V = X[k][IDX]; D = V - V[:, BASE].mean(1, keepdims=True)
        out.append(np.where(MK & np.isfinite(D), D, np.nan))
    return out, S.Mouse_ID.values


def cl(tm, thr):
    o = []
    for sg in (1, -1):
        L_, k = label(sg * tm > thr)
        for c in range(1, k + 1):
            o.append((float(np.abs(tm[L_ == c]).sum()), sg, L_ == c))
    return o


def contrast(selA, selB, lab):
    DA, gA = pull(selA); DB, gB = pull(selB)
    g = np.concatenate([gA, gB]); n = len(g)
    y0 = np.concatenate([np.ones(len(gA)), np.zeros(len(gB))])
    idx = [np.where(g == u)[0] for u in np.unique(g)]
    Z, Fm = [], []
    for a, b in zip(DA, DB):
        M = np.vstack([a, b])                      # 개체 내 중심화(라벨 무관)
        for i in idx:
            M[i] -= np.nanmean(M[i], 0, keepdims=True)
        F = np.isfinite(M); Z.append(np.nan_to_num(M)); Fm.append(F.astype(float))
    Y = np.empty((n, NPERM + 1)); Y[:, 0] = y0
    for p in range(1, NPERM + 1):
        yp = y0.copy()
        for i in idx:
            yp[i] = y0[rng.permutation(i)]
        Y[:, p] = yp
    Yb = 1 - Y
    thr = stats.t.ppf(1 - CF / 2, 60)
    TM = []
    for M, F in zip(Z, Fm):
        S1 = M.T @ Y; S2 = (M ** 2).T @ Y; C1 = F.T @ Y
        T1 = M.T @ Yb; T2 = (M ** 2).T @ Yb; C2 = F.T @ Yb
        with np.errstate(invalid='ignore', divide='ignore'):
            mA, mB = S1 / C1, T1 / C2
            vA = (S2 - C1 * mA ** 2) / np.maximum(C1 - 1, 1)
            vB = (T2 - C2 * mB ** 2) / np.maximum(C2 - 1, 1)
            t = (mA - mB) / np.sqrt(vA / C1 + vB / C2)
        t = np.where((C1 >= MINN) & (C2 >= MINN) & np.isfinite(t), t, 0.0)
        TM.append(t)
    null = np.zeros(NPERM)
    for p in range(1, NPERM + 1):
        mx = 0.0
        for t in TM:
            c = cl(t[:, p], thr)
            if c:
                mx = max(mx, max(x[0] for x in c))
        null[p - 1] = mx
    rows = []
    for k, t in zip(KEYS, TM):
        for mass, sg, mk in cl(t[:, 0], thr):
            tt = T[mk]
            rows.append(dict(contrast=lab, region=k[0], band=k[1], t_start=float(tt.min()),
                             t_end=float(tt.max() + 1 / FS), sign='+' if sg > 0 else '−',
                             mass=round(mass, 1), P_family=float((null >= mass).mean()),
                             nA=len(gA), nB=len(gB)))
    R = pd.DataFrame(rows)
    print('\n== %s   n = %d vs %d, 군집 %d개' % (lab, len(gA), len(gB), len(R)))
    if len(R) and (R.P_family < 0.10).any():
        print(R[R.P_family < 0.10].sort_values('P_family')
              [['region', 'band', 't_start', 't_end', 'sign', 'P_family']].to_string(index=False))
    else:
        print('   P < 0.10 인 군집 없음')
    return R


out = [contrast(fol & cov & (meta.frank == 1).values, non & cov, 'first follower vs non-follower'),
       contrast(fol & cov, non & cov, 'any follower vs non-follower'),
       contrast(fol & cov & (meta.L <= 10).values, non & cov, 'fast follower (L<=10 s) vs non-follower')]
pd.concat(out).to_csv('pair_contrasts.csv', index=False)
