# -*- coding: utf-8 -*-
"""Fig. 5D 용: θ tilt 의 within-mouse t 시간 곡선 (동료 진입 정렬, −8…+3 s) 과 시간축 군집 검정.
ext_scan_s.py 와 동일: 개체 절편 + 개체별 세션 기울기 제거, 부분상관 t, 개체 내 순열 2000,
family-wise 는 tilt·PFC high β 두 지도의 최대 군집 질량. 평균 유발 반응(기저 −8…−5 s 대비)도 함께 저장."""
import numpy as np, pandas as pd, pickle
from scipy import stats
from scipy.ndimage import label
rng = np.random.default_rng(20260930); NPERM = 2000; CF = 0.10; MINN = 40
d = pickle.load(open('ext_data_s.pkl', 'rb')); meta, X, T = d['meta'].copy(), d['X'], d['T']
NP5 = ['A1', 'A3', 'A4', 'A5', 'A6']
sel = meta.Mouse_ID.isin(NP5).values
m = meta[sel].reset_index(drop=True); g = m.Mouse_ID.values; pos = m.pos.values
y = m.follow.values.astype(float); n = len(m)
tilt = 0.5 * ((X[('PFC', 'htheta')] - X[('PFC', 'ltheta')]) + (X[('NAc', 'htheta')] - X[('NAc', 'ltheta')]))[sel]
hbeta = X[('PFC', 'hbeta')][sel]
MAPS = {'theta tilt (PFC+NAc)': tilt, 'PFC high beta': hbeta}


def design(gg, pp):
    u = np.unique(gg)
    return np.column_stack([(gg == v).astype(float) for v in u] + [pp - pp.mean()] +
                           [(gg == v) * (pp - pp[gg == v].mean()) for v in u])


idx = [np.where(g == u)[0] for u in np.unique(g)]
Yp = np.empty((n, NPERM + 1)); Yp[:, 0] = y
for k in range(1, NPERM + 1):
    yp = y.copy()
    for i in idx:
        yp[i] = y[rng.permutation(i)]
    Yp[:, k] = yp
Tm = {k: np.zeros((len(T), NPERM + 1)) for k in MAPS}; nb = np.zeros(len(T), int)
for b in range(len(T)):
    av = np.all([np.isfinite(A[:, b]) for A in MAPS.values()], axis=0)
    if av.sum() < MINN or len(np.unique(g[av])) < 5:
        continue
    nb[b] = av.sum(); Dm = design(g[av], pos[av]); R = np.eye(av.sum()) - Dm @ np.linalg.pinv(Dm)
    Yr = R @ Yp[av]; Yr /= np.linalg.norm(Yr, axis=0)
    df = av.sum() - np.linalg.matrix_rank(Dm) - 1
    for k, A in MAPS.items():
        x = R @ A[av, b].astype(float); x /= np.linalg.norm(x); r = x @ Yr
        Tm[k][b] = r * np.sqrt(df / (1 - r ** 2))
thr = stats.t.ppf(1 - CF / 2, max(nb[nb > 0].min() - 12, 10))


def cl(v):
    out = []
    for s in (1, -1):
        lab, kk = label(np.nan_to_num(s * v) > thr)
        for c in range(1, kk + 1):
            out.append((np.abs(v[lab == c]).sum(), s, lab == c))
    return out


null = np.array([max([c[0] for k in MAPS for c in cl(Tm[k][:, p])] or [0]) for p in range(1, NPERM + 1)])
rows = []
for k in MAPS:
    for mass, s, mk in cl(Tm[k][:, 0]):
        tt = T[mk]
        rows.append(dict(key=k, t_start=float(tt.min()), t_end=float(tt.max() + 0.125),
                         sign='+' if s > 0 else '−', mass=round(mass, 1),
                         peak_t=float(tt[np.argmax(np.abs(Tm[k][mk, 0]))]),
                         P_family=float((null >= mass).mean()), n_min=int(nb[mk].min()), n_max=int(nb[mk].max())))
CL = pd.DataFrame(rows).sort_values('P_family')
print('threshold |t| > %.2f ; bins tested %d/%d' % (thr, (nb > 0).sum(), len(T)))
print(CL.to_string(index=False))
TC = pd.concat([pd.DataFrame(dict(key=k, t=T, tstat=Tm[k][:, 0], n=nb)) for k in MAPS])
TC.to_csv('tilt_timecourse.csv', index=False); CL.to_csv('tilt_clusters.csv', index=False)

# 평균 유발 반응 (기저 −8…−5 s 대비, 비제공자 5마리) — '평균은 움직이지 않는다' 를 같은 축에 놓기 위함
BASE = (T >= -8) & (T < -5)
ev = []
for k, A in MAPS.items():
    D = A - np.nanmean(A[:, BASE], axis=1, keepdims=True)
    mu = np.nanmean(D, 0); se = np.nanstd(D, 0) / np.sqrt(np.isfinite(D).sum(0))
    ev.append(pd.DataFrame(dict(key=k, t=T, mean=mu, sem=se, n=np.isfinite(D).sum(0))))
pd.concat(ev).to_csv('tilt_evoked.csv', index=False)
print('\n평균 유발 반응(tilt) 0.25–2.25 s: %.3f ± %.3f (z)' %
      (np.nanmean(ev[0][(ev[0].t >= 0.25) & (ev[0].t < 2.25)]['mean']),
       np.nanmean(ev[0][(ev[0].t >= 0.25) & (ev[0].t < 2.25)]['sem'])))
