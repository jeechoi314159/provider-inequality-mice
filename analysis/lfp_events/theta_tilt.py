# -*- coding: utf-8 -*-
"""사전 선언한 검정: θ tilt = (θ 8–12) − (θ 4–8), PFC·NAc 평균, 창 0.25–2.25 s.
β 특징과 똑같은 절차 — 개체 내, 개체별 세션 추세 제거, 개체 내 순열 10,000,
두 표본(n=616 / n=327), 기저 구간 운동에너지 공변량, 개체별 부호."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, pickle
rng = np.random.default_rng(20260930); NPERM = 10000
REG = ['PFC', 'NAc', 'BLA']; B = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
NP5 = ['A1', 'A3', 'A4', 'A5', 'A6']
d = pickle.load(open('ext_data_s.pkl', 'rb')); meta, X, T = d['meta'].copy(), d['X'], d['T']
for wn, m in (('post', (T >= 0.25) & (T < 2.25)), ('pre', (T >= -2.0) & (T < 0.0))):
    for g_ in REG:
        for b_ in B:
            A = X[(g_, b_)][:, m]
            meta['%s_%s_%s' % (wn, g_, b_)] = np.where(np.isfinite(A).all(1), A.mean(1), np.nan)
CP = ['post_%s_%s' % (g_, b_) for g_ in REG for b_ in B]
CR = ['pre_%s_%s' % (g_, b_) for g_ in REG for b_ in B]
meta['tilt'] = ((meta.post_PFC_htheta - meta.post_PFC_ltheta) +
                (meta.post_NAc_htheta - meta.post_NAc_ltheta)) / 2.0
meta['hbeta'] = meta.post_PFC_hbeta
KE = pd.read_csv((INTER + 'ke_group.csv'))
meta = meta.merge(KE, on=['Mouse_ID', 'trial'], how='left')
meta['lke_base'] = np.log10(meta.ke_base.clip(lower=1e-8))
meta['lke_act'] = np.log10(meta.ke_act.clip(lower=1e-8))


def resid(M, g, pos):
    out = np.empty_like(M, dtype=float)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        D = np.column_stack([np.ones(len(i)), pos[i]])
        out[i] = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
    return out


def fit(S, cols, lab):
    g = S.Mouse_ID.values; pos = S.pos.values
    Z = resid(S[cols].values.astype(float), g, pos); Z = Z / Z.std(0)
    y = resid(S[['follow']].values.astype(float), g, pos)[:, 0]
    b0 = np.linalg.lstsq(Z, y, rcond=None)[0]
    idx = [np.where(g == u)[0] for u in np.unique(g)]
    null = np.empty((NPERM, len(cols)))
    for k in range(NPERM):
        yp = y.copy()
        for i in idx:
            yp[i] = y[rng.permutation(i)]
        null[k] = np.linalg.lstsq(Z, yp, rcond=None)[0]
    P = (np.abs(null) >= np.abs(b0)).mean(0)
    per = {u: round(float(np.linalg.lstsq(Z[g == u], y[g == u], rcond=None)[0][0]), 3)
           for u in np.unique(g)}
    sgn = sum(np.sign(v) == np.sign(b0[0]) for v in per.values())
    print('  %-34s n=%-4d  %s   부호 %d/%d' % (lab, len(S),
          '  '.join('%s %+.4f (P=%.4f)' % (c, b, p) for c, b, p in zip(cols, b0, P)), sgn, len(per)))
    return b0, P, per


for tag, need in (('n=616 (사후 창만)', CP), ('n=327 (진입 전 창까지 요구)', CP + CR)):
    S = meta[meta.Mouse_ID.isin(NP5)].dropna(subset=need + ['tilt', 'hbeta']).reset_index(drop=True)
    print('\n===== %s' % tag)
    fit(S, ['tilt'], 'θ tilt 단독')
    fit(S, ['hbeta'], 'PFC high β 단독 (비교)')
    fit(S, ['tilt', 'hbeta'], 'θ tilt + PFC high β')
    S2 = S.dropna(subset=['lke_base']).reset_index(drop=True)
    fit(S2, ['tilt', 'lke_base'], 'θ tilt + 기저 운동량 (사전 공변량)')
    fit(S2, ['tilt', 'lke_base', 'lke_act'], 'θ tilt + 기저 + 채집 운동량 (과잉 통제)')
    print('    상관 r(tilt, hbeta) = %+.3f' % S[['tilt', 'hbeta']].corr().iloc[0, 1])
