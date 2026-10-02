# -*- coding: utf-8 -*-
"""구성 변경 자료에서의 θ tilt 검정. 코호트 A 와 동일 절차를 주분석으로 하고,
A5 의 채널 이상(θ2−θ1 중앙값 −2.03, 다른 개체와 분포가 겹치지 않음)에 대한 민감도 둘을 함께 본다."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd
rng = np.random.default_rng(20260930); NPERM = 10000
d = pd.read_csv((LFP + 'member_change/' + 'tilt_member_change.csv'))
d = d[d.mouse != 'A2'].copy()                       # 1행뿐
d['key'] = d.day * 1000 + d.trial
d['pos'] = d.key.rank(method='dense'); d['pos'] = (d.pos - d.pos.min()) / (d.pos.max() - d.pos.min())


def resid(M, g, pos):
    out = np.empty_like(M, dtype=float)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        D = np.column_stack([np.ones(len(i)), pos[i]])
        out[i] = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
    return out


def test(S, col, lab, within=False):
    S = S.dropna(subset=[col]).reset_index(drop=True)
    g = S.mouse.values; pos = S.pos.values
    Z = resid(S[[col]].values.astype(float), g, pos)
    if within:
        for u in np.unique(g):
            i = g == u; Z[i] = Z[i] / Z[i].std()
    else:
        Z = Z / Z.std(0)
    y = resid(S[['follow']].values.astype(float), g, pos)[:, 0]
    b0 = float(np.linalg.lstsq(Z, y, rcond=None)[0][0])
    idx = [np.where(g == u)[0] for u in np.unique(g)]
    null = np.empty(NPERM)
    for k in range(NPERM):
        yp = y.copy()
        for i in idx:
            yp[i] = y[rng.permutation(i)]
        null[k] = np.linalg.lstsq(Z, yp, rcond=None)[0][0]
    P = float((np.abs(null) >= abs(b0)).mean())
    per = {u: round(float(np.linalg.lstsq(Z[g == u], y[g == u], rcond=None)[0][0]), 3)
           for u in np.unique(g)}
    sgn = sum(np.sign(v) == np.sign(b0) for v in per.values())
    print('  %-42s n=%-4d 개체 %d  β = %+.4f/SD  P = %.4f  부호 %d/%d'
          % (lab, len(S), len(per), b0, P, sgn, len(per)))
    print('      개체별: %s' % per)
    return b0, P


print('== 주분석 (코호트 A 와 동일 절차)')
test(d, 'tilt', 'θ tilt, 전체 개체')
test(d, 'hbeta_PFC', 'PFC high β, 전체 개체 (비교)')
print('\n== 민감도 1: A5 제외 (채널 이상)')
test(d[d.mouse != 'A5'], 'tilt', 'θ tilt, A5 제외')
test(d[d.mouse != 'A5'], 'hbeta_PFC', 'PFC high β, A5 제외')
print('\n== 민감도 2: 개체 안에서 표준화 (척도 차이 흡수)')
test(d, 'tilt', 'θ tilt, 개체 내 표준화', within=True)
test(d[d.mouse != 'A5'], 'tilt', 'θ tilt, A5 제외 + 개체 내 표준화', within=True)
print('\n== 영역별 (A5 제외)')
for c in ('tilt_PFC', 'tilt_NAc'):
    test(d[d.mouse != 'A5'], c, c)
