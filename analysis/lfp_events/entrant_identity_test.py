# -*- coding: utf-8 -*-
"""fig. S10 대조 검정: 관찰 중인 비제공자(5마리)의 동료 진입 후 0.25–2.25 s 대역 파워가
진입한 동료가 제공자(A2)인지에 따라 다른가 — 개체 내 검정.
개체별 세션 추세(절편 + pos) 제거 → 특징 표준화 → 진입자 지표(제공자 = 1)에 회귀;
지표를 개체 내에서 5,000회 순열. 전 18 특징을 훑고 가족별(최대 |b|) P 를 함께 보고.
자료: ext_data_s.pkl (θ tilt 와 동일 추출; 사후 창 유한값 요구)."""
import numpy as np, pandas as pd, pickle, sys
rng = np.random.default_rng(20261001); NPERM = 5000
REG = ['PFC', 'NAc', 'BLA']; B = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
HZ = {'ltheta': 'θ 4–8', 'htheta': 'θ 8–12', 'lbeta': 'β 18–24', 'hbeta': 'β 24–32', 'lgamma': 'γ 35–50', 'hgamma': 'γ 70–90'}
NP5 = ['A1', 'A3', 'A4', 'A5', 'A6']; PROVIDER = 'A2'
d = pickle.load(open('ext_data_s.pkl', 'rb')); meta, X, T = d['meta'].copy(), d['X'], d['T']
m = (T >= 0.25) & (T < 2.25)
cols = []
for g_ in REG:
    for b_ in B:
        A = X[(g_, b_)][:, m]
        c = 'post_%s_%s' % (g_, b_); cols.append(c)
        meta[c] = np.where(np.isfinite(A).all(1), A.mean(1), np.nan)
S = meta[meta.Mouse_ID.isin(NP5) & (meta.starter != meta.Mouse_ID)].dropna(subset=cols).reset_index(drop=True)
S['prov'] = (S.starter == PROVIDER).astype(float)


def resid(M, g, pos):
    out = np.empty_like(M, dtype=float)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        D = np.column_stack([np.ones(len(i)), pos[i]])
        out[i] = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
    return out


def test(S, lab):
    g = S.Mouse_ID.values; pos = S.pos.values
    Z = resid(S[cols].values.astype(float), g, pos); Z = Z / Z.std(0)
    x = resid(S[['prov']].values.astype(float), g, pos)[:, 0]
    xx = float(x @ x)
    b0 = (Z.T @ x) / xx                        # 특징 z 단위 / 제공자 지표
    idx = [np.where(g == u)[0] for u in np.unique(g)]
    null = np.empty((NPERM, len(cols)))
    for k in range(NPERM):
        xp = x.copy()
        for i in idx:
            xp[i] = x[rng.permutation(i)]
        null[k] = (Z.T @ xp) / float(xp @ xp)
    P = (np.abs(null) >= np.abs(b0)).mean(0)
    Pfam = (np.abs(null).max(1)[:, None] >= np.abs(b0)[None, :]).mean(0)
    print('\n===== %s  n = %d trials, %d mice; provider entrant on %d' % (lab, len(S), len(idx), int(S.prov.sum())))
    rows = []
    for c, b, p, pf in zip(cols, b0, P, Pfam):
        g_, b_ = c.split('_')[1], c.split('_')[2]
        rows.append(dict(subset=lab, region=g_, band=HZ[b_], slope_z=round(float(b), 3), P=round(float(p), 4), P_family=round(float(pf), 4), n=len(S)))
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    return R


out = [test(S, 'all observers'), test(S[S.follow == 1].reset_index(drop=True), 'followers'), test(S[S.follow == 0].reset_index(drop=True), 'stayed out')]
R = pd.concat(out, ignore_index=True)
R.to_csv('entrant_identity_test.csv', index=False)
print('\nsaved entrant_identity_test.csv')
