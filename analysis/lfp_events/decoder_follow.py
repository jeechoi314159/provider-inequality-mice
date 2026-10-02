# -*- coding: utf-8 -*-
"""개체 제외 교차검증 해독기: 동료 진입 후 18특징 → 그 시행의 합류.
개체별 세션 추세 제거 후 개체 안에서 표준화 → 4마리로 릿지 적합(λ 는 훈련 안에서 다시
개체 제외로 선택) → 남은 1마리에서 AUC. 귀무는 합류 라벨을 개체 안에서 섞고 절차 전체 반복.
분할이 고정이므로 (X'X+λI)^-1 을 미리 계산해 순열마다 행렬-벡터 곱만 한다."""
import numpy as np, pandas as pd, pickle, sys
rng = np.random.default_rng(20260930); NPERM = 1000
REG = ['PFC', 'NAc', 'BLA']; B = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
KEY = [(g_, b_) for g_ in REG for b_ in B]; NP5 = ['A1', 'A3', 'A4', 'A5', 'A6']
LAM = 10.0 ** np.arange(-1, 3.01, 0.5)

d = pickle.load(open('ext_data_s.pkl', 'rb')); meta, X, T = d['meta'].copy(), d['X'], d['T']
for wn, m in (('post', (T >= 0.25) & (T < 2.25)), ('pre', (T >= -2.0) & (T < 0.0))):
    for k in KEY:
        A = X[k][:, m]
        meta['%s_%s_%s' % (wn, k[0], k[1])] = np.where(np.isfinite(A).all(1), A.mean(1), np.nan)
C = {w: ['%s_%s_%s' % (w, g_, b_) for g_ in REG for b_ in B] for w in ('post', 'pre')}
S = meta[meta.Mouse_ID.isin(NP5)].dropna(subset=C['post'] + C['pre']).reset_index(drop=True)
g = S.Mouse_ID.values; pos = S.pos.values; yb = S.follow.values.astype(float)
MICE = np.unique(g)


def prep(cols):
    M = S[cols].values.astype(float); out = np.empty_like(M)
    for u in MICE:
        i = np.where(g == u)[0]
        D = np.column_stack([np.ones(len(i)), pos[i]])
        R = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
        out[i] = R / R.std(0)
    return out


def auc(sc, lab):
    o = np.argsort(sc, kind='mergesort'); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def build(Z):
    """분할별 (X'X+λI)^-1 X'  를 미리 만든다 → 순열마다 (…)@y 한 번."""
    P = {}
    for u in MICE:
        te = g == u; tr = ~te
        inner = []
        for v in MICE[MICE != u]:
            ite = tr & (g == v); itr = tr & (g != v)
            Xi = Z[itr]; G = Xi.T @ Xi
            H = [np.linalg.solve(G + l * np.eye(Z.shape[1]), Xi.T) for l in LAM]
            inner.append((itr, ite, H))
        Xt = Z[tr]; G = Xt.T @ Xt
        Ht = [np.linalg.solve(G + l * np.eye(Z.shape[1]), Xt.T) for l in LAM]
        P[u] = dict(te=te, tr=tr, inner=inner, Ht=Ht)
    return P


def ycent(yv):
    o = yv.copy()
    for u in MICE:
        i = g == u; o[i] = yv[i] - yv[i].mean()
    return o


def lomo(Z, P, yv, ret_w=False):
    yc = ycent(yv); a = []; ws = []
    for u in MICE:
        p = P[u]; best, bk = -np.inf, 0
        for k in range(len(LAM)):
            s = [auc(Z[ite] @ (H[k] @ yc[itr]), yv[ite]) for itr, ite, H in p['inner']]
            s = np.nanmean(s)
            if s > best:
                best, bk = s, k
        w = p['Ht'][bk] @ yc[p['tr']]
        a.append(auc(Z[p['te']] @ w, yv[p['te']])); ws.append(w)
    return (np.nanmean(a), np.array(a), np.mean(ws, 0)) if ret_w else np.nanmean(a)


def test(Z, lab):
    P = build(Z)
    a0, per, w = lomo(Z, P, yb, ret_w=True)
    idx = [np.where(g == u)[0] for u in MICE]
    null = np.empty(NPERM)
    for p in range(NPERM):
        yp = yb.copy()
        for i in idx:
            yp[i] = yb[rng.permutation(i)]
        null[p] = lomo(Z, P, yp)
    pv = (null >= a0).mean()
    print('  %-24s AUC = %.3f   P = %.4f   개체별 %s   귀무 평균 %.3f'
          % (lab, a0, pv, np.round(per, 2).tolist(), null.mean())); sys.stdout.flush()
    return a0, pv, per, w


Zp, Zr = prep(C['post']), prep(C['pre'])
print('시행 %d, 개체 %d, 합류율 %.2f' % (len(S), len(MICE), yb.mean()))
print('\n== 개체 제외 교차검증 해독기')
r_post = test(Zp, '진입 후 0.25–2.25 s')
r_pre = test(Zr, '진입 전 −2–0 s')
r_both = test(np.column_stack([Zr, Zp]), '전 + 후 (36특징)')

print('\n== 같은 평가 기준의 단일 측정치 (적합 없음)')
W = pd.read_csv('irc_weights.csv').weight.values
for nm, sc in (('PFC high β 단독', Zp[:, C['post'].index('post_PFC_hbeta')]),
               ('IRC 투영', Zp @ W), ('cost 성분', Zp @ np.where(W < 0, -W, 0))):
    a = [auc(sc[g == u], yb[g == u]) for u in MICE]
    print('  %-24s AUC = %.3f                개체별 %s' % (nm, np.nanmean(a), np.round(a, 2).tolist()))

print('\n== 해독기 가중치 (5겹 평균, |w| 상위 6)')
w = r_post[3]
for k in np.argsort(-np.abs(w))[:6]:
    print('   %-16s %+.3f' % (C['post'][k].replace('post_', ''), w[k]))
pd.DataFrame(dict(feature=[c.replace('post_', '') for c in C['post']], weight=w)).to_csv('decoder_weights.csv', index=False)
pd.DataFrame(dict(analysis=['post', 'pre', 'pre+post'], auc=[r_post[0], r_pre[0], r_both[0]],
                  P=[r_post[1], r_pre[1], r_both[1]])).to_csv('decoder_auc.csv', index=False)
