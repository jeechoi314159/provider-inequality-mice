# -*- coding: utf-8 -*-
"""저차원·부분집합 해독기 변형 — 목록은 돌리기 전에 고정했다.
  주성분 1·2·3·5 (PCA 는 훈련 겹 안에서만 적합), 영역 한정 PFC/NAc/BLA, 대역 한정 θ/β/γ,
  그리고 기준선인 전체 18특징.
평가는 이전과 동일: 개체 제외 릿지, λ 는 훈련 안에서 다시 개체 제외로 선택, 제외 개체 AUC.
귀무 1,000회는 모든 변형에 같은 순열을 쓰고, 변형 간 최대 AUC 로 family-wise 보정한다."""
import numpy as np, pandas as pd, pickle, sys
rng = np.random.default_rng(20260930); NPERM = 1000
REG = ['PFC', 'NAc', 'BLA']; B = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
KEY = [(g_, b_) for g_ in REG for b_ in B]; NP5 = ['A1', 'A3', 'A4', 'A5', 'A6']
LAM = 10.0 ** np.arange(-1, 3.01, 0.5)
d = pickle.load(open('ext_data_s.pkl', 'rb')); meta, X, T = d['meta'].copy(), d['X'], d['T']
m = (T >= 0.25) & (T < 2.25)
for k in KEY:
    A = X[k][:, m]
    meta['post_%s_%s' % k] = np.where(np.isfinite(A).all(1), A.mean(1), np.nan)
COLS = ['post_%s_%s' % (g_, b_) for g_ in REG for b_ in B]
S = meta[meta.Mouse_ID.isin(NP5)].dropna(subset=COLS).reset_index(drop=True)
g = S.Mouse_ID.values; pos = S.pos.values; yb = S.follow.values.astype(float); MICE = np.unique(g)
M = S[COLS].values.astype(float); Z = np.empty_like(M)
for u in MICE:
    i = np.where(g == u)[0]
    D = np.column_stack([np.ones(len(i)), pos[i]])
    R = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
    Z[i] = R / R.std(0)

idxc = {c: k for k, c in enumerate(COLS)}
SUB = {'PFC only': [idxc['post_PFC_%s' % b_] for b_ in B],
       'NAc only': [idxc['post_NAc_%s' % b_] for b_ in B],
       'BLA only': [idxc['post_BLA_%s' % b_] for b_ in B],
       'theta only': [idxc['post_%s_%s' % (g_, b_)] for g_ in REG for b_ in ('ltheta', 'htheta')],
       'beta only': [idxc['post_%s_%s' % (g_, b_)] for g_ in REG for b_ in ('lbeta', 'hbeta')],
       'gamma only': [idxc['post_%s_%s' % (g_, b_)] for g_ in REG for b_ in ('lgamma', 'hgamma')]}
VARIANTS = ([('all 18 features', None, None)] +
            [('%d principal component%s' % (k, 's' if k > 1 else ''), None, k) for k in (1, 2, 3, 5)] +
            [(n, c, None) for n, c in SUB.items()])


def auc(sc, lab):
    o = np.argsort(sc, kind='mergesort'); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pca_fit(Xtr, k):
    mu = Xtr.mean(0); V = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:k]
    return mu, V.T


def build(cols, npc):
    """분할별로 (필요하면 PCA 적합 후) (X'X+λI)^-1 X' 를 미리 만든다."""
    Zs = Z if cols is None else Z[:, cols]
    P = {}
    for u in MICE:
        te = g == u; tr = ~te
        def mk(trm, tem):
            A = Zs[trm]
            if npc:
                mu, V = pca_fit(A, npc); A = (A - mu) @ V; Bt = (Zs[tem] - mu) @ V
            else:
                Bt = Zs[tem]
            G = A.T @ A
            H = [np.linalg.solve(G + l * np.eye(A.shape[1]), A.T) for l in LAM]
            return trm, tem, H, Bt
        inner = [mk(tr & (g != v), tr & (g == v)) for v in MICE[MICE != u]]
        P[u] = dict(te=te, tr=tr, inner=inner, outer=mk(tr, te))
    return P


def ycent(yv):
    o = yv.copy()
    for u in MICE:
        i = g == u; o[i] = yv[i] - yv[i].mean()
    return o


def lomo(P, yv):
    yc = ycent(yv); a = []
    for u in MICE:
        p = P[u]; best, bk = -np.inf, 0
        for k in range(len(LAM)):
            s = np.nanmean([auc(Bt @ (H[k] @ yc[trm]), yv[tem]) for trm, tem, H, Bt in p['inner']])
            if s > best:
                best, bk = s, k
        trm, tem, H, Bt = p['outer']
        a.append(auc(Bt @ (H[bk] @ yc[trm]), yv[tem]))
    return np.nanmean(a), np.array(a)


PS = [(n, build(c, k)) for n, c, k in VARIANTS]
obs = {n: lomo(P, yb) for n, P in PS}
idx = [np.where(g == u)[0] for u in MICE]
null = np.zeros((NPERM, len(PS)))
for p in range(NPERM):
    yp = yb.copy()
    for i in idx:
        yp[i] = yb[rng.permutation(i)]
    for j, (n, P) in enumerate(PS):
        null[p, j] = lomo(P, yp)[0]
    if (p + 1) % 250 == 0:
        print('  순열 %d/%d' % (p + 1, NPERM)); sys.stdout.flush()
mx = null.max(1)
rows = []
for j, (n, _) in enumerate(PS):
    a0, per = obs[n]
    rows.append(dict(variant=n, auc=round(a0, 3), P=float((null[:, j] >= a0).mean()),
                     P_family=float((mx >= a0).mean()), per_mouse=str(np.round(per, 2).tolist()),
                     null_mean=round(null[:, j].mean(), 3)))
# 적합 없는 기준선도 같은 평가로
for nm, sc in (('PFC high β alone (no fitting)', Z[:, idxc['post_PFC_hbeta']]),):
    a = [auc(sc[g == u], yb[g == u]) for u in MICE]
    rows.append(dict(variant=nm, auc=round(np.nanmean(a), 3), P=np.nan, P_family=np.nan,
                     per_mouse=str(np.round(a, 2).tolist()), null_mean=np.nan))
R = pd.DataFrame(rows).sort_values('auc', ascending=False)
R.to_csv('decoder_variants.csv', index=False)
pd.set_option('display.width', 200)
print('\n== 사전 선언한 변형 전부 (AUC 내림차순), family-wise 는 11개 변형의 최대 AUC 기준')
print(R[['variant', 'auc', 'P', 'P_family', 'per_mouse']].to_string(index=False))
print('\n귀무 최대 AUC 의 95%% 지점 = %.3f' % np.quantile(mx, 0.95))
