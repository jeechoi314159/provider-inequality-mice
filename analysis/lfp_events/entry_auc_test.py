# -*- coding: utf-8 -*-
"""진입 AUC 검정 (벡터화, Frisch–Waugh). 특징별 P 와 18개 특징 family-wise P 를 함께 낸다.
측정 품질(단일시행 효과크기·개체 간 분산 비중·반분 신뢰도)과 검출 가능한 효과 크기 하한도 보고."""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(20260930); NPERM = 10000
S = pd.read_csv('entry_auc_trials.csv')
REG = ['PFC', 'NAc', 'BLA']; BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
FE = ['%s_%s' % (g, b) for g in REG for b in BANDS]
NICE = {'ltheta': 'θ 4–8', 'htheta': 'θ 8–12', 'lbeta': 'β 18–24', 'hbeta': 'β 24–32',
        'lgamma': 'γ 35–50', 'hgamma': 'γ 70–90'}
lab = lambda f: '%s %s' % (f.split('_')[0], NICE[f.split('_', 1)[1]])
S['log_entry'] = np.log10(S.entry.clip(lower=0.2)); S['n_before_f'] = S.n_before.astype(float)


def resid(M, g, pos):
    out = np.empty_like(M, dtype=float)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        D = np.column_stack([np.ones(len(i)), pos[i]])
        out[i] = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
    return out


def run(df, ycol, covs, title):
    d = df.dropna(subset=[ycol] + FE + covs).copy()
    g = d.Mouse_ID.values; pos = d.pos.values; n = len(d)
    Y = resid(d[[ycol]].values.astype(float), g, pos)[:, 0]; Y = (Y - Y.mean()) / Y.std()
    A = resid(d[FE].values.astype(float), g, pos); A = A / A.std(0)
    C = np.column_stack([np.ones(n)] + ([resid(d[covs].values.astype(float), g, pos)] if covs else []))
    Mc = np.eye(n) - C @ np.linalg.pinv(C)
    Ap = Mc @ A; den = (Ap ** 2).sum(0)
    idx = [np.where(g == u)[0] for u in np.unique(g)]
    Yp = np.empty((n, NPERM + 1)); Yp[:, 0] = Y
    for p in range(1, NPERM + 1):
        y = Y.copy()
        for i in idx:
            y[i] = Y[rng.permutation(i)]
        Yp[:, p] = y
    B = (Ap.T @ (Mc @ Yp)) / den[:, None]          # 18 x (NPERM+1)
    b0 = B[:, 0]; null = B[:, 1:]
    P_feat = (np.abs(null) >= np.abs(b0)[:, None]).mean(1)
    P_fam = (np.abs(null).max(0)[None, :] >= np.abs(b0)[:, None]).mean(1)
    bound = float(np.quantile(np.abs(null).max(0), 0.95))
    R = pd.DataFrame(dict(analysis=title, feature=[lab(f) for f in FE], beta=b0.round(4),
                          P=P_feat, P_family=P_fam, n=n, mice=d.Mouse_ID.nunique(),
                          detect_bound=round(bound, 3))).sort_values('P_family')
    print('\n== %s   n=%d, 개체 %d, 검출 하한 |β| ≥ %.3f SD' % (title, n, d.Mouse_ID.nunique(), bound))
    print(R.head(4)[['feature', 'beta', 'P', 'P_family']].to_string(index=False))
    return R


out = [run(S, 'n_before_f', ['log_entry'], 'how many are already inside'),
       run(S, 'worked', ['log_entry', 'n_before_f'], 'whether this mouse retrieves'),
       run(S, 'log_entry', ['n_before_f'], 'how late this mouse enters'),
       run(S, 'log_next', ['log_entry', 'n_before_f'], 'the next mouse’s wait'),
       run(S, 'n_never', ['log_entry', 'n_before_f'], 'how many never enter'),
       run(S[S['rank'] == 1], 'log_next', ['log_entry'], 'the next mouse’s wait (starters only)'),
       run(S[S['rank'] == 1], 'n_never', ['log_entry'], 'how many never enter (starters only)')]
R = pd.concat(out); R.to_csv('entry_auc_tests.csv', index=False)

print('\n\n== AUC 측정 품질 (진입에서 유의했던 여섯 특징)')
K6 = ['PFC_ltheta', 'PFC_lbeta', 'NAc_lbeta', 'BLA_lbeta', 'BLA_hbeta', 'BLA_lgamma']
rows = []
for f in K6:
    v = S[f].values; g = S.Mouse_ID.values
    mu, sd = v.mean(), v.std()
    gm = np.array([v[g == u].mean() for u in np.unique(g)])
    ss_b = sum((g == u).sum() * (v[g == u].mean() - mu) ** 2 for u in np.unique(g))
    icc = ss_b / ((v - mu) ** 2).sum()
    h = []
    for u in np.unique(g):
        i = np.where(g == u)[0]; r = rng.permutation(i)
        h.append((v[r[:len(r) // 2]].mean(), v[r[len(r) // 2:]].mean()))
    h = np.array(h); sh = np.corrcoef(h[:, 0], h[:, 1])[0, 1]
    rows.append(dict(feature=lab(f), mean=round(mu, 3), sd_trial=round(sd, 3),
                     d=round(mu / sd, 3), between_mouse_share=round(icc, 3),
                     splithalf_r=round(sh, 2)))
print(pd.DataFrame(rows).to_string(index=False))
