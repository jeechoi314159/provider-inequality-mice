# -*- coding: utf-8 -*-
"""사회적 맥락 분해: 누가 들어갔나(P/NP) × 누가 보고 있나(P/NP).
Fig. 5D 의 PFC high β → 합류 관계를 네 칸(존재하는 세 칸)으로 나눈다.
지표·통제·검정은 two_signals.py / ext_scan_s.py 와 동일: 개체 내, 개체별 세션 추세 제거,
개체 내 순열(10,000회). 시간 곡선의 t 는 ext_scan_s.py 와 같은 부분상관 t."""
import numpy as np, pandas as pd, pickle
from scipy import stats
from scipy.ndimage import label

rng = np.random.default_rng(20260930)
NPERM = 10000
d = pickle.load(open('ext_data_s.pkl', 'rb'))
meta, X, T = d['meta'].copy(), d['X'], d['T']
PROV = 'A2'
meta['obs_P'] = (meta.Mouse_ID == PROV)
meta['ent_P'] = (meta.starter == PROV)


def wmean(key, tw):
    b = (T >= tw[0]) & (T < tw[1]); A = X[key][:, b]
    return np.where(np.isfinite(A).all(1), A.mean(1), np.nan)


meta['S_pre'] = wmean(('PFC', 'g51_100'), (-2, 0))
meta['S_post'] = wmean(('PFC', 'hbeta'), (0.25, 2.25))


def resid_mat(M, g, pos):
    out = np.empty_like(M, dtype=float)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        D = np.column_stack([np.ones(len(i)), pos[i]])
        out[i] = M[i] - D @ np.linalg.lstsq(D, M[i], rcond=None)[0]
    return out


def joint(S, label_):
    g = S.Mouse_ID.values; pos = S.pos.values
    Z = resid_mat(S[['S_pre', 'S_post']].values.astype(float), g, pos); Z = Z / Z.std(0)
    y = resid_mat(S[['follow']].values.astype(float), g, pos)[:, 0]
    B = np.linalg.lstsq(Z, y, rcond=None)[0]
    idx = [np.where(g == u)[0] for u in np.unique(g)]
    null = np.empty((NPERM, 2))
    for k in range(NPERM):
        yp = y.copy()
        for i in idx:
            yp[i] = y[rng.permutation(i)]
        null[k] = np.linalg.lstsq(Z, yp, rcond=None)[0]
    P = (np.abs(null) >= np.abs(B)).mean(0)
    per = {}
    for u in np.unique(g):
        i = g == u
        per[u] = float(np.linalg.lstsq(Z[i], y[i], rcond=None)[0][1])
    return dict(cell=label_, n=len(S), mice=S.Mouse_ID.nunique(),
                follow_rate=float(S.follow.mean()),
                b_pre=B[0], P_pre=P[0], b_post=B[1], P_post=P[1],
                pos_mice='%d/%d' % (sum(v > 0 for v in per.values()), len(per)),
                per_mouse='; '.join('%s %+.3f' % (k, v) for k, v in sorted(per.items())))


CELLS = [('NP observes P enter', ~meta.obs_P & meta.ent_P),
         ('NP observes NP enter', ~meta.obs_P & ~meta.ent_P),
         ('P observes NP enter', meta.obs_P & ~meta.ent_P)]
rows = []
for lab, sel in CELLS:
    S = meta[sel].dropna(subset=['S_pre', 'S_post']).reset_index(drop=True)
    rows.append(joint(S, lab))
S = meta[~meta.obs_P].dropna(subset=['S_pre', 'S_post']).reset_index(drop=True)
rows.append(joint(S, 'All NP observers (Fig. 5D as drawn)'))
R = pd.DataFrame(rows)
print('== PFC high β (0.25–2.25 s) → joins, by social context')
print(R[['cell', 'n', 'mice', 'follow_rate', 'b_post', 'P_post', 'pos_mice']].to_string(index=False))
print()
print('   (γ before entry in the same model)')
print(R[['cell', 'b_pre', 'P_pre']].to_string(index=False))
print()
for _, r in R.iterrows():
    print('  %-36s %s' % (r.cell, r.per_mouse))

# ---- 상호작용: NP 관찰자에서 진입자가 P 일 때 β 기울기가 달라지는가 (라벨 순열)
S = meta[~meta.obs_P].dropna(subset=['S_pre', 'S_post']).reset_index(drop=True)
g = S.Mouse_ID.values; pos = S.pos.values
Z = resid_mat(S[['S_pre', 'S_post']].values.astype(float), g, pos); Z = Z / Z.std(0)
y = resid_mat(S[['follow']].values.astype(float), g, pos)[:, 0]
e = S.ent_P.values.astype(float); e = e - e.mean()


def fit(ev):
    M = np.column_stack([Z, ev, Z[:, 1] * ev])
    return np.linalg.lstsq(M, y, rcond=None)[0]


b0 = fit(e)
idx = [np.where(g == u)[0] for u in np.unique(g)]
null = np.empty(NPERM)
for k in range(NPERM):
    ep = e.copy()
    for i in idx:
        ep[i] = e[rng.permutation(i)]
    null[k] = fit(ep)[3]
print('\n== interaction  β × (enterer is the provider), NP observers, n=%d' % len(S))
print('   coefficient %+.4f/SD, P = %.4f (entrant label permuted within mouse)'
      % (b0[3], (np.abs(null) >= abs(b0[3])).mean()))

# ---- 시간 곡선 (ext_scan_s.py 와 동일한 부분상관 t)
def design(g, pos):
    u = np.unique(g)
    return np.column_stack([(g == v).astype(float) for v in u] + [pos - pos.mean()] +
                           [(g == v) * (pos - pos[g == v].mean()) for v in u])


key = ('PFC', 'hbeta')
tc = []
for lab, sel in CELLS:
    sel = sel.values
    m = meta[sel]; gg = m.Mouse_ID.values; pp = m.pos.values; yy = m.follow.values.astype(float)
    A = X[key][sel]
    for b in range(len(T)):
        av = np.isfinite(A[:, b])
        if av.sum() < 25 or len(np.unique(gg[av])) < len(np.unique(gg)):
            tc.append(dict(cell=lab, t=float(T[b]), tstat=np.nan, n=int(av.sum()))); continue
        Dm = design(gg[av], pp[av]); R_ = np.eye(av.sum()) - Dm @ np.linalg.pinv(Dm)
        yr = R_ @ yy[av]; yr = yr / np.linalg.norm(yr)
        xr = R_ @ A[av, b].astype(float); xr = xr / np.linalg.norm(xr)
        r = float(xr @ yr); df = av.sum() - np.linalg.matrix_rank(Dm) - 1
        tc.append(dict(cell=lab, t=float(T[b]), tstat=r * np.sqrt(df / (1 - r ** 2)), n=int(av.sum())))
TC = pd.DataFrame(tc)
TC.to_csv('ctx_split_timecourse.csv', index=False)
R.to_csv('ctx_split_joint.csv', index=False)
print('\n0.25–2.25 s 평균 t:')
print(TC[(TC.t >= 0.25) & (TC.t < 2.25)].groupby('cell').tstat.mean().round(2).to_string())

# ---- 연속 조절변수: 진입자의 작업률 (P/NP 이분법이 아니라)
WR = {'A1': 0.617, 'A2': 46.296, 'A3': 11.728, 'A4': 0.617, 'A5': 23.457, 'A6': 17.284}
S = meta[~meta.obs_P].dropna(subset=['S_pre', 'S_post']).reset_index(drop=True)
g = S.Mouse_ID.values; pos = S.pos.values
Z = resid_mat(S[['S_pre', 'S_post']].values.astype(float), g, pos); Z = Z / Z.std(0)
y = resid_mat(S[['follow']].values.astype(float), g, pos)[:, 0]
w = np.array([WR[s] for s in S.starter]); w = (w - w.mean()) / w.std()
b0 = fit(w); idx = [np.where(g == u)[0] for u in np.unique(g)]
null = np.empty(NPERM)
for k in range(NPERM):
    wp = w.copy()
    for i in idx:
        wp[i] = w[rng.permutation(i)]
    null[k] = fit(wp)[3]
print('\n== interaction  β × (entrant work rate, continuous), n=%d' % len(S))
print('   coefficient %+.4f/SD, P = %.4f' % (b0[3], (np.abs(null) >= abs(b0[3])).mean()))
print('   entrant work rate in these trials: %s'
      % pd.Series(w).describe()[['min', 'max']].round(2).to_dict())
print('   trials by entrant: %s' % S.starter.value_counts().to_dict())
