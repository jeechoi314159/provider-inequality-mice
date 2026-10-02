# -*- coding: utf-8 -*-
"""R6·R7 (LFP) 데이터 추출 — Fig. 4 패널별 원천을 data/ 로 동결한다.

원천
  powerspect_extract/*.npz : 역할별 대역 파워 시계열 (8 Hz, Start_act·Act_ride)
  data_for_IRC.csv         : 18마리 × 18개 진입 전 스펙트럼 특징, 작업률
  fig3de_utility_data.csv  : 표본 밖 식별 AUC
  ref/ext_*.pkl, ref/scan_clusters.csv, ref/two_signals_*.csv : 개체 내 분석
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 1))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, sys, pickle
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = FIGDATA                           # figure source data
ANA  = (INTER)
REF  = ANA + 'ref/'
PS   = (LFP + 'powerspect_extract/')
REG  = ['PFC', 'NAc', 'BLA']
BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
ROLES = [('worker', 'Retrieved'), ('participant', 'Entered'), ('freerider', 'Stayed out')]

# ============================================================== A 역할별 대역 파워 궤적
def norm_trace(a, n):
    a = a[np.isfinite(a)]
    if len(a) < 4:
        return np.full(n, np.nan)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)

NPT = 40
rows = []
for role, lab in ROLES:
    d1 = np.load(PS + f'ps_Start_act_{role}.npz')
    d2 = np.load(PS + f'ps_Act_ride_{role}.npz')
    n = d1['PFC_trial'].shape[0]
    for g in REG:
        for b in BANDS:
            A = np.vstack([np.concatenate([norm_trace(d1[f'{g}_{b}'][i], NPT),
                                           norm_trace(d2[f'{g}_{b}'][i], NPT)]) for i in range(n)])
            m = np.nanmean(A, axis=0)
            se = np.nanstd(A, axis=0) / np.sqrt(np.sum(np.isfinite(A), axis=0))
            for k in range(2 * NPT):
                rows.append(dict(role=lab, region=g, band=b, x=k, mean=float(m[k]), se=float(se[k]),
                                 epoch='Approach' if k < NPT else 'Retrieval', n_trials=int(n)))
TR = pd.DataFrame(rows)
TR.to_csv(OUT + 'fig4a_band_traces.csv', index=False)

# ============================================== B 회수 대 진입을 가르는 특징 (LDA 계수)
rows = []
for role, lab in ROLES[:2]:                      # worker vs participant
    d1 = np.load(PS + f'ps_Start_act_{role}.npz')
    n = d1['PFC_trial'].shape[0]
    for i in range(n):
        r = dict(label=lab, trial=int(d1['PFC_trial'][i]))
        ok = True
        for g in REG:
            for b in BANDS:
                a = d1[f'{g}_{b}'][i]; a = a[np.isfinite(a)]
                if len(a) < 4:
                    ok = False; break
                r[f'{g}_{b}'] = float(np.mean(a))
            if not ok:
                break
        if ok:
            rows.append(r)
FE = pd.DataFrame(rows).dropna()
FCOLS = [f'{g}_{b}' for g in REG for b in BANDS]
Z = (FE[FCOLS] - FE[FCOLS].mean()) / FE[FCOLS].std(ddof=0)
y = (FE.label == 'Retrieved').astype(int).values
lda = LinearDiscriminantAnalysis(solver='svd').fit(Z.values, y)
co = pd.DataFrame(dict(feature=FCOLS, coef=lda.coef_[0]))
co['abs'] = co.coef.abs()
co = co.sort_values('abs', ascending=False).reset_index(drop=True)
co['n_retrieved'] = int(y.sum()); co['n_entered'] = int((1 - y).sum())
co.to_csv(OUT + 'fig4b_lda_coefficients.csv', index=False)

# ===================================== C 비용–편익 가중치 (elastic net) 와 cohort 제외 안정성
IR = pd.read_csv(ANA + 'data_for_IRC.csv')
F = [c for c in IR.columns if c.startswith('Pre_')]
mice = sorted(IR.Mouse_ID.unique())
Mf = IR.groupby('Mouse_ID')[F].mean().loc[mice]
WR = IR.groupby('Mouse_ID').WR.first().loc[mice]
grp = IR.groupby('Mouse_ID').Group.first().loc[mice]
role = IR.groupby('Mouse_ID').Role.first().loc[mice]


def enet(Mfit, yfit):
    mu, sd = Mfit.mean(), Mfit.std(ddof=0)
    en = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0, max_iter=50000)
    en.fit(((Mfit - mu) / sd).values, yfit)
    return en.coef_, mu, sd


b, mu, sd = enet(Mf, WR.values)
W = pd.DataFrame(dict(feature=F, weight=b))
W['region'] = W.feature.str.split('_').str[1]
W['band'] = W.feature.str.replace('Pre_', '', regex=False).str.split('_', n=1).str[1]
W.to_csv(OUT + 'fig4c_enet_weights.csv', index=False)

rho = []
for gname in sorted(set(grp)):
    keep = [m for m in mice if grp[m] != gname]
    bb, _, _ = enet(Mf.loc[keep], WR.loc[keep].values)
    if np.std(bb) > 0:
        rho.append(stats.spearmanr(b, bb).statistic)
pd.DataFrame(dict(left_out=sorted(set(grp))[:len(rho)], rho=rho)).to_csv(
    OUT + 'fig4c_loco_stability.csv', index=False)

# ============================== D 편익–비용 평면 (표본 내) 과 표본 밖 식별 AUC
Zm = (Mf - mu) / sd
ben = (Zm.values * np.where(b > 0, b, 0)).sum(1)
cost = (Zm.values * np.where(b < 0, -b, 0)).sum(1)
BC = pd.DataFrame(dict(Mouse_ID=mice, Group=grp.values, Role=role.values,
                       benefit=ben, cost=cost, score=ben - cost, work_rate=WR.values))
t = stats.mannwhitneyu(BC.score[BC.Role == 'P'], BC.score[BC.Role == 'NP'])
BC['mwu_P'] = t.pvalue
BC.to_csv(OUT + 'fig4d_benefit_cost.csv', index=False)
AUC = pd.read_csv(ANA + 'fig3de_utility_data.csv')
AUC[AUC.block == 'model'].to_csv(OUT + 'fig4d_heldout_auc.csv', index=False)

# ======================================= E 동료 진입 정렬 within-mouse t 시간 곡선
W1 = pickle.load(open(REF + 'ext_wait_NP.pkl', 'rb'))
W2 = pickle.load(open(REF + 'ext_follow_NP.pkl', 'rb'))
Tg = pickle.load(open(REF + 'ext_data.pkl', 'rb'))['T']
KEYS = {('PFC', 'g51_100'): 'PFC γ (51–100 Hz)', ('PFC', 'hbeta'): 'PFC high β (24–32 Hz)'}
rows = []
for name, D in (('wait', W1), ('follow', W2)):
    for k, lab in KEYS.items():
        i = D['keys'].index(k)
        for j, tt in enumerate(Tg):
            rows.append(dict(analysis=name, key=lab, t=float(tt), tstat=float(D['T0'][i, j]),
                             n=int(D['nb'][j])))
pd.DataFrame(rows).to_csv(OUT + 'fig4e_timecourses.csv', index=False)
SC = pd.read_csv(REF + 'scan_clusters.csv')
g_cl = SC[(SC.block == 'cag') & (SC.outcome == 'logL') & (SC.region == 'PFC') &
          (SC.f_lo == 51)].sort_values('P_family').iloc[0]
SM = pd.read_csv(REF + 'ext_clusters_smoothed.csv')
b_cl = SM[(SM.analysis == 'follow_NP') & (SM.region == 'PFC') &
          (SM.band == 'hbeta')].sort_values('P_family').iloc[0]
pd.DataFrame([dict(panel='gamma', analysis='wait', key='PFC γ (51–100 Hz)',
                   t_start=float(g_cl.t_start), t_end=float(g_cl.t_end),
                   P_family=float(g_cl.P_family), n=int(g_cl.n)),
              dict(panel='beta', analysis='follow', key='PFC high β (24–32 Hz)',
                   t_start=float(b_cl.t_start), t_end=float(b_cl.t_end),
                   P_family=float(b_cl.P_family), n=int(b_cl.n_max))]).to_csv(
    OUT + 'fig4e_clusters.csv', index=False)

# ============================================= F 결합 모형의 부분 효과 (2 × 2)
J = pd.read_csv(REF + 'two_signals_joint.csv')
rows = []
for _, r in J.iterrows():
    out = 'Waits longer' if r.label.startswith('Wait') else 'Follows'
    rows.append(dict(outcome=out, signal='γ before entry', beta=float(r.b_pre), P=float(r.P_pre),
                     n=int(r.n)))
    rows.append(dict(outcome=out, signal='β after entry', beta=float(r.b_post), P=float(r.P_post),
                     n=int(r.n)))
pd.DataFrame(rows).to_csv(OUT + 'fig4f_joint_model.csv', index=False)

# ============================================= G 견고성 (추적 가능한 점검만)
def parse_robust():
    txt = open(REF + 'robust_output.txt').read().splitlines()
    out = {}
    for ln in txt:
        if 'L≥3s' in ln and 'ctrl=True' in ln:
            out['ref'] = (float(ln.split('slope/sd')[1].split()[0]), float(ln.split('P=')[1].split()[0]))
        if 'ctrl=False' in ln and 'L≥3s' in ln:
            out['nosess'] = (float(ln.split('slope/sd')[1].split()[0]), float(ln.split('P=')[1].split()[0]))
        if ln.strip().startswith('LOMO cross-validated (cluster'):
            out['lomo'] = (float(ln.split('slope/sd')[1].split()[0]), float(ln.split('P=')[1].split()[0]))
        if 'L≥5s' in ln and 'ctrl=True' in ln:
            out['long'] = (float(ln.split('slope/sd')[1].split()[0]), float(ln.split('P=')[1].split()[0]))
    return out


R = parse_robust()
ext = {}
for ln in open(REF + 'ext_robust_output.txt').read().splitlines():
    if ln.startswith('E2 follow: PFC high β'):
        ext['beta'] = (float(ln.split('fixed window:')[1].split()[0]),
                       float(ln.split('P=')[1].split()[0]))
    if ln.startswith('E2 follow: PFC high γ'):
        ext['gamma'] = (float(ln.split('fixed window:')[1].split()[0]),
                        float(ln.split('P=')[1].split()[0]))
jw = J[J.label.str.startswith('Wait')].iloc[0]
jf = J[~J.label.str.startswith('Wait')].iloc[0]
rows = [
    dict(signal='γ → waits longer', check='All trials', beta=float(jw.b_pre), P=float(jw.P_pre)),
    dict(signal='γ → waits longer', check='No session control', beta=R['nosess'][0], P=R['nosess'][1]),
    dict(signal='γ → waits longer', check='Held-out mouse', beta=R['lomo'][0], P=R['lomo'][1]),
    dict(signal='γ → waits longer', check='Waits ≥ 5 s only', beta=R['long'][0], P=R['long'][1]),
    dict(signal='γ → waits longer', check='Second dataset', beta=ext['gamma'][0], P=ext['gamma'][1]),
    dict(signal='β → follows', check='All trials', beta=float(jf.b_post), P=float(jf.P_post)),
    dict(signal='β → follows', check='Second dataset', beta=ext['beta'][0], P=ext['beta'][1]),
]
pd.DataFrame(rows).to_csv(OUT + 'fig4g_robustness.csv', index=False)

# ============================================= H 쥐별 계수 (결합 모형)
TS = pd.read_csv(REF + 'two_signals_trials.csv')


def resid(M, g, pos):
    out = np.empty_like(M, dtype=float)
    for u in np.unique(g):
        i = np.where(g == u)[0]
        X = np.column_stack([np.ones(len(i)), pos[i]])
        B = np.linalg.lstsq(X, M[i], rcond=None)[0]
        out[i] = M[i] - X @ B
    return out


rows = []
for name, sel, ycol in (('Waits longer', (TS.follow == 1) & (TS.L >= 3), 'logL'),
                        ('Follows', TS.index == TS.index, 'follow')):
    S = TS[sel].dropna(subset=['S_pre', 'S_post']).copy()
    S = S[S.Mouse_ID.isin(['A1', 'A3', 'A4', 'A5', 'A6'])]
    S['logL'] = np.log10(S.L)
    S = S.dropna(subset=[ycol])
    Z = resid(S[['S_pre', 'S_post']].values.astype(float), S.Mouse_ID.values, S.pos.values)
    Z = Z / Z.std(0)
    yv = resid(S[[ycol]].values.astype(float), S.Mouse_ID.values, S.pos.values)[:, 0]
    for u in sorted(S.Mouse_ID.unique()):
        i = (S.Mouse_ID.values == u)
        B = np.linalg.lstsq(Z[i], yv[i], rcond=None)[0]
        rows.append(dict(outcome=name, mouse=u, beta_gamma=float(B[0]), beta_beta=float(B[1]),
                         n=int(i.sum())))
pd.DataFrame(rows).to_csv(OUT + 'fig4h_per_mouse.csv', index=False)

FILES = ['fig4a_band_traces.csv', 'fig4b_lda_coefficients.csv', 'fig4c_enet_weights.csv',
         'fig4c_loco_stability.csv', 'fig4d_benefit_cost.csv', 'fig4d_heldout_auc.csv',
         'fig4e_timecourses.csv', 'fig4e_clusters.csv', 'fig4f_joint_model.csv',
         'fig4g_robustness.csv', 'fig4h_per_mouse.csv']
pd.DataFrame([dict(file=f, rows=len(pd.read_csv(OUT + f))) for f in FILES]).to_csv(
    OUT + '_sources_r67.csv', index=False)
print(pd.read_csv(OUT + '_sources_r67.csv').to_string(index=False))
print('\nLDA top features:'); print(co.head(6).to_string(index=False))
print('\nenet weights:'); print(W.pivot_table(index='region', columns='band', values='weight').round(2).to_string())
print('LOCO rho median %.2f' % np.median(rho))
print('\nB-C: P = %.5f' % BC.mwu_P.iloc[0])
print(pd.read_csv(OUT + 'fig4e_clusters.csv').to_string(index=False))
