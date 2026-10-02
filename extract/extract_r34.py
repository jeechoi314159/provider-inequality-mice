# -*- coding: utf-8 -*-
"""R3–R4 패널(Fig. 2, fig. S3–S5)의 바탕 자료를 data/ 아래 패널별 파일로 동결한다.
그리기 코드는 data/ 의 파일만 읽으며 원본에는 접근하지 않는다."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 1))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, warnings
import numpy as np, pandas as pd
from scipy import stats, optimize
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
SUB  = os.path.dirname(HERE)
DATA = FIGDATA.rstrip(_os.sep)          # figure source data
ANA  = (INTER[:-1])
RAW  = (RAWROOT[:-1])
os.makedirs(DATA, exist_ok=True)
prov_log = {}

def out(name, df, note=''):
    df.to_csv(os.path.join(DATA, name), index=False)
    prov_log[name] = {'rows': int(len(df)), 'note': note}
    print(f'  {name:42s} {len(df):5d} rows  {note}')

def find(root, name):
    for d, _, fs in os.walk(root):
        if name in fs: return os.path.join(d, name)
    raise FileNotFoundError(name)

SRC = find(RAW, '[Submission]Source Data_260521.xlsx')
def sheet(name):
    raw = pd.read_excel(SRC, sheet_name=name, header=None)
    key = raw[0].astype(str).str.strip()
    idx = raw.index[key.isin(['Group', 'Mouse'])]
    return pd.read_excel(SRC, sheet_name=name, header=int(idx[0]) if len(idx) else 0)

# ---------------------------------------------------------------- 공통 시행 자료
F = pd.read_csv(os.path.join(ANA, 'follower_retrieval_latency.csv'))
W = pd.read_csv(os.path.join(ANA, 'fig2a_work_rate_data.csv'))
PROVM = {c: (g.pivot_table(index='Trial', columns='Mouse_ID', values='Worked', aggfunc='max')
             .fillna(0).mean()).idxmax() for c, g in W.groupby('Group')}
COH = sorted(PROVM)
F['is_provider'] = [m == PROVM[g] for m, g in zip(F.Mouse_ID, F.Group)]
F['T'] = F.groupby('Group').Trial.transform('max')
F['pos'] = (F.Trial - 1) / (F['T'] - 1)          # 세션 내 상대 위치 0–1
F['role'] = np.where(F.is_provider, 'provider', 'other')

# ---------------------------------------------------------------- Fig. 2A · fig. S3A–C
print('Fig. 2A · fig. S3A–C  경쟁 위험 회수율의 Shapley 분해')
def halves(c):
    g = F[F.Group == c]
    tr = g.groupby('Trial').agg(pos=('pos', 'first')).reset_index()
    tr['E'] = tr.Trial.map(g[g.Worked == 1].groupby('Trial').RetrievalLatency.min())
    rows = {}
    for lo, hi, lab in ((0, 0.5, 'early'), (0.5, 1.01, 'late')):
        tt = tr[(tr.pos >= lo) & (tr.pos < hi)]
        E = float(tt.E.sum()); gw = g[g.Trial.isin(tt.Trial) & (g.Worked == 1)]
        wP = int(gw.is_provider.sum()); wN = int((~gw.is_provider).sum())
        rows[lab] = dict(E=E, n_trials=len(tt), wP=wP, wN=wN,
                         lamP=wP / E * 60, lamN=wN / E * 60)      # 분당 회수율
    return rows

shp = lambda lp, ln: lp / (lp + ln) if (lp + ln) > 0 else np.nan
rows, long = [], []
for c in COH:
    h = halves(c); e, l = h['early'], h['late']
    see, sll = shp(e['lamP'], e['lamN']), shp(l['lamP'], l['lamN'])
    sle, sel = shp(l['lamP'], e['lamN']), shp(e['lamP'], l['lamN'])
    dP = 0.5 * ((sle - see) + (sll - sel))
    dN = 0.5 * ((sel - see) + (sll - sle))
    rows.append(dict(cohort=c, provider=PROVM[c], share_early=see, share_late=sll,
                     delta=sll - see, provider_part=dP, others_part=dN,
                     lamP_early=e['lamP'], lamP_late=l['lamP'],
                     lamN_early=e['lamN'], lamN_late=l['lamN'],
                     n_trials_early=e['n_trials'], n_trials_late=l['n_trials']))
    for lab, d in h.items():
        for who, lam in (('provider', d['lamP']), ('others', d['lamN'])):
            long.append(dict(cohort=c, half=lab, who=who, rate_per_min=lam,
                             n_retrievals=d['wP'] if who == 'provider' else d['wN'],
                             exposure_s=d['E']))
S = pd.DataFrame(rows)
tot, pp, nn = S.delta.sum(), S.provider_part.sum(), S.others_part.sum()
out('fig2a_entrenchment_shapley.csv', S,
    f'8 cohort Shapley 분해: Σ Δshare {tot:.2f} 중 나머지 기여분 {100 * nn / tot:.0f}%')
out('figS03a_rates_by_half.csv', pd.DataFrame(long), 'cohort·반기·역할별 회수율 (분당)과 노출 시간')

# 초기 대 후기 회수율 검정 (cohort 층화 Poisson, 노출 시간 offset)
import statsmodels.api as sm_api
_rt = pd.DataFrame(long); _rows = []
for _who in ('provider', 'others'):
    _d = _rt[_rt.who == _who].copy()
    _d = _d[_d.groupby('cohort').n_retrievals.transform('sum') > 0]
    _d['late'] = (_d.half == 'late').astype(int)
    _m = smf.glm('n_retrievals ~ late + C(cohort)', data=_d,
                 offset=np.log(_d.exposure_s / 60.0),
                 family=sm_api.families.Poisson()).fit()
    _rows.append(dict(who=_who, n_cohorts=_d.cohort.nunique(),
                      rate_ratio=float(np.exp(_m.params['late'])), p=float(_m.pvalues['late'])))
out('fig2a_early_late_test.csv', pd.DataFrame(_rows),
    'cohort 층화 Poisson: ' + '; '.join(f"{r['who']} RR {r['rate_ratio']:.2f}, P = {r['p']:.2g}" for r in _rows))

ratio = S.assign(provider=S.lamP_late / S.lamP_early,
                 others=S.lamN_late / S.lamN_early.replace(0, np.nan))[['cohort', 'provider', 'others']]
gm = lambda v: float(np.exp(np.log(v[v > 0].dropna()).mean()))
out('figS03c_rate_ratios.csv', ratio,
    f'후기/초기 회수율 비: 제공자 기하평균 {gm(ratio.provider):.2f}, '
    f'나머지 {gm(ratio.others):.2f} ({int(ratio.others.notna().sum())} cohort)')

# bootstrap CI (시행 재추출) — cohort별 배열을 미리 만들어 두고 재추출
rng = np.random.default_rng(7)
ARR = {}
for c in COH:
    g = F[F.Group == c]
    tr = g.groupby('Trial').agg(pos=('pos', 'first')).reset_index()
    tr['E'] = tr.Trial.map(g[g.Worked == 1].groupby('Trial').RetrievalLatency.min())
    tr['win'] = tr.Trial.map(g[g.Worked == 1].groupby('Trial').is_provider.max())
    tr = tr.dropna(subset=['E', 'win'])
    ARR[c] = {lab: (tr[(tr.pos >= lo) & (tr.pos < hi)].E.values,
                    tr[(tr.pos >= lo) & (tr.pos < hi)].win.values.astype(float))
              for lo, hi, lab in ((0, 0.5, 'early'), (0.5, 1.01, 'late'))}
boot = []
for b in range(2000):
    tP = tN = 0.0
    for c in COH:
        acc = {}
        for lab in ('early', 'late'):
            E, wn = ARR[c][lab]
            i = rng.integers(0, len(E), len(E))
            Es, ws = E[i], wn[i]
            tot_E = Es.sum()
            acc[lab] = (ws.sum() / tot_E, (len(ws) - ws.sum()) / tot_E)
        (lpe, lne), (lpl, lnl) = acc['early'], acc['late']
        see, sll = shp(lpe, lne), shp(lpl, lnl)
        sle, sel = shp(lpl, lne), shp(lpe, lnl)
        if np.isnan(see) or np.isnan(sll): continue
        tP += 0.5 * ((sle - see) + (sll - sel)); tN += 0.5 * ((sel - see) + (sll - sle))
    boot.append((tP, tN))
B = pd.DataFrame(boot, columns=['provider_part', 'others_part'])
B['others_share_of_delta'] = B.others_part / (B.provider_part + B.others_part)
q = B.others_share_of_delta.quantile([0.025, 0.975]).values
out('figS03b_shapley_bootstrap.csv', B,
    f'시행 재추출 2000회: 나머지 기여 비율 {100 * nn / tot:.0f}% (95% CI {100 * q[0]:.0f}–{100 * q[1]:.0f}%)')

# ---------------------------------------------------------------- Fig. 2B · 2E 궤적
print('Fig. 2B · 2E  역할별 진입·회수 잠복기 궤적')
def traj(col, mask, stem, label):
    d = F[mask & F[col].notna() & (F[col] > 0)].copy()
    d['lat'] = np.log(d[col]); d['isP'] = d.is_provider.astype(int)     # 자연로그 (본문과 같은 단위)
    m = smf.mixedlm('lat ~ pos*isP', d, groups=d.Mouse_ID).fit()
    sl_o, sl_p = m.params['pos'], m.params['pos'] + m.params['pos:isP']
    fit = pd.DataFrame([dict(role=r, intercept=m.params['Intercept'] + (m.params['isP'] if r == 'provider' else 0),
                             slope_ln=(sl_p if r == 'provider' else sl_o),
                             se_ln=(m.bse['pos:isP'] if r == 'provider' else m.bse['pos']))
                        for r in ('other', 'provider')])
    cv = m.cov_params()
    var_p = (cv.loc['pos', 'pos'] + cv.loc['pos:isP', 'pos:isP'] + 2 * cv.loc['pos', 'pos:isP'])
    p_prov = 2 * stats.norm.sf(abs(sl_p / np.sqrt(var_p)))
    fit['slope_p'] = [m.pvalues['pos'], p_prov]          # 역할별 기울기가 0 과 다른지
    fit['interaction_p'] = m.pvalues['pos:isP']; fit['slope_p_others'] = m.pvalues['pos']
    fit['n_trials'] = len(d); fit['n_mice'] = d.Mouse_ID.nunique()
    out(f'{stem}_trials.csv',
        d[['Group', 'Mouse_ID', 'Trial', 'pos', 'role', col]]
        .rename(columns={'Group': 'cohort', 'Mouse_ID': 'mouse', 'Trial': 'trial', col: 'latency_s'}),
        f'{label} 시행 단위 ({len(d)}시행, {d.Mouse_ID.nunique()}마리)')
    out(f'{stem}_fit.csv', fit,
        f'{label} LMM ln: 나머지 기울기 {sl_o:+.2f}, 제공자 {sl_p:+.2f}; 역할×시행 P = {m.pvalues["pos:isP"]:.2g}')
    return m

traj('EntryLatency', F.Entered == 1, 'fig2b_entry_latency', '진입 잠복기')
traj('RetrievalLatency', F.Worked == 1, 'fig2e_retrieval_latency', '회수 잠복기')

# ---------------------------------------------------------------- Fig. 2D 추종 잠복기
print('Fig. 2D  선발자별 추종 잠복기 LMM')
ST = pd.read_csv(os.path.join(ANA, 'starter_labels.csv'))
ST['is_provider'] = [m == PROVM[g] for m, g in zip(ST.Mouse_ID, ST.Group)]
ST['T'] = ST.groupby('Group').Trial.transform('max')
ST['pos'] = (ST.Trial - 1) / (ST['T'] - 1)
ST['starter_latency_s'] = ST.groupby(['Group', 'Trial']).EntryLatency.transform('min')
ST['follow_delay_s'] = ST.EntryLatency - ST.starter_latency_s      # 선발자 진입 이후의 지연
fol = ST[(ST.TrialRole == 'Follower') & (ST.follow_delay_s > 0)].copy()
fol['lat'] = np.log(fol.follow_delay_s)
STRATA = [('Provider started, others follow', (fol.StarterRole == 'P') & (~fol.is_provider)),
          ('Another mouse started, others follow', (fol.StarterRole != 'P') & (~fol.is_provider)),
          ('Another mouse started, provider follows', (fol.StarterRole != 'P') & (fol.is_provider))]
rows = []
for lab, msk in STRATA:
    d = fol[msk]
    m = smf.mixedlm('lat ~ pos', d, groups=d.Mouse_ID).fit()
    b, se = m.params['pos'], m.bse['pos']
    rows.append(dict(stratum=lab, slope_ln=b, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                     p=m.pvalues['pos'], n_trials=len(d), n_mice=d.Mouse_ID.nunique()))
F2D = pd.DataFrame(rows)
out('fig2d_follower_lmm.csv', F2D,
    '; '.join(f'{r.stratum[:16]} β={r.slope_ln:.2f} P={r.p:.2g}' for r in F2D.itertuples()))
out('fig2d_follower_trials.csv',
    fol.assign(stratum=np.select([m for _, m in STRATA], [l for l, _ in STRATA], 'other'))
       [['Group', 'Mouse_ID', 'Trial', 'pos', 'stratum', 'follow_delay_s']]
       .rename(columns={'Group': 'cohort', 'Mouse_ID': 'mouse', 'Trial': 'trial'}),
    f'추종 시행 {len(fol)}건')

# ---------------------------------------------------------------- Fig. 2C · fig. S4 경주 모형
print('Fig. 2C · fig. S4  경주 모형 (강화 대 철수)')
def cohort_race(c):
    g = F[F.Group == c]
    tr = g.groupby('Trial').agg(pos=('pos', 'first')).reset_index()
    tr['E'] = tr.Trial.map(g[g.Worked == 1].groupby('Trial').RetrievalLatency.min())
    tr['winP'] = tr.Trial.map(g[g.Worked == 1].groupby('Trial').is_provider.max())
    tr = tr.dropna(subset=['E', 'winP']).sort_values('Trial').reset_index(drop=True)
    T = len(tr)
    cP = np.r_[0, np.cumsum(tr.winP.values.astype(float))[:-1]] / T
    cN = np.r_[0, np.cumsum(1 - tr.winP.values.astype(float))[:-1]] / T
    return tr.E.values, tr.winP.values.astype(bool), cP, cN, tr.pos.values, tr.Trial.values

def nll(theta, E, win, cP, cN, use_a, use_b):
    aP, aN = theta[0], theta[1]
    al = theta[2] if use_a else 0.0
    be = theta[2 + int(use_a)] if use_b else 0.0
    lP = np.exp(aP + al * cP)
    lN = np.exp(aN + al * cN - be * cP)
    lw = np.where(win, lP, lN)
    return -np.sum(np.log(np.maximum(lw, 1e-300)) - (lP + lN) * E)

MODELS = [('Constant', False, False), ('Reinforcement', True, False),
          ('Withdrawal', False, True), ('Both', True, True)]
fitrows, trialrows = [], []
for c in COH:
    E, win, cP, cN, pos, trial_no = cohort_race(c)
    if len(E) < 8 or win.all() or (~win).all():
        base = None
    for name, ua, ub in MODELS:
        k = 2 + int(ua) + int(ub)
        x0 = np.r_[np.log(max(win.mean(), 1e-3) / E.mean()), np.log(max((~win).mean(), 1e-3) / E.mean()),
                   np.zeros(k - 2)]
        r = optimize.minimize(nll, x0, args=(E, win, cP, cN, ua, ub), method='Nelder-Mead',
                              options=dict(maxiter=8000, xatol=1e-6, fatol=1e-6))
        ll = -r.fun; aic = 2 * k - 2 * ll
        th = r.x
        al = th[2] if ua else 0.0
        be = th[2 + int(ua)] if ub else 0.0
        lP = np.exp(th[0] + al * cP); lN = np.exp(th[1] + al * cN - be * cP)
        fitrows.append(dict(cohort=c, model=name, alpha=al, beta=be, logLik=ll, k=k, aic=aic,
                            n_trials=len(E)))
        if name == 'Both':          # fig. S4, C·D 용 시행 단위 표
            for q in range(len(E)):
                trialrows.append(dict(cohort=c, trial=int(trial_no[q]), pos=float(pos[q]),
                                      exposure_s=float(E[q]),
                                      winner='provider' if win[q] else 'others',
                                      cum_provider=float(cP[q]), cum_others=float(cN[q]),
                                      lambda_provider=float(lP[q] * 60),
                                      lambda_others=float(lN[q] * 60)))
FIT = pd.DataFrame(fitrows)
out('figS04a_trial_table.csv', pd.DataFrame(trialrows),
    '시행 단위 경주 자료와 전체 모형(α·β) 적합 위험률 (분당)')
aic = FIT.groupby('model').aic.sum().sort_values()
out('figS04d_model_comparison.csv', FIT,
    'AIC 합: ' + ', '.join(f'{m} {v:.0f}' for m, v in aic.items()))

# 모수 회복 (모형이 참일 때 α·β 를 되찾는지)
rng2 = np.random.default_rng(3)
rec = []
for rep in range(60):
    a_true, b_true = rng2.uniform(-1.5, 1.5), rng2.uniform(-0.5, 3.0)
    E, win, cP, cN, _ = cohort_race('A')
    T = len(E)
    cPs = np.zeros(T); cNs = np.zeros(T); w = np.zeros(T, bool); Es = np.zeros(T)
    aP, aN = -1.0, -0.5
    nP = nN = 0
    for t in range(T):
        cPs[t], cNs[t] = nP / T, nN / T
        lP = np.exp(aP + a_true * cPs[t]); lN = np.exp(aN + a_true * cNs[t] - b_true * cPs[t])
        Es[t] = rng2.exponential(1 / (lP + lN))
        w[t] = rng2.random() < lP / (lP + lN)
        nP += int(w[t]); nN += int(not w[t])
    r = optimize.minimize(nll, np.r_[aP, aN, 0.0, 0.0], args=(Es, w, cPs, cNs, True, True),
                          method='Nelder-Mead', options=dict(maxiter=8000))
    rec.append(dict(alpha_true=a_true, beta_true=b_true, alpha_fit=r.x[2], beta_fit=r.x[3]))
REC = pd.DataFrame(rec)
out('figS04b_parameter_recovery.csv', REC,
    f'모의 60회: α 상관 {REC.alpha_true.corr(REC.alpha_fit):.2f}, β 상관 {REC.beta_true.corr(REC.beta_fit):.2f}')
out('figS04c_fitted_parameters.csv', FIT[FIT.model == 'Both'][['cohort', 'alpha', 'beta', 'n_trials']],
    'cohort별 α(강화)·β(철수) 추정치')

# ---------------------------------------------------------------- Fig. 2F·2G, fig. S5 혼자 대 집단
print('Fig. 2F · 2G · fig. S5  혼자 대 집단')
def cond_sheet(name, valcol, role):
    d = sheet(name)
    d.columns = ['cohort', 'mouse_n', 'value', 'condition', 'role_src']
    d = d.dropna(subset=['value']).copy()
    d['mouse'] = d.cohort.astype(str) + d.mouse_n.astype(int).astype(str)
    d['condition'] = d.condition.astype(str).str.strip()
    d['role'] = role
    return d[['cohort', 'mouse', 'role', 'condition', 'value']].rename(columns={'value': valcol})

EN = pd.concat([cond_sheet('Extended Fig 1b', 'entry_latency_s', 'provider'),
                cond_sheet('Extended Fig 1c', 'entry_latency_s', 'other')], ignore_index=True)
out('fig2f_entry_solitary_vs_group.csv', EN,
    f'혼자 대 집단 진입 잠복기 시행 단위 ({len(EN)}건, {EN.mouse.nunique()}마리)')
st = {}
for role in ('other', 'provider'):
    a = EN[(EN.role == role) & (EN.condition == 'Solitary')].entry_latency_s
    b = EN[(EN.role == role) & (EN.condition == 'Group')].entry_latency_s
    st[role] = stats.mannwhitneyu(a, b)
RT = pd.concat([cond_sheet('Extended Fig 1d', 'retrieval_latency_s', 'provider'),
                cond_sheet('Extended Fig 1e', 'retrieval_latency_s', 'other')], ignore_index=True)
out('figS05b_retrieval_solitary_vs_group.csv', RT,
    f'혼자 대 집단 회수 잠복기 시행 단위 ({len(RT)}건)')
out('fig2f_entry_stats.csv',
    pd.DataFrame([dict(role=r, U=float(st[r].statistic), p=float(st[r].pvalue),
                       n_solitary=int((EN.role == r).sum() and ((EN.role == r) & (EN.condition == 'Solitary')).sum()),
                       n_group=int(((EN.role == r) & (EN.condition == 'Group')).sum())) for r in st]),
    f'진입 잠복기 혼자 대 집단: 나머지 P = {st["other"].pvalue:.2g}, 제공자 P = {st["provider"].pvalue:.2g}')

KE = sheet('Extended Fig 1a')
KE.columns = ['mouse_n', 'trial', 'condition', 'kinetic_energy']
KE['condition'] = KE.condition.astype(str).str.strip()
KE['mouse'] = 'A' + KE.mouse_n.astype(int).astype(str)
uk = stats.mannwhitneyu(KE[KE.condition == 'Solitary'].kinetic_energy,
                        KE[KE.condition == 'Group'].kinetic_energy)
out('fig2g_movement_solitary_vs_group.csv', KE[['mouse', 'trial', 'condition', 'kinetic_energy']],
    f'cohort A 운동 에너지 {len(KE)}시행; 혼자 대 집단 P = {uk.pvalue:.2f}')

# 혼자 시행 동안의 추세 (피로·포만 배제)
sol = pd.read_csv(os.path.join(DATA, 'fig1b_solitary_latency.csv'))
tr_rows = []
for col, lab in (('entry_latency_s', 'Entry'), ('total_latency_s', 'Retrieval')):
    r = stats.linregress(sol.trial, np.log(sol[col].replace(0, np.nan).dropna()))
    tr_rows.append(dict(measure=lab, slope_ln_per_trial=r.slope, p=r.pvalue, n=len(sol)))
out('figS05c_solitary_trend.csv', pd.DataFrame(tr_rows),
    '혼자 18시행 동안 추세: ' + ', '.join(f'{r["measure"]} P = {r["p"]:.2f}' for r in tr_rows))

# ---------------------------------------------------------------- fig. S3D–F 전이와 이탈
print('fig. S3D–F  시행 간 전이와 이탈 형태')
F['state'] = np.where(F.Worked == 1, 'retrieved', np.where(F.Entered == 1, 'entered', 'stayed out'))
rows = []
for role in ('provider', 'other'):
    d = F[F.role == role].sort_values(['Mouse_ID', 'Trial'])
    nxt = d.groupby('Mouse_ID').state.shift(-1)
    t = pd.crosstab(d.state, nxt, normalize='index')
    for a in t.index:
        for b in t.columns:
            rows.append(dict(role=role, from_state=a, to_state=b, prob=float(t.loc[a, b]),
                             n_from=int((d.state == a).sum())))
TR = pd.DataFrame(rows)
p_again = {r: float(TR[(TR.role == r) & (TR.from_state == 'retrieved') &
                       (TR.to_state == 'retrieved')].prob.iloc[0]) for r in ('provider', 'other')}
out('figS03de_transitions.csv', TR,
    f'회수 후 다시 회수할 확률: 제공자 {p_again["provider"]:.2f}, 나머지 {p_again["other"]:.2f}')

KEt = pd.read_csv(os.path.join(DATA, 'figS02h_kinetic_energy.csv'))
KEt = KEt[KEt.state == 'stayed out'].copy()
KEt['T'] = KEt.groupby('mouse').trial.transform('max')
KEt['pos'] = (KEt.trial - 1) / (KEt['T'] - 1)
KEt['half'] = np.where(KEt.pos < 0.5, 'early', 'late')
ue = stats.mannwhitneyu(KEt[KEt.half == 'early'].ke_foraging.dropna(),
                        KEt[KEt.half == 'late'].ke_foraging.dropna())
out('figS03f_stayout_engagement.csv', KEt[['mouse', 'trial', 'pos', 'half', 'ke_foraging']],
    f'미진입 시행의 운동 에너지: 초기 대 후기 P = {ue.pvalue:.2g} '
    f'(중앙값 {KEt[KEt.half == "early"].ke_foraging.median():.1e} → {KEt[KEt.half == "late"].ke_foraging.median():.1e})')

prov = pd.DataFrame([dict(file=k, rows=v['rows'], note=v['note']) for k, v in prov_log.items()])
prov.to_csv(os.path.join(DATA, '_sources_r34.csv'), index=False)
print('_sources_r34.csv 갱신:', len(prov), '개 파일')
