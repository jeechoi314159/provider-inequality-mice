# -*- coding: utf-8 -*-
"""R1–R2 패널(Fig. 1 A–E, fig. S1–S3)의 바탕 자료를 data/ 아래 패널별 파일로 동결한다.
그리기 코드는 data/ 의 파일만 읽으며 원본에는 접근하지 않는다."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 1))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, glob, json
import numpy as np, pandas as pd
import scipy.io as sio
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SUB  = os.path.dirname(HERE)
DATA = FIGDATA.rstrip(_os.sep)          # figure source data
ANA  = (INTER[:-1])
RAW  = (RAWROOT[:-1])
os.makedirs(DATA, exist_ok=True)
prov_log = {}

def out(name, df, note=''):
    p = os.path.join(DATA, name)
    df.to_csv(p, index=False)
    prov_log[name] = {'rows': int(len(df)), 'note': note}
    print(f'  {name:38s} {len(df):5d} rows  {note}')

# ---------------------------------------------------------------- 공통: 집단 시행
W = pd.read_csv(os.path.join(ANA, 'fig2a_work_rate_data.csv'))
COH = sorted(W.Group.unique())
PIV = {c: W[W.Group == c].pivot_table(index='Trial', columns='Mouse_ID',
                                      values='Worked', aggfunc='max').fillna(0).astype(int)
       for c in COH}
PROVM = {c: (PIV[c].sum() / len(PIV[c])).idxmax() for c in COH}

# ---------------------------------------------------------------- Fig. 1B / fig. S2A
print('Fig. 1B · fig. S2A  혼자 시행 회수 잠복기')
act = sio.loadmat(os.path.join(RAW, 'Data/act latency data/act_latency.mat'),
                  squeeze_me=True, struct_as_record=False)['tau_act']
get = sio.loadmat(os.path.join(RAW, 'Data/act latency data/get_duration.mat'),
                  squeeze_me=True, struct_as_record=False)['tau_get']
rows = []
for i in range(len(act)):
    a = np.asarray(act[i].solitary, dtype=float)
    g = np.asarray(get[i].solitary, dtype=float)
    n = min(a.size, g.size)
    for t in range(n):
        rows.append(dict(mouse=f'A{i+1}', trial=t + 1, entry_latency_s=a[t],
                         retrieval_duration_s=g[t], total_latency_s=a[t] + g[t], retrieved=1))
sol = pd.DataFrame(rows)
kw = stats.kruskal(*[v.total_latency_s.values for _, v in sol.groupby('mouse')])
out('fig1b_solitary_latency.csv', sol,
    f'cohort A 6마리 × 18시행 = {len(sol)}회 전부 회수; Kruskal-Wallis P = {kw.pvalue:.2f}')

# ---------------------------------------------------------------- Fig. 1C 좌: Lorenz
print('Fig. 1C · fig. S2D  기여 Lorenz')
rows, gsum = [], []
for c in COH:
    s = np.sort((PIV[c].sum() / PIV[c].sum().sum()).values)
    x = np.r_[0, np.arange(1, len(s) + 1) / len(s)]
    y = np.r_[0, np.cumsum(s)]
    gini = 1 - np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))
    gsum.append(dict(cohort=c, n_mice=len(s), n_trials=len(PIV[c]), gini=gini))
    for xi, yi in zip(x, y):
        rows.append(dict(cohort=c, cum_frac_mice=xi, cum_share_retrievals=yi))
out('fig1c_lorenz_points.csv', pd.DataFrame(rows), '8 cohort Lorenz 좌표')
G = pd.DataFrame(gsum)
out('fig1c_gini_by_cohort.csv', G,
    f'기여 Gini 평균 {G.gini.mean():.2f} ± {G.gini.std(ddof=1):.2f} (SD)')

# ---------------------------------------------------------------- Fig. 1C 우: 역할별 섭식
print('Fig. 1C · fig. S2E  역할별 섭식 시간')
def find(root, name):
    for d, _, fs in os.walk(root):
        if name in fs: return os.path.join(d, name)
    raise FileNotFoundError(name)
eat = sio.loadmat(find(RAW, 'eat_data.mat'), squeeze_me=True, struct_as_record=False)
T_eat = np.asarray(eat['T_eat'], dtype=float)          # (170, 6) 섭식 시간
tau_eat = np.asarray(eat['tau_eat'], dtype=float)      # (170, 6) 섭식까지 잠복기
role = np.asarray(sio.loadmat(os.path.join(RAW, 'Data/act latency data/role_set4.mat'),
                              squeeze_me=True, struct_as_record=False)['role'])
R = role[:, 3:9]                                       # 2=회수, 1=진입, 0=미진입
best = None
for tag, sl in (('head', slice(0, R.shape[0])), ('tail', slice(T_eat.shape[0] - R.shape[0], None))):
    E = T_eat[sl]
    grp = [E[R == k] for k in (2, 1, 0)]
    p = stats.kruskal(*grp).pvalue
    if best is None or abs(p - 0.20) < abs(best[1] - 0.20): best = (tag, p, sl)
tag, p_align, sl = best
E, L = T_eat[sl], tau_eat[sl]
rows = []
lab = {2: 'retrieved', 1: 'entered', 0: 'stayed out'}
for t in range(R.shape[0]):
    for m in range(6):
        rows.append(dict(trial=t + 1, mouse=f'A{m+1}', role=lab[int(R[t, m])],
                         eating_duration_s=E[t, m], latency_to_eat_s=L[t, m]))
# 시행 단위 재계산은 정렬 확인용이며 패널에는 쓰지 않는다 (그림은 Supplementary Table 4 공표값 사용).
print(f'  (확인) 시행 단위 재계산 정렬 {tag}; 역할 간 Kruskal-Wallis P = {p_align:.2f}')

# ---------------------------------------------------------------- fig. S2F 소비 대 기여 Gini
cA = PIV['A']
share_work = (cA.sum() / cA.sum().sum()).values
def gini(v):
    v = np.sort(np.asarray(v, dtype=float)); n = v.size
    return (2 * np.sum((np.arange(1, n + 1)) * v) / (n * v.sum()) - (n + 1) / n)
cons = np.nansum(E, axis=0)
# ---------------------------------------------------------------- Fig. 1D 좌: 선두 raster
print('Fig. 1D · fig. S3A  단독 선두 raster')
rows = []
for c in COH:
    p = PIV[c]; cum = p.cumsum(); prov = PROVM[c]
    for t in range(min(40, len(p))):
        r = cum.iloc[t]
        top = r.max()
        lone = bool(r[prov] == top and (r == top).sum() == 1 and top > 0)
        joint = bool(r[prov] == top and (r == top).sum() > 1 and top > 0)
        who = ('provider' if p[prov].iloc[t] == 1
               else ('other' if p.iloc[t].sum() > 0 else 'none'))
        rows.append(dict(cohort=c, trial=t + 1, retrieved_by=who,
                         provider_cum=int(r[prov]), best_other_cum=int(r.drop(prov).max()),
                         lead_state='lone' if lone else ('joint' if joint else 'behind')))
lead = pd.DataFrame(rows)
by10 = lead[lead.trial == 10].set_index('cohort')
n_lone = int((by10.lead_state == 'lone').sum()); n_joint = int((by10.lead_state == 'joint').sum())
first10 = lead[lead.trial <= 10].groupby('cohort').retrieved_by.apply(lambda v: (v == 'provider').sum())
out('fig1f_retrieval_raster.csv', lead,
    f'첫 40시행 회수자 표시; 첫 10시행 중 제공자 회수 {first10.min()}–{first10.max()}회 '
    f'(중앙값 {first10.median():.1f}), 시행 10에서 단독 선두 {n_lone}/{len(by10)}, 공동 {n_joint}/{len(by10)}')

# ---------------------------------------------------------------- Fig. 1D 우: 첫 10시행 HHI
print('Fig. 1D · fig. S3B  첫 10시행 집중도')
rng = np.random.default_rng(7)
rows = []
for c in COH:
    p = PIV[c].iloc[:10]; n = p.shape[1]
    k = p.sum().values
    obs = np.sum((k / k.sum()) ** 2) if k.sum() else np.nan
    sim = []
    for _ in range(20000):
        d = rng.integers(0, n, size=int(k.sum()))
        kk = np.bincount(d, minlength=n)
        sim.append(np.sum((kk / kk.sum()) ** 2))
    rows.append(dict(cohort=c, n_mice=n, n_retrievals_first10=int(k.sum()),
                     hhi_observed=obs, hhi_equal_expected=float(np.mean(sim))))
H = pd.DataFrame(rows)
out('fig1d_hhi_first10.csv', H,
    f'관측 {H.hhi_observed.mean():.2f} 대 균등 성향 기대 {H.hhi_equal_expected.mean():.2f}')

# ---------------------------------------------------------------- Fig. 1E: 형질 대 초기 몫
print('Fig. 1E · fig. S3F  형질 대 첫 10시행 몫')
tr = pd.ExcelFile(os.path.join(RAW, 'Data/tube test etc  logistic/data_230918_update.xls')).parse('Sheet1')
tr.columns = ['mouse', 'work_rate_pct', 'rotarod_pct', 'oft_center_pct', 'epm_open_pct',
              'tube_rank_pct', 'ymaze_alternation_pct']
for c in tr.columns[1:]: tr[c] = pd.to_numeric(tr[c], errors='coerce')
tr['role'] = np.where(tr.work_rate_pct > 43, 'provider', 'other')
out('fig1e_trait_table.csv', tr,
    f'형질 검사 {len(tr)}마리 (제공자 {int((tr.role=="provider").sum())})')

share10 = []
for c in COH:
    p = PIV[c].iloc[:10]; s = p.sum(); tot = s.sum()
    for m in p.columns:
        share10.append(dict(cohort=c, mouse=m, first10_share=(s[m] / tot) if tot else 0.0,
                            role='provider' if m == PROVM[c] else 'other'))
S10 = pd.DataFrame(share10)
out('fig1e_first10_share.csv', S10, '8 cohort 46마리 첫 10시행 회수 몫')

def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan, np.nan, np.nan, len(a), len(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)
    g = J * d
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    p = stats.mannwhitneyu(a, b).pvalue
    return g, g - 1.96 * se, g + 1.96 * se, p, na, nb

rows = []
TRAITS = [('tube_rank_pct', 'Social dominance (tube rank)'),
          ('oft_center_pct', 'Open field centre time'),
          ('epm_open_pct', 'Elevated plus maze open arm'),
          ('ymaze_alternation_pct', 'Y-maze alternation'),
          ('rotarod_pct', 'Rotarod latency to fall')]
for col, lab_ in TRAITS:
    g, lo, hi, p, na, nb = hedges_g(tr.loc[tr.role == 'provider', col], tr.loc[tr.role == 'other', col])
    rows.append(dict(measure=lab_, kind='baseline trait', g=g, ci_lo=lo, ci_hi=hi, p=p,
                     n_provider=na, n_other=nb))
g, lo, hi, p, na, nb = hedges_g(S10.loc[S10.role == 'provider', 'first10_share'],
                                S10.loc[S10.role == 'other', 'first10_share'])
rows.append(dict(measure='Fraction of first 10 retrievals', kind='early behaviour', g=g, ci_lo=lo,
                 ci_hi=hi, p=p, n_provider=na, n_other=nb))
F = pd.DataFrame(rows)
out('fig1e_effect_sizes.csv', F, '제공자 대 나머지 Hedges g와 95% CI')

# ---------------------------------------------------------------- fig. S2B/C 회수율 분포·집중
print('fig. S2B · S2C  회수율 분포와 작업 집중')
rows = []
for c in COH:
    s = PIV[c].sum() / len(PIV[c])
    for m, v in s.items():
        rows.append(dict(cohort=c, mouse=m, retrieval_rate=v,
                         role='provider' if m == PROVM[c] else 'other'))
RR = pd.DataFrame(rows)
out('figS02b_retrieval_rate_per_mouse.csv', RR, f'46마리 회수율, 경계 0.43')
rows = []
for c in COH:
    s = PIV[c].sum() / PIV[c].sum().sum()
    rows.append(dict(cohort=c, provider_share=s[PROVM[c]], others_mean_share=s.drop(PROVM[c]).mean()))
SH = pd.DataFrame(rows)
w = stats.wilcoxon(SH.provider_share, SH.others_mean_share)
out('figS02c_share_provider_vs_others.csv', SH, f'cohort 대응 Wilcoxon P = {w.pvalue:.3f}')

# ---------------------------------------------------------------- fig. S3C 세션 진행 집중도
print('fig. S3C  세션 진행에 따른 집중도')
q = pd.read_csv(os.path.join(ANA, 'quintile_concentration.csv'))
out('figS03c_hhi_by_quintile.csv', q, '세션 5분위별 HHI (원본 quintile_concentration.csv)')

# ---------------------------------------------------------------- fig. S3D 연속 회수 bout
print('fig. S3D  연속 회수 bout 길이')
rows = []
for c in COH:
    p = PIV[c]
    for m in p.columns:
        v = p[m].values; run = 0
        for i, x in enumerate(v):
            if x: run += 1
            elif run: rows.append(dict(cohort=c, mouse=m, bout_len=run,
                                       role='provider' if m == PROVM[c] else 'other')); run = 0
        if run: rows.append(dict(cohort=c, mouse=m, bout_len=run,
                                 role='provider' if m == PROVM[c] else 'other'))
B = pd.DataFrame(rows)
out('figS03d_retrieval_bouts.csv', B,
    f"제공자 평균 {B[B.role=='provider'].bout_len.mean():.1f} 대 나머지 {B[B.role=='other'].bout_len.mean():.1f} 시행")

# ---------------------------------------------------------------- fig. S3E cohort D·E 일별
print('fig. S3E  조작 없는 cohort의 일별 최다 회수자')
rows = []
for c in ('D', 'E'):
    p = PIV[c]; nd = 6
    edges = np.array_split(np.arange(len(p)), nd)
    for d, idx in enumerate(edges, 1):
        if len(idx) == 0: continue
        s = p.iloc[idx].sum()
        if s.sum() == 0: top = 'none'
        else: top = s.idxmax()
        rows.append(dict(cohort=c, block=d, n_trials=len(idx), top_retriever=top,
                         is_provider=int(top == PROVM[c])))
# 일별 최다 회수자 패널은 제외됨 (메시지 불명확) — 파일로 남기지 않는다.
print('  (확인) cohort D·E 시행 블록별 최다 회수자', len(rows), '블록')

# ---------------------------------------------------------------- provenance
prov = pd.DataFrame([dict(file=k, rows=v['rows'], note=v['note']) for k, v in prov_log.items()])
prov.to_csv(os.path.join(DATA, '_sources.csv'), index=False)
print('\n_sources.csv 기록:', len(prov), '개 파일')

# ---------------------------------------------------------------- fig. S1B 설계 일정
print('fig. S1B  cohort 구성과 시행 수')
MANIP = {'A': 'group → provider removed → split → reunited',
         'B': 'group → provider removed → split → reunited',
         'C': 'group → provider removed → reunited',
         'D': 'group only', 'E': 'group only', 'F': 'group only',
         'G': 'group only', 'H': 'group only'}
# 전극 수술은 cohort 단위로 전원 시행. cohort D 는 c-Fos 염색용이라 수술하지 않음.
# 분석 개체 수는 세 영역 신호가 모두 쓸 만한 개체 (audit_per_mouse_IRC.csv).
AN = (pd.read_csv(os.path.join(ANA, 'audit_per_mouse_IRC.csv'))
      .groupby('Group').Mouse_ID.nunique().to_dict())
# 집단 먹이 활동 일수: A–D 는 실험 기록(2025-02-04830 Supplementary Table 1),
# E–H 는 프리프린트 이후 측정한 코호트로 모두 5일.
DAYS = {'A': 14, 'B': 13, 'C': 16, 'D': 7, 'E': 5, 'F': 5, 'G': 5, 'H': 5}
rows = []
for c in COH:
    n = PIV[c].shape[1]; na = int(AN.get(c, 0)); P = PIV[c]
    prov_tr = int(P[PROVM[c]].sum())
    any_tr = int((P.sum(axis=1) > 0).sum())
    rows.append(dict(cohort=c, n_mice=n, n_trials=len(P), n_days=DAYS[c], manipulation=MANIP[c],
                     n_implanted=(0 if na == 0 else n), n_analysed=na,
                     provider=PROVM[c], provider_rate=float(P[PROVM[c]].mean()),
                     trials_provider_retrieved=prov_tr,
                     trials_other_retrieved=any_tr - prov_tr,
                     trials_no_retrieval=len(P) - any_tr))
S = pd.DataFrame(rows)
out('figS01b_cohort_schedule.csv', S,
    f'8 cohort {S.n_mice.sum()}마리 {S.n_trials.sum()}시행 {S.n_days.sum()}일; 수술 {S.n_implanted.sum()}마리, 분석 {S.n_analysed.sum()}마리')

# ---------------------------------------------------------------- fig. S1D 행동 상태 raster
print('fig. S1D  한 session의 행동 상태')
rows = []
for t in range(R.shape[0]):
    for m in range(6):
        rows.append(dict(trial=t + 1, mouse=f'A{m+1}', state=lab[int(R[t, m])]))
out('figS01d_state_raster_cohortA.csv', pd.DataFrame(rows),
    'cohort A 162시행 × 6마리 행동 상태 (회수·진입·미진입)')

# ---------------------------------------------------------------- fig. S2H 운동 에너지
print('fig. S2H  역할별 운동 에너지')
kp = os.path.join(RAW, 'Data/Kinetic energy /Set4_Kinetic_energy.xlsx')
if os.path.exists(kp):
    def sheet(n):
        d = pd.read_excel(kp, sheet_name=n)
        d.columns = ['trial'] + [f'A{i}' for i in range(1, 7)]
        return d.melt('trial', var_name='mouse', value_name=n)
    KE = sheet('kinetic energy group_foraging')
    KE.columns = ['trial', 'mouse', 'ke_foraging']
    beh = pd.read_csv(os.path.join(ANA, 'follower_retrieval_latency.csv'))
    A = beh[beh.Group == 'A'].rename(columns={'Trial': 'trial', 'Mouse_ID': 'mouse'})
    Dk = KE.merge(A[['trial', 'mouse', 'Entered', 'Worked']], on=['trial', 'mouse'], how='inner')
    Dk['state'] = np.where(Dk.Worked == 1, 'retrieved',
                           np.where(Dk.Entered == 1, 'entered', 'stayed out'))
    out('figS02h_kinetic_energy.csv', Dk[['trial', 'mouse', 'state', 'ke_foraging']].dropna(),
        'cohort A 시행별 운동 에너지 (가속도계)')
else:
    print('  (운동 에너지 원본 없음 — 패널 보류)')

# ---------------------------------------------------------------- fig. S1B: 마우스별 회수 수
print('fig. S1B  cohort별 마우스별 회수 수')
rows = []
for c in COH:
    per = PIV[c].sum().sort_values(ascending=False)
    tot = int(per.sum())
    for k, (m, n) in enumerate(per.items()):
        rows.append(dict(cohort=c, mouse=m, rank=k + 1, n_retrieved=int(n),
                         share_of_cohort_retrievals=float(n) / tot,
                         is_provider=int(m == PROVM[c])))
out('figS01b_retrievals_per_mouse.csv', pd.DataFrame(rows),
    '마우스별 회수 시행 수와 cohort 내 비중 (회수 순 정렬)')

# ---------------------------------------------------------------- fig. S2A: 개체별 혼자 대 집단
print('fig. S2A  cohort A 개체별 혼자 대 집단 비교')
rows = []
PA = PIV['A']
for i in range(len(act)):
    m = f'A{i+1}'
    sa = np.atleast_1d(np.asarray(act[i].solitary, dtype=float))
    sg = np.atleast_1d(np.asarray(get[i].solitary, dtype=float))
    ga = np.atleast_1d(np.asarray(act[i].group, dtype=float))
    gg = np.atleast_1d(np.asarray(get[i].group, dtype=float))
    n = min(sa.size, sg.size); tot_s = sa[:n] + sg[:n]
    ng = min(ga.size, gg.size); tot_g = ga[:ng] + gg[:ng]
    rows.append(dict(mouse=m, role=('provider' if m == PROVM['A'] else 'other'),
                     n_solitary_trials=int(n), n_solitary_retrieved=int(n),
                     rate_solitary=1.0,
                     median_latency_solitary_s=float(np.median(tot_s)),
                     n_group_trials=int(len(PA)),
                     n_group_retrieved=int(PA[m].sum()),
                     rate_group=float(PA[m].mean()),
                     median_latency_group_s=(float(np.median(tot_g)) if ng else np.nan)))
SV = pd.DataFrame(rows)
wr = stats.wilcoxon(SV.rate_solitary, SV.rate_group)
wl = stats.wilcoxon(SV.median_latency_solitary_s, SV.median_latency_group_s)
out('figS02a_solitary_vs_group_per_mouse.csv', SV,
    f'cohort A 6마리: 혼자 18/18 회수, 집단 회수율 Wilcoxon P = {wr.pvalue:.3f}; '
    f'회수 잠복기 중앙값 6/6 증가, Wilcoxon P = {wl.pvalue:.3f}')

# ---------------------------------------------------------------- fig. S2N: 먹이 활동 전후 서열
print('fig. S2N  집단 먹이 활동 전후 tube test 서열')
SRC = find(RAW, '[Submission]Source Data_260521.xlsx')
def sheet(name):
    raw = pd.read_excel(SRC, sheet_name=name, header=None)
    hdr = raw.index[raw[0].astype(str).str.strip() == 'Group'][0]
    return pd.read_excel(SRC, sheet_name=name, header=hdr)
DM = sheet('Extended Fig 3c')
DM.columns = ['cohort', 'mouse_n', 'dominance_before_pct', 'dominance_after_pct', 'difference', 'role_src']
DM = DM.dropna(subset=['dominance_before_pct', 'dominance_after_pct']).copy()
DM['role'] = np.where(DM.role_src.astype(str).str.strip() == 'P', 'provider', 'other')
DM['mouse'] = DM.cohort.astype(str) + DM.mouse_n.astype(int).astype(str)
w_all = stats.wilcoxon(DM.dominance_before_pct, DM.dominance_after_pct)
out('figS02n_dominance_before_after.csv',
    DM[['cohort', 'mouse', 'role', 'dominance_before_pct', 'dominance_after_pct']],
    f'{DM.cohort.nunique()} cohort {len(DM)}마리 전후 서열 점수; Wilcoxon P = {w_all.pvalue:.2f} '
    f'(프리프린트: cohort A W = 3, P = 0.84; cohort C W = 10, P = 0.13)')

# ---------------------------------------------------------------- Fig. 1H: 체중 효과 크기 추가
print('Fig. 1H  체중 효과 크기')
BW = sheet('Extended Fig 3h')
BW.columns = ['cohort', 'mouse_n', 'body_weight_g', 'role_src']
BW = BW.dropna(subset=['body_weight_g']).copy()
BW['role'] = np.where(BW.role_src.astype(str).str.strip() == 'P', 'provider', 'other')
bp = BW.loc[BW.role == 'provider', 'body_weight_g'].values
bo = BW.loc[BW.role == 'other', 'body_weight_g'].values
g_bw, lo_bw, hi_bw, p_bw, na_bw, nb_bw = hedges_g(bp, bo)
ES = pd.read_csv(os.path.join(DATA, 'fig1e_effect_sizes.csv'))
ES = ES[ES.measure != 'Body weight'].reset_index(drop=True)
row = pd.DataFrame([dict(measure='Body weight', kind='baseline trait', g=g_bw, ci_lo=lo_bw,
                         ci_hi=hi_bw, p=p_bw, n_provider=na_bw, n_other=nb_bw)])
ES = pd.concat([ES.iloc[:-1], row, ES.iloc[-1:]], ignore_index=True)
out('fig1e_effect_sizes.csv', ES,
    f'사전 형질 + 첫 10시행 몫 효과 크기; 체중 Hedges g = {g_bw:.2f}, P = {p_bw:.2f}')
out('figS02m_body_weight.csv', BW[['cohort', 'mouse_n', 'role', 'body_weight_g']],
    f'{len(BW)}마리 기저 체중')

# ---------------------------------------------------------------- fig. S2H: 고정 먹이 대조
print('fig. S2H  고정 대 이동 먹이 회수율')
FX = sheet('Extended Fig 1g')
FX.columns = ['cohort', 'mouse_n', 'trial', 'role_src', 'condition', 'retrieved']
FX['condition'] = FX.condition.astype(str).str.strip().replace({'Non-fixed': 'Moving', 'Fixed': 'Fixed'})
FX['mouse'] = FX.cohort.astype(str) + FX.mouse_n.astype(int).astype(str)
FX['role'] = np.where(FX.role_src.astype(str).str.strip() == 'P', 'provider', 'other')
per = (FX.groupby(['mouse', 'role', 'condition']).retrieved.mean().unstack('condition').reset_index())
per = per.dropna(subset=['Fixed', 'Moving'])
wf = stats.wilcoxon(per.Fixed, per.Moving)
out('figS02h_fixed_vs_moving.csv',
    per.rename(columns={'Fixed': 'rate_fixed', 'Moving': 'rate_moving'}),
    f'{len(per)}마리 (cohort A·B) 고정 {per.Fixed.mean():.2f} 대 이동 {per.Moving.mean():.2f}; '
    f'Wilcoxon P = {wf.pvalue:.3f}')

# ---------------------------------------------------------------- fig. S2I: 닫힌 칸 결과 분포
print('fig. S2I  닫힌 칸에서의 섭식 대 반출')
CB = sheet('Extended Fig 1i')
CB.columns = ['cohort', 'mouse_n', 'trial', 'consumed_inside', 'behaviour']
CB['behaviour'] = CB.behaviour.astype(str).str.strip()
cnt = CB.behaviour.value_counts().rename_axis('outcome').reset_index(name='n_trials')
cnt['share'] = cnt.n_trials / cnt.n_trials.sum()
out('figS02i_chamber_outcome.csv', cnt,
    f'{int(cnt.n_trials.sum())}시행: ' + ', '.join(f'{r.outcome} {r.n_trials}' for r in cnt.itertuples()))

prov = pd.DataFrame([dict(file=k, rows=v['rows'], note=v['note']) for k, v in prov_log.items()])
prov.to_csv(os.path.join(DATA, '_sources.csv'), index=False)
print('_sources.csv 갱신:', len(prov), '개 파일')

# ---------------------------------------------------------------- Fig. 1B 우: 8 cohort 초기 진입
print('Fig. 1B  8 cohort 첫 10시행 진입 잠복기')
FL = pd.read_csv(os.path.join(ANA, 'follower_retrieval_latency.csv'))
FL['role'] = np.where([PROVM[g] == m for g, m in zip(FL.Group, FL.Mouse_ID)], 'provider', 'other')
ever = FL.groupby(['Group', 'Mouse_ID']).Entered.sum()
early = FL[(FL.Trial <= 10) & FL.EntryLatency.notna()].copy()
per = (early.groupby(['Group', 'Mouse_ID', 'role']).EntryLatency
       .agg(['median', 'count']).reset_index()
       .rename(columns={'median': 'entry_latency_median_s', 'count': 'n_trials'}))
out('fig1b_early_entry_per_mouse.csv', per,
    f'8 cohort {len(per)}마리 첫 10시행 진입 잠복기 중앙값; 한 번이라도 진입 {int((ever>0).sum())}/{len(ever)}')
out('fig1b_early_entry_trials.csv',
    early[['Group', 'Mouse_ID', 'role', 'Trial', 'EntryLatency']]
    .rename(columns={'Group': 'cohort', 'Mouse_ID': 'mouse', 'Trial': 'trial',
                     'EntryLatency': 'entry_latency_s'}),
    '첫 10시행 진입 시행 단위 자료')

prov = pd.DataFrame([dict(file=k, rows=v['rows'], note=v['note']) for k, v in prov_log.items()])
prov.to_csv(os.path.join(DATA, '_sources.csv'), index=False)
