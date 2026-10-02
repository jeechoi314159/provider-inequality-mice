# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 1))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os
"""R5 (조작 실험) 데이터 추출 — 일별 격자, 단계 요약, 해제 속도, 효과 크기, starter 분석."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats

OUT = (FIGDATA)
RAW = (BEHAV)
ANA = (INTER)

# ----------------------------------------------------------------- 원자료 적재
# 시행표: 한 시행에 두 마리가 함께 회수한 기록은 두 행으로 펼침. trial 열로 시행 수를 셈.
def _wrec(fn, mapping, drop_types=()):
    d = pd.read_excel(RAW + 'Raw_data/' + fn, header=1)
    d.columns = [str(c).strip() for c in d.columns]
    d = d.dropna(subset=['ExpType'])
    d['ExpType'] = d.ExpType.astype(str).str.strip()
    d = d[~d.ExpType.isin(drop_types)].reset_index(drop=True)
    d['ExpDay'] = d.ExpDay.astype(int)
    d['trial'] = np.arange(len(d))
    out = []
    for r in d.itertuples():
        w = str(r.Worker).replace('#', '').strip()
        parts = [q.strip() for q in w.split(',')] if w not in ('nan', '') else []
        who = [mapping.get(q, 'none' if q in ('Give_up', 'Give up', '-', 'X', 'nan', '') else q)
               for q in parts] or ['none']
        for m in dict.fromkeys(who):
            out.append(dict(ExpDay=r.ExpDay, ExpType=r.ExpType, trial=r.trial, who=m))
    return pd.DataFrame(out)


SET4 = 'Set 4 (with CBRAIN. 2022.7 ~ 2022. 10)/Behavior/Colosseum_Set4_Worker_Recording.xlsx'
SET1 = 'Set 1 (with CBRAIN. 2021.11 ~ 2021. 12) /Behavior /Colosseum_Set1_Worker_Recording.xlsx'
A = _wrec(SET4, {'1': 'A1', '2': 'A2', '3': 'A3', '4': 'A4', '5': 'A5', '0': 'A6'})
B = _wrec(SET1, {'2': 'B2', '3': 'B3', '5': 'B5', '0': 'B6'}, drop_types=('Hard_Foraging',))

# --- 기준기는 분석에 쓰인 판정표로 교체 (Fig. 2 와 같은 시행 집합) --------------
# cohort A: 개체별 시행 역할표 (14일 162시행, fig2a_work_rate_data 와 개체별 회수 수 일치)
FA = pd.read_excel(RAW + '[F] Set4_Role_20231008.xlsx', 'Sheet1')
FA.columns = [str(c).strip() for c in FA.columns]
mcols = [c for c in FA.columns if c.lower().startswith('mouse')]
FA['d'] = FA['Day'].ffill()
FA['trial'] = -1 - np.arange(len(FA))
days_a = {d: i + 1 for i, d in enumerate(sorted(FA.d.unique()))}
rows = []
for r in FA.itertuples(index=False):
    rec = dict(zip(FA.columns, r))
    got = [('A' + c.split()[-1]) for c in mcols if str(rec[c]).strip() == 'w']
    for m in (got or ['none']):
        rows.append(dict(ExpDay=days_a[rec['d']], ExpType='Group_Foraging',
                         trial=rec['trial'], who=m))
A = pd.concat([pd.DataFrame(rows), A[A.ExpType != 'Group_Foraging']], ignore_index=True)

# cohort B: 일별 개체별 회수 수 표
EB = pd.read_excel(RAW + 'Raw_data/Set 1 (with CBRAIN. 2021.11 ~ 2021. 12) /Behavior /'
                         'Set1_Group_Foraging_Results_Eachday.xlsx', header=0)
EB.columns = ['day'] + [str(c).strip() for c in EB.columns[1:]]
EB = EB.dropna(subset=['day'])
bmap = {'#0': 'B6', '#1': 'B1', '#2': 'B2', '#3': 'B3', '#4': 'B4', '#5': 'B5'}
rows, t = [], -1
for r in EB.itertuples(index=False):
    rec = dict(zip(EB.columns, r))
    for col, m in bmap.items():
        for _ in range(int(rec[col])):
            t -= 1
            rows.append(dict(ExpDay=int(rec['day']), ExpType='Group_Foraging', trial=t, who=m))
B = pd.concat([pd.DataFrame(rows), B[B.ExpType != 'Group_Foraging']], ignore_index=True)

# cohort C: 기준기는 제공자 C1 이 90시행 전부를 회수 (fig2a_work_rate_data), 하루 10시행
WC = pd.read_csv(ANA + 'fig2a_work_rate_data.csv')
WC = WC[(WC.Group == 'C') & (WC.Worked == 1)]
Cb = pd.DataFrame(dict(ExpDay=((WC.Trial.values - 1) // 10 + 1).astype(int),
                       ExpType='Group_Foraging', trial=-len(WC) + np.arange(len(WC)),
                       who=WC.Mouse_ID.values))
MT = pd.read_csv((INTER + 'membership_trials.csv'))
off, blocks, t = int(Cb.ExpDay.max()), [], 0
for ph in ['Without_1st_Worker', 'Separated_Group (both subgroups)',
           'Reunion_2nd_Worker', 'All_Reunion']:
    g = MT[(MT.cohort == 'C') & (MT.ExpType == ph)].copy()
    g['who'] = g.who.astype(str).str.replace('#', '', regex=False)
    g['ExpDay'] = g.ExpDay.astype(int) + off
    g['trial'] = np.arange(t, t + len(g)); t += len(g)
    off = int(g.ExpDay.max())
    blocks.append(g[['ExpDay', 'ExpType', 'trial', 'who']])
C = pd.concat([Cb] + blocks, ignore_index=True)

RAWD = {'A': A, 'B': B, 'C': C}


# ------------------------------- cohort별 시행 순서 테이블 (단계 순 → 날짜 순 → 시행 순)
PORDER_RAW = ['Group_Foraging', 'Without_1st_Worker', 'Without_1st_2nd_Worker',
              'Separated_Group_Active', 'Separated_Group_Passive',
              'Separated_Group (both subgroups)', 'Reunion_3rd_Worker',
              'Reunion_2nd_Worker', 'Reunion_2nd,3rd_Worker', 'All_Reunion']


def trial_table(coh):
    d = RAWD[coh].copy()
    d['who'] = d.who.astype(str)
    d = d[d.who.str.match(r'^[A-C]\d+$') | (d.who == 'none')]
    d['po'] = d.ExpType.map({p: i for i, p in enumerate(PORDER_RAW)})
    key = d[['po', 'ExpDay', 'trial']].drop_duplicates().sort_values(['po', 'ExpDay', 'trial'])
    key['seq'] = np.arange(len(key))
    d = d.merge(key, on=['po', 'ExpDay', 'trial'])
    return d.sort_values('seq')


TT = {c: trial_table(c) for c in 'ABC'}

# --------------------------------------------------- 단계 정의(라벨·부재·하위집단)
# 부재·하위집단은 ExpType 라벨과 기준기 기여 순위로 정함. 분리 단계는 순위 전반/후반.
PHASES = {
    'Group_Foraging':                   ('Group',                    'none'),
    'Without_1st_Worker':               ('Provider out',             'out1'),
    'Without_1st_2nd_Worker':           ('+ successor out',          'out2'),
    'Separated_Group_Active':           ('Split',                    'splitA'),
    'Separated_Group_Passive':          ('Split',                    'splitB'),
    'Separated_Group (both subgroups)': ('Split',                    'splitC'),
    'Reunion_3rd_Worker':               ('Rejoined, provider out',   'out2'),
    'Reunion_2nd,3rd_Worker':           ('Successor back',           'out1'),
    'Reunion_2nd_Worker':               ('Rejoined, provider out',   'out1'),
    'All_Reunion':                      ('All back',                 'none'),
}

ROSTER = {'A': ['A%d' % k for k in range(1, 7)],
          'B': ['B%d' % k for k in range(1, 7)],
          'C': ['C1', 'C2', 'C4', 'C5', 'C6']}
SPLIT = {'A': 3, 'B': 3, 'C': 2}          # 제공자가 속한 하위집단의 크기 (원 설계: 기여 순위 전반)


def cohort_grid(coh):
    d = RAWD[coh].copy()
    d['who'] = d.who.astype(str)
    d = d[d.who.str.match(r'^[A-C]\d+$') | (d.who == 'none')]
    roster = ROSTER[coh]
    bl = d[d.ExpType == 'Group_Foraging']
    cnt = bl[bl.who != 'none'].who.value_counts()
    base_order = [m for m in cnt.index if m in roster] + \
                 [m for m in roster if m not in list(cnt.index)]
    prov = base_order[0]
    # 승계 순서: 제공자 → 제거 단계마다 새로 나타난 최다 회수자 → 나머지는 기준기 기여 순
    chain = [prov]
    for ph in ('Without_1st_Worker', 'Without_1st_2nd_Worker'):
        g = d[(d.ExpType == ph) & (d.who != 'none')]
        if len(g):
            m = g.who.value_counts().index[0]
            if m not in chain:
                chain.append(m)
    succ = chain[1] if len(chain) > 1 else None
    tot_all = d[d.who != 'none'].who.value_counts()
    rest = sorted([m for m in base_order if m not in chain],
                  key=lambda m: (-int(tot_all.get(m, 0)), base_order.index(m)))
    rank = chain + rest
    k = SPLIT[coh]
    gA, gB = set(rank[:k]), set(rank[k:])

    def members(rule):
        if rule == 'none':
            return set(rank), ''
        if rule == 'out1':
            return set(rank) - {prov}, ''
        if rule == 'out2':
            return set(rank) - {prov, succ}, ''
        if rule == 'splitA':
            return gA, 'with provider'
        if rule == 'splitB':
            return gB, 'without provider'
        return set(rank), 'both'

    rows, strip = [], []
    for day, g in d.groupby('ExpDay'):
        ntr_day = g.trial.nunique()
        lab0 = PHASES[g.ExpType.iloc[0]][0]
        strip.append(dict(cohort=coh, day=int(day), n_trials=ntr_day,
                          frac_retrieved=float(g[g.who != 'none'].trial.nunique() / ntr_day),
                          phase=lab0))
        seen = set()
        for ph in g.ExpType.unique():
            gg = g[g.ExpType == ph]
            lab, rule = PHASES[ph]
            present, sub = members(rule)
            n = gg.trial.nunique()
            for m in present:
                if m in seen:
                    continue
                seen.add(m)
                kk = int((gg.who == m).sum())
                rows.append(dict(cohort=coh, day=int(day), phase=lab, subgroup=sub, mouse=m,
                                 rank=rank.index(m) + 1, is_provider=int(m == prov), present=1,
                                 retrievals=kk, n_trials=n, share=float(kk / n) if n else np.nan))
        for m in set(rank) - seen:
            rows.append(dict(cohort=coh, day=int(day), phase=lab0, subgroup='', mouse=m,
                             rank=rank.index(m) + 1, is_provider=int(m == prov), present=0,
                             retrievals=0, n_trials=0, share=np.nan))
    G = pd.DataFrame(rows)
    return G.sort_values(['day', 'rank']), pd.DataFrame(strip), rank, prov, succ


GRID, STRIP, RANKS = {}, {}, {}
for coh in 'ABC':
    g, s, rk, prov, succ = cohort_grid(coh)
    GRID[coh], STRIP[coh] = g, s
    RANKS[coh] = dict(order=rk, provider=prov, successor=succ)
GG = pd.concat(GRID.values(), ignore_index=True)
SS = pd.concat(STRIP.values(), ignore_index=True)
GG.to_csv(OUT + 'fig3a_daily_grid.csv', index=False)
SS.to_csv(OUT + 'fig3a_daily_strip.csv', index=False)

PORDER = ['Group', 'Provider out', '+ successor out', 'Split',
          'Rejoined, provider out', 'Successor back', 'All back']
RNG = np.random.default_rng(20260928)


def episodes(coh):
    """구성 episode = (단계, 하위집단). 각 episode 의 참여 개체·시행 수·회수 수."""
    g = GRID[coh].assign(subgroup=GRID[coh].subgroup.fillna(''))
    base = g[(g.phase == 'Group') & (g.present == 1)]
    nb = base.groupby('day').n_trials.first().sum()
    base_rate = base.groupby('mouse').retrievals.sum() / nb
    out = []
    for (ph, sub), gg in g[g.present == 1].groupby(['phase', 'subgroup']):
        n = int(gg.groupby('day').n_trials.first().sum())
        k = gg.groupby('mouse').retrievals.sum()
        if n == 0 or k.sum() == 0:
            continue
        out.append(dict(phase=ph, subgroup=sub, n_trials=n, k=k,
                        base=base_rate.reindex(k.index).fillna(0.0)))
    return base_rate, sorted(out, key=lambda r: PORDER.index(r['phase']))


# ------------------ Fig. 3B 이전 순위와 기여: 남은 개체 중 1위가 그 자리를 맡는가
rows, inset = [], []
for coh in 'ABC':
    base_rate, eps = episodes(coh)
    for e in eps:
        k, n, b = e['k'], e['n_trials'], e['base']
        order = b.sort_values(ascending=False, kind='mergesort')
        rank = {m: i + 1 for i, m in enumerate(order.index)}
        top = k.idxmax()
        for m in k.index:
            rows.append(dict(cohort=coh, phase=e['phase'], subgroup=e['subgroup'], mouse=m,
                             avail_rank=rank[m], base_rate=float(b[m]),
                             delivery_share=float(k[m] / k.sum()),
                             deliveries_per_trial=float(k[m] / n),
                             n_available=len(k), n_trials=n,
                             is_top=int(m == top)))
        # 두 가지 귀무 예측 아래에서 '1위가 최다 회수자'가 될 확률
        p_eq = np.full(len(k), 1.0 / len(k))
        w = b.values + 0.5 / n                      # 0회 개체 보정
        p_bs = w / w.sum()
        r1 = list(order.index).index(order.index[0])
        r1i = list(k.index).index(order.index[0])
        sim = {}
        for name, pv in (('Equal among available', p_eq), ('Baseline rates, renormalised', p_bs)):
            draws = RNG.multinomial(int(k.sum()), pv, size=4000)
            win = (draws[:, r1i] == draws.max(axis=1)) & (draws[:, r1i] > 0)
            tie = (draws[:, r1i] == draws.max(axis=1)).sum() - win.sum()
            sim[name] = float(win.mean())
        inset.append(dict(cohort=coh, phase=e['phase'], subgroup=e['subgroup'],
                          observed=int(top == order.index[0]), n_available=len(k),
                          **sim))
RK = pd.DataFrame(rows)
RK['phase_order'] = RK.phase.map({p: i for i, p in enumerate(PORDER)})
RK.to_csv(OUT + 'fig3b_rank_vs_contribution.csv', index=False)
INS = pd.DataFrame(inset)
INS.to_csv(OUT + 'fig3b_top_rank_prediction.csv', index=False)

# ------------- Fig. 3C 가역적 역할 교대: 상위 개체 있음 → 없음 → 복귀, 개체별 시행당 회수
rows = []
for coh in 'ABC':
    base_rate, eps = episodes(coh)
    g = GRID[coh].assign(subgroup=GRID[coh].subgroup.fillna(''))

    def per_trial(ph, mouse, sub=''):
        gg = g[(g.phase == ph) & (g.present == 1) & (g.subgroup == sub)]
        n = gg.groupby('day').n_trials.first().sum()
        return float(gg[gg.mouse == mouse].retrievals.sum() / n) if n else np.nan

    prov = RANKS[coh]['provider']
    seen = set()
    for e in eps:
        if e['phase'] in ('Group', 'All back'):
            continue
        k, b = e['k'], e['base']
        # 후보: 그 episode 의 최다 회수자. 두 하위집단이 한 블록에 섞인 경우
        # (cohort C 의 분리 단계) 원 제공자를 뺀 최다 회수자도 대체 개체로 본다.
        cands = [k.idxmax()]
        if e['subgroup'] == 'both':
            k2 = k.drop(index=[m for m in (prov,) if m in k.index])
            if len(k2) and k2.max() > 0:
                cands.append(k2.idxmax())
        for focal in cands:
            if focal in seen or focal == prov:
                continue
            higher = base_rate[base_rate > b[focal] + 1e-12]
            if len(higher) == 0 and e['subgroup'] != 'both':
                continue                            # 자기보다 위가 없으면 교대가 아님
            seen.add(focal)
            rows.append(dict(cohort=coh, mouse=focal, episode=e['phase'], subgroup=e['subgroup'],
                             n_higher_removed=int(len(higher)),
                             highest_available=int(b[focal] == b.max()),
                             overall_fraction=float(
                                 GRID[coh][GRID[coh].mouse == focal].retrievals.sum() /
                                 GRID[coh].groupby(['day', 'subgroup'], dropna=False)
                                 .n_trials.first().sum()),
                             present=per_trial('Group', focal),
                             absent=float(k[focal] / e['n_trials']),
                             returned=per_trial('All back', focal)))
REV = pd.DataFrame(rows)
REV.to_csv(OUT + 'fig3c_three_conditions.csv', index=False)

# ------------------------- Fig. 3D 구성 변화 전후 시행 정렬 (제거 사건 · 재결합 사건)
EV = {'removal': 'Without_1st_Worker', 'reunion': 'All_Reunion'}
rows = []
for coh in 'ABC':
    t = TT[coh]
    prov = RANKS[coh]['provider']
    out = GRID[coh][(GRID[coh].phase == 'Provider out') & (GRID[coh].present == 1)]
    repl = out.groupby('mouse').retrievals.sum().idxmax()
    seqs = t[['seq', 'ExpType']].drop_duplicates('seq')
    for ev, ph in EV.items():
        s0 = seqs[seqs.ExpType == ph].seq.min()
        if not np.isfinite(s0):
            continue
        for mouse, role in ((repl, 'Replacement'), (prov, 'Returning provider')):
            sub = t[t.who == mouse]
            got = set(sub.seq)
            for s in seqs.seq:
                r = int(s - s0)
                if abs(r) > 60:
                    continue
                rows.append(dict(cohort=coh, event=ev, mouse=mouse, role=role,
                                 rel_trial=r, retrieved=int(s in got)))
EA = pd.DataFrame(rows)
EA.to_csv(OUT + 'fig3d_event_aligned.csv', index=False)

# --------------------------------------------- 해제 속도 (fig. S6): 제거 후 날짜별 회수율
rows = []
for coh in 'ABC':
    g = GRID[coh]
    base = g[(g.phase == 'Group') & (g.present == 1)]
    pv = base[base.is_provider == 1].retrievals.sum() / base.groupby('day').n_trials.first().sum()
    sub = STRIP[coh][STRIP[coh].phase == 'Provider out'].sort_values('day').reset_index(drop=True)
    for i, r in sub.iterrows():
        rows.append(dict(cohort=coh, day=i + 1, n_trials=int(r.n_trials),
                         frac_retrieved=float(r.frac_retrieved), provider_fraction_before=float(pv)))
pd.DataFrame(rows).to_csv(OUT + 'figS06c_release_by_day.csv', index=False)

# ------------------------------- 기준기 기여율과 불확실성 (fig. S6: 순위 신뢰도)
rows = []
for coh in 'ABC':
    g = GRID[coh]
    base = g[(g.phase == 'Group') & (g.present == 1)]
    n = int(base.groupby('day').n_trials.first().sum())
    k = base.groupby('mouse').retrievals.sum()
    for m in k.index:
        lo, hi = stats.beta.ppf([0.025, 0.975], k[m] + 0.5, n - k[m] + 0.5)
        rows.append(dict(cohort=coh, mouse=m, n_trials=n, retrievals=int(k[m]),
                         rate=float(k[m] / n), lo=float(lo), hi=float(hi)))
pd.DataFrame(rows).to_csv(OUT + 'figS06d_baseline_rank.csv', index=False)

# ------------------------------------------------------------------ 효과 크기 (fig. S6)
def cohen_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


PREV = pd.read_csv(OUT + 'fig3d_effect_sizes.csv') if os.path.exists(OUT + 'fig3d_effect_sizes.csv') \
    else pd.read_csv(OUT + 'figS06h_effect_sizes.csv')
keep = PREV[PREV.effect.str.contains('started')].copy()
rows = []
for coh in 'ABC':
    g = GRID[coh]
    base = g[(g.phase == 'Group') & (g.present == 1)]
    nb = base.groupby('day').n_trials.first().sum()
    oth = base[base.is_provider == 0].retrievals.sum() / nb
    out = g[(g.phase == 'Provider out') & (g.present == 1)]
    no = out.groupby('day').n_trials.first().sum()
    oth2 = out.retrievals.sum() / no
    rows.append(dict(effect='Provider in group\nvs removed', cohort=coh, p_ref=float(oth),
                     p_alt=float(oth2), h=float(abs(cohen_h(oth2, oth))), n_ref=int(nb), n_alt=int(no)))
pd.concat([pd.DataFrame(rows), keep], ignore_index=True).to_csv(
    OUT + 'figS06h_effect_sizes.csv', index=False)

# --------------------------------------------- starter 표현형·과다·회수율 (fig. S6)
SL = pd.read_csv(ANA + 'starter_labels.csv')
SH = pd.read_csv(ANA + 'fig4c_starter_halves.csv')
SH.to_csv(OUT + 'figS06e_starter_halves.csv', index=False)

FILES = ['fig3a_daily_grid.csv', 'fig3a_daily_strip.csv', 'fig3b_rank_vs_contribution.csv',
         'fig3b_top_rank_prediction.csv', 'fig3c_three_conditions.csv', 'fig3d_event_aligned.csv',
         'figS06c_release_by_day.csv', 'figS06d_baseline_rank.csv', 'figS06e_starter_halves.csv',
         'figS06f_starter_excess.csv', 'figS06g_rate_by_starter.csv', 'figS06h_effect_sizes.csv']
src = pd.DataFrame([dict(file=f, rows=len(pd.read_csv(OUT + f))) for f in FILES
                    if os.path.exists(OUT + f)])
src.to_csv(OUT + '_sources_r5.csv', index=False)
print(src.to_string(index=False))
print(INS.groupby('cohort')[['observed', 'Equal among available', 'Baseline rates, renormalised']].mean().to_string())
print('overall observed %.2f  equal %.2f  baseline %.2f  (n=%d episodes)' % (
    INS.observed.mean(), INS['Equal among available'].mean(),
    INS['Baseline rates, renormalised'].mean(), len(INS)))
print(REV.to_string(index=False))
