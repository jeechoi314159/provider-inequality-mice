# -*- coding: utf-8 -*-
"""DREADD 행동 — 시행을 모아 표본을 늘린 분석 (numpy 벡터화).
층화(개체/코호트 안에서 조건 라벨 순열)로 모으므로 유사반복이 아님."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, pandas as pd, glob
from scipy import stats
rng = np.random.default_rng(20260930); NPERM = 20000

rows = []
for f in sorted(glob.glob((CHEMO + 'raw/' + 'PFC Inhibition/Group*/Behavior/*.xlsx'))):
    d = pd.read_excel(f); d['cohort'] = f.split('/')[-3].replace('Group', ''); rows.append(d)
P = pd.concat(rows, ignore_index=True)
P['retrieved'] = P['get'].notna().astype(int)
P['t_get'] = P['get'] - P.start; P['t_end'] = P.end - P.start
P['time'] = np.where(P.retrieved == 1, P.t_get, P.t_end.fillna(P.t_end.max()))
P['event'] = P.retrieved
P['rolescore'] = P.role.map({'w': 2, 'p': 1, 'f': 0})
P['cno'] = (P.Condition == 'CNO').astype(int)
N = pd.read_csv('nac_dreadd_trials.csv')
N = N[N.condition.isin(['Saline', 'CNO']) & (N.worker != '-')].copy()
N['cno'] = (N.condition == 'CNO').astype(int); N['retrieved'] = N.worker_retrieved
print('PFC 억제: 시행 %d (Saline %d, CNO %d), 제공자 %d' % (len(P), (P.cno == 0).sum(), (P.cno == 1).sum(), P.cohort.nunique()))
print('NAc 억제: 시행 %d (Saline %d, CNO %d), worker %d' % (len(N), (N.cno == 0).sum(), (N.cno == 1).sum(), N.group.nunique()))


def strat_perm(y, c, s, nperm=NPERM):
    """층별 (CNO 평균 − Saline 평균) 의 표본가중 평균; 층 안에서 라벨 순열."""
    y = np.asarray(y, float); c = np.asarray(c).astype(int); s = np.asarray(s)
    groups = [np.where(s == u)[0] for u in np.unique(s)]
    groups = [g for g in groups if 0 < c[g].sum() < len(g)]
    W = np.array([len(g) for g in groups], float)

    def T(cc):
        return sum(w * (y[g][cc[g] == 1].mean() - y[g][cc[g] == 0].mean()) for g, w in zip(groups, W)) / W.sum()
    t0 = T(c); null = np.empty(nperm)
    for k in range(nperm):
        cp = c.copy()
        for g in groups:
            cp[g] = c[g][rng.permutation(len(g))]
        null[k] = T(cp)
    return t0, float((np.abs(null) >= abs(t0) - 1e-12).mean())


def table(df, strat, lab):
    a = df.groupby('cno').retrieved.agg(['sum', 'size'])
    eff, p = strat_perm(df.retrieved, df.cno, df[strat])
    per = df.groupby([strat, 'cno']).retrieved.mean().unstack('cno')
    dec = int((per[1] < per[0] - 1e-9).sum())
    fp = stats.fisher_exact([[a.loc[0, 'sum'], a.loc[0, 'size'] - a.loc[0, 'sum']],
                             [a.loc[1, 'sum'], a.loc[1, 'size'] - a.loc[1, 'sum']]])[1]
    print('\n== %s' % lab)
    print('   회수  Saline %d/%d (%.2f) → CNO %d/%d (%.2f)   Fisher P = %.3f' % (
        a.loc[0, 'sum'], a.loc[0, 'size'], a.loc[0, 'sum'] / a.loc[0, 'size'],
        a.loc[1, 'sum'], a.loc[1, 'size'], a.loc[1, 'sum'] / a.loc[1, 'size'], fp))
    print('   층화 순열: 회수율 차 %+.3f, P = %.4f ; 회수율이 떨어진 개체 %d/%d' % (eff, p, dec, len(per)))
    return dict(label=lab, s_ret=int(a.loc[0, 'sum']), s_n=int(a.loc[0, 'size']), c_ret=int(a.loc[1, 'sum']),
                c_n=int(a.loc[1, 'size']), fisher_P=fp, strat_diff=eff, strat_P=p, animals_down=dec, animals=len(per))


out = [table(P, 'cohort', 'PFC 억제 — 제공자 회수 (본문 실험)'),
       table(N, 'group', 'NAc 억제 — worker 회수 (set 6)')]
C = pd.concat([pd.DataFrame(dict(animal='PFC_' + P.cohort, cno=P.cno, retrieved=P.retrieved)),
               pd.DataFrame(dict(animal='NAc_' + N.group, cno=N.cno, retrieved=N.retrieved))], ignore_index=True)
out.append(table(C, 'animal', '두 실험 합산 — 12 개체'))

print('\n== PFC 억제 — 시행 수준 지표 (코호트 안에서 조건 순열)')
res2 = []
for col, lab in (('Tau_get', 'log10 회수 소요 (진입→회수), 회수 시행'), ('Tau_entry', 'log10 진입 잠복기, 진입 시행')):
    d = P.dropna(subset=[col]); d = d[d[col] > 0]
    eff, p = strat_perm(np.log10(d[col]), d.cno, d.cohort)
    print('   %-40s n=%d  Δlog10 = %+.3f (×%.2f)  P = %.4f' % (lab, len(d), eff, 10 ** eff, p))
    res2.append(dict(measure=lab, n=len(d), effect=eff, fold=10 ** eff, P=p))
eff, p = strat_perm(P.rolescore, P.cno, P.cohort)
print('   %-40s n=%d  Δ = %+.3f  P = %.4f   Saline %s / CNO %s' % ('역할 점수 (w=2, p=1, f=0)', len(P), eff, p,
      P[P.cno == 0].role.value_counts().to_dict(), P[P.cno == 1].role.value_counts().to_dict()))
res2.append(dict(measure='role score', n=len(P), effect=eff, fold=np.nan, P=p))


def logrank_strat(df, nperm=10000):
    strata = []
    for s, g in df.groupby('cohort'):
        strata.append((g.time.values.astype(float), g.event.values.astype(int), g.cno.values.astype(int)))

    def OE(cs):
        tot = 0.0
        for (t, e, _), cc in zip(strata, cs):
            for tt in np.unique(t[e == 1]):
                atrisk = t >= tt; d = (t == tt) & (e == 1)
                n = atrisk.sum()
                if n:
                    tot += (d & (cc == 1)).sum() - d.sum() * (atrisk & (cc == 1)).sum() / n
        return tot
    o = OE([c for _, _, c in strata]); null = np.empty(nperm)
    for k in range(nperm):
        null[k] = OE([c[rng.permutation(len(c))] for _, _, c in strata])
    return o, float((np.abs(null) >= abs(o) - 1e-12).mean())


print('\n== PFC 억제 — 시행 시작→회수 생존분석 (미회수 = 중도절단, 코호트 층화 log-rank, 순열 10,000)')
o, p = logrank_strat(P)
med = {k: float(np.median(g.time[g.event == 1])) for k, g in P.groupby('cno')}
print('   전체 %d 시행: O−E(CNO) = %+.2f, P = %.4f ; 회수 시행의 중앙 시각 Saline %.1f s → CNO %.1f s' % (len(P), o, p, med[0], med[1]))
P2 = P.copy(); m = (P2.cohort == 'E') & (P2.day_trial == 'd4t10'); P2.loc[m, ['event', 'time']] = [0, 180.0]
o2, p2 = logrank_strat(P2)
print('   민감도 (E d4t10, get = 490 s 를 180 s 중도절단으로): O−E = %+.2f, P = %.4f' % (o2, p2))
res2.append(dict(measure='survival log-rank (stratified)', n=len(P), effect=o, fold=np.nan, P=p))
res2.append(dict(measure='survival log-rank, E d4t10 censored', n=len(P), effect=o2, fold=np.nan, P=p2))
pd.DataFrame(out).to_csv('dreadd_pooled_retrieval.csv', index=False)
pd.DataFrame(res2).to_csv('dreadd_pooled_pfc_measures.csv', index=False)
P.to_csv('dreadd_pfc_trials.csv', index=False)
print('\n개체별 회수율 (Saline → CNO)')
print(pd.concat([P.groupby(['cohort', 'cno']).retrieved.agg(['sum', 'size']).unstack('cno'),
                 N.groupby(['group', 'cno']).retrieved.agg(['sum', 'size']).unstack('cno')]).to_string())
