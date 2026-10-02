# -*- coding: utf-8 -*-
"""DREADD 시행별 신경 + 행동 결합 표. 채널 대응: 파일 열 BLA_=ch1=실제 PFC, NAc_=ch2, PFC_=ch3=실제 BLA.
소자 선별은 analyze_dreadd.py 규칙과 동일."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import pandas as pd, numpy as np
B = pd.read_csv((CHEMO + 'bandpower_whole.csv'))
BEH = pd.read_csv('dreadd_pfc_trials.csv')
COL = ['BLA', 'NAc', 'PFC']; REAL = {'BLA': 'PFC', 'NAc': 'NAc', 'PFC': 'BLA'}
BANDS = ['theta1', 'theta2', 'beta1', 'beta2', 'gamma1', 'gamma2']
def onef(g, rg): return np.median(np.log10(g['%s_theta1' % rg]) - np.log10(g['%s_gamma2' % rg]))
ok = {}
for (coh, cond, dev), g in B.groupby(['cohort', 'condition', 'device']):
    o = [onef(g, rg) for rg in COL]; sd = [g['sd_ch%d' % i].median() for i in (1, 2, 3)]
    synth = (max(o) - min(o) < 0.03) and (max(sd) - min(sd)) / max(sd) < 0.02
    for i, rg in enumerate(COL): ok[(coh, cond, dev, rg)] = (not synth) and (0.01 < sd[i] < 0.5)
use = {}
for coh in sorted(B.cohort.unique()):
    for dev in sorted(B[B.cohort == coh].device.unique()):
        if not all(((B.cohort == coh) & (B.condition == c) & (B.device == dev)).any() for c in ('CNO', 'Saline')): continue
        keep = [rg for rg in COL if ok.get((coh, 'CNO', dev, rg)) and ok.get((coh, 'Saline', dev, rg))]
        if keep: use[(coh, dev)] = keep
rows = []
for (coh, dev), keep in use.items():
    s = B[(B.cohort == coh) & (B.device == dev) & (B.artifact_frac <= 0.01)]
    for _, r in s.iterrows():
        d = dict(cohort=coh, condition=r.condition, segment=r.segment, trial=int(r.trial), seconds=r.seconds)
        for rg in keep:
            for bn in BANDS:
                d['%s_%s' % (REAL[rg], bn)] = np.log10(max(r['%s_%s' % (rg, bn)], 1e-12))
            d['tilt_%s' % REAL[rg]] = d['%s_theta2' % REAL[rg]] - d['%s_theta1' % REAL[rg]]
        rows.append(d)
N = pd.DataFrame(rows)
N['tilt'] = N[['tilt_PFC', 'tilt_NAc']].mean(axis=1) if 'tilt_NAc' in N else N['tilt_PFC']
W = N.pivot_table(index=['cohort', 'condition', 'trial'], columns='segment',
                  values=[c for c in N.columns if c.startswith(('PFC_', 'NAc_', 'BLA_', 'tilt'))])
W.columns = ['%s__%s' % (a, b) for a, b in W.columns]; W = W.reset_index()
BEH['condition'] = BEH.Condition.map(lambda x: 'CNO' if str(x).upper() == 'CNO' else 'Saline')
M = BEH.merge(W, on=['cohort', 'condition', 'trial'], how='left')
M['cno'] = (M.condition == 'CNO').astype(int)
M.to_csv('dreadd_trial_neural.csv', index=False)
print('시행 %d, PFC 신경값 있는 시행 %d, 코호트별 %s' % (len(M), M['tilt_PFC__Baseline'].notna().sum(),
      M.groupby('cohort')['tilt_PFC__Baseline'].apply(lambda v: int(v.notna().sum())).to_dict()))
print('사용 소자:', {k[0]: (int(k[1]), [REAL[x] for x in v]) for k, v in use.items()})
# 개체별 요약: 신경 Δ (CNO−Saline) 와 행동 Δ
rows = []
for coh, g in M.groupby('cohort'):
    d = dict(cohort=coh, provider={'E': 'E1', 'F': 'F4', 'G': 'G2', 'H': 'H4'}[coh])
    for seg in ('Baseline', 'Foraging'):
        for feat in ('tilt_PFC', 'tilt_NAc', 'tilt', 'PFC_beta2', 'PFC_gamma2', 'PFC_theta1', 'PFC_theta2'):
            c = '%s__%s' % (feat, seg)
            if c in g and g[c].notna().any():
                d['d_%s_%s' % (feat, seg)] = g[g.cno == 1][c].mean() - g[g.cno == 0][c].mean()
    d['ret_sal'] = g[g.cno == 0].retrieved.mean(); d['ret_cno'] = g[g.cno == 1].retrieved.mean()
    d['d_ret'] = d['ret_cno'] - d['ret_sal']
    tg = g.dropna(subset=['Tau_get']); tg = tg[tg.Tau_get > 0]
    d['d_logTauGet'] = np.log10(tg[tg.cno == 1].Tau_get).mean() - np.log10(tg[tg.cno == 0].Tau_get).mean()
    d['d_logTime'] = np.log10(g[(g.cno == 1) & (g.event == 1)].time).mean() - np.log10(g[(g.cno == 0) & (g.event == 1)].time).mean()
    rows.append(d)
S = pd.DataFrame(rows); S.to_csv('dreadd_animal_summary.csv', index=False)
pd.set_option('display.width', 220)
print(S[['provider', 'd_tilt_PFC_Baseline', 'd_tilt_PFC_Foraging', 'd_tilt_Baseline', 'd_PFC_beta2_Foraging', 'd_PFC_gamma2_Foraging', 'd_ret', 'd_logTauGet']].round(3).to_string(index=False))
# 시행 수준: tilt(기저·채집) 와 회수 시각의 개체 내 상관
from scipy import stats
for c in ('tilt_PFC__Baseline', 'tilt_PFC__Foraging', 'tilt__Baseline', 'PFC_beta2__Foraging'):
    s = M.dropna(subset=[c, 'time']); s = s[s.event == 1]
    s = s.assign(xc=s[c] - s.groupby('cohort')[c].transform('mean'), yc=np.log10(s.time) - s.groupby('cohort')['time'].transform(lambda v: np.log10(v).mean()))
    r, p = stats.spearmanr(s.xc, s.yc)
    print('%-22s vs log 회수시각 (회수 시행, 개체 중심화): ρ = %+.3f, P = %.3f, n = %d' % (c, r, p, len(s)))
