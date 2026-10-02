# -*- coding: utf-8 -*-
"""fig. S2 A·B 자료 생성 — 전 코호트 단독 대 집단.

A: 단독 채집에서는 46마리 모두 회수함(20분 제한시간, 실험자 보고). 따라서 rate_solitary = 1.0.
   집단 회수율은 개체별 실측값(figS02b_retrieval_rate_per_mouse.csv).
B: 회수 소요 시간은 fig. S5B 자료(figS05b_retrieval_solitary_vs_group.csv)에서 개체별 중앙값.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os
import pandas as pd
from scipy import stats

D = FIGDATA.rstrip(_os.sep)              # figure source data
G = pd.read_csv(os.path.join(D, 'figS02b_retrieval_rate_per_mouse.csv'))
n = G.groupby('cohort').mouse.nunique().rename('n_mice')
S = G.merge(n, on='cohort')
S['rate_solitary'] = 1.0
S['equal_share'] = 1.0 / S.n_mice
S = S.rename(columns={'retrieval_rate': 'rate_group'})
S = S[['cohort', 'mouse', 'role', 'n_mice', 'rate_solitary', 'rate_group', 'equal_share']]
S = S.sort_values(['cohort', 'mouse']).reset_index(drop=True)
S.to_csv(os.path.join(D, 'figS02a_solitary_vs_group_per_mouse.csv'), index=False)

w = stats.wilcoxon(S.rate_solitary, S.rate_group)
print('A: %d마리 / %d코호트, 집단에서 낮아진 개체 %d, Wilcoxon P = %.2g'
      % (len(S), S.cohort.nunique(), int((S.rate_group < S.rate_solitary).sum()), w.pvalue))
print('   균등 기여선 1/n = %.2f–%.2f (코호트 크기 %d–%d)'
      % (S.equal_share.min(), S.equal_share.max(), S.n_mice.min(), S.n_mice.max()))

R = pd.read_csv(os.path.join(D, 'figS05b_retrieval_solitary_vs_group.csv'))
per = (R.groupby(['cohort', 'mouse', 'role', 'condition']).retrieval_latency_s.median()
         .unstack('condition').dropna().reset_index())
w2 = stats.wilcoxon(per.Group, per.Solitary)
print('B: %d마리 / %d코호트, 집단에서 느려진 개체 %d, Wilcoxon P = %.3f'
      % (len(per), per.cohort.nunique(), int((per.Group > per.Solitary).sum()), w2.pvalue))
