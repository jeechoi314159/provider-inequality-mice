# -*- coding: utf-8 -*-
"""사전 형질 자료를 제출용 Source Data 워크북(Extended Fig 3b, 3d–3h)에서 다시 추출 (2026-10-01).
이전 fig1e_trait_table.csv 는 data_230918_update.xls(36마리, 형질별 2–5 제공자) 기반으로 표본이 작아 Table S3 과 어긋났음.
출력: data/fig1e_trait_table.csv (개체별 백분위 점수), data/fig1e_effect_sizes.csv (Hedges g, 95% CI, P)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, numpy as np, pandas as pd
from scipy import stats
D = FIGDATA.rstrip(_os.sep)              # figure source data
SRC = _os.environ.get('PVSNP_SOURCE_DATA') or next(iter(__import__('glob').glob(_os.path.join(__import__('glob').escape(RAWROOT), '2025-resubmission', '*', '[[]Submission[]]Source Data_260521.xlsx'))), '')   # 2026 source-data workbook (not deposited; set PVSNP_SOURCE_DATA)
x = pd.ExcelFile(SRC)
def sheet(name, col):
    d = x.parse(name, header=None)
    hdr = d.index[d[0].astype(str).eq('Group')][0]
    d.columns = d.iloc[hdr]; d = d.iloc[hdr + 1:].dropna(subset=['Group'])
    d = d.rename(columns={[c for c in d.columns if c not in ('Group', 'Mouse', 'Role')][0]: col})
    d['mouse'] = d.Group.astype(str) + d.Mouse.astype(int).astype(str)
    d['role'] = np.where(d.Role.astype(str).str.strip() == 'P', 'provider', 'other')
    d[col] = pd.to_numeric(d[col], errors='coerce')
    return d[['mouse', 'role', col]]
T = sheet('Extended Fig 3b', 'tube_rank_pct')
for nm, col in (('Extended Fig 3d', 'oft_center_pct'), ('Extended Fig 3e', 'epm_open_pct'),
                ('Extended Fig 3f', 'ymaze_alternation_pct'), ('Extended Fig 3g', 'rotarod_pct'), ('Extended Fig 3h', 'body_weight_g')):
    T = T.merge(sheet(nm, col), on=['mouse', 'role'], how='outer')
F10 = pd.read_csv(os.path.join(D, 'fig1e_first10_share.csv'))
T.to_csv(os.path.join(D, 'fig1e_trait_table.csv'), index=False)
def hedges(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b); sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp; J = 1 - 3 / (4 * (n1 + n2) - 9); g = J * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2))); p = stats.mannwhitneyu(a, b).pvalue
    return g, g - 1.96 * se, g + 1.96 * se, p, n1, n2
rows = []
for col, lab in (('tube_rank_pct', 'Social dominance (tube rank)'), ('oft_center_pct', 'Open field centre time'),
                 ('epm_open_pct', 'Elevated plus maze open arm'), ('ymaze_alternation_pct', 'Y-maze alternation'),
                 ('rotarod_pct', 'Rotarod latency to fall'), ('body_weight_g', 'Body weight')):
    g, lo, hi, p, n1, n2 = hedges(T.loc[T.role == 'provider', col], T.loc[T.role == 'other', col])
    rows.append(dict(measure=lab, kind='baseline trait', g=g, ci_lo=lo, ci_hi=hi, p=p, n_provider=n1, n_other=n2))
fc = [c for c in F10.columns if c not in ('mouse', 'cohort', 'role')][0]
g, lo, hi, p, n1, n2 = hedges(F10.loc[F10.role == 'provider', fc], F10.loc[F10.role != 'provider', fc])
rows.append(dict(measure='Fraction of first 10 retrievals', kind='early behaviour', g=g, ci_lo=lo, ci_hi=hi, p=p, n_provider=n1, n_other=n2))
E = pd.DataFrame(rows); E.to_csv(os.path.join(D, 'fig1e_effect_sizes.csv'), index=False)
pd.set_option('display.width', 200); print(E.round(3).to_string(index=False))
