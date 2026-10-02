# -*- coding: utf-8 -*-
"""fig. S2G (2026-10-02, red note "write who were the mice for these 3 trials"): 닫힌 칸 시험에서 먹이를 밖으로
가지고 나간 시행의 개체·시행 번호. 원자료: 2026 Source data (J. Lee) 시트 'Extended Fig 1i' (figS02i_chamber_outcome.csv 와 같은 시트).
역할(provider/other)은 figS02b_retrieval_rate_per_mouse.csv 에서 붙임."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, glob
import pandas as pd
D = FIGDATA.rstrip(_os.sep)              # figure source data
SRC = _os.environ.get('PVSNP_SOURCE_DATA') or next(iter(__import__('glob').glob(_os.path.join(__import__('glob').escape(RAWROOT), '2025-resubmission', '*', '[[]Submission[]]Source Data_260521.xlsx'))), '')   # 2026 source-data workbook (not deposited; set PVSNP_SOURCE_DATA)
assert SRC, 'set PVSNP_SOURCE_DATA to the source-data workbook'
raw = pd.read_excel(SRC, sheet_name='Extended Fig 1i', header=None)
hdr = raw.index[raw[0].astype(str).str.strip() == 'Group'][0]
T = raw.iloc[hdr + 1:, :5].copy(); T.columns = ['cohort', 'mouse_n', 'trial', 'consumed_inside', 'behaviour']
T = T.dropna(subset=['cohort'])
T['behaviour'] = T.behaviour.astype(str).str.strip()
assert len(T) == 70 and (T.behaviour == 'Bring out').sum() == 3
O = T[T.behaviour == 'Bring out'].copy()
O['mouse'] = O.cohort.astype(str).str.strip() + O.mouse_n.astype(int).astype(str)
R = pd.read_csv(os.path.join(D, 'figS02b_retrieval_rate_per_mouse.csv'))
O = O.merge(R[['mouse', 'role', 'retrieval_rate']], on='mouse', how='left')
O = O[['cohort', 'mouse', 'trial', 'behaviour', 'role', 'retrieval_rate']].astype({'trial': int})
O.to_csv(os.path.join(D, 'figS02g_carried_out_trials.csv'), index=False)
print('source:', SRC); print(O.to_string(index=False))
