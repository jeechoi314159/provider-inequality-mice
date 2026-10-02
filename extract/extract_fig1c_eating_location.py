# -*- coding: utf-8 -*-
"""Fig. 1C 먹는 위치 (cohort A): 기존 DBSCAN 섭식 군집 파이프라인을 파이썬으로 재현.
원본: [S] 2024 Group foraging/Data/eating data (DBScan)/main_4_detect_eating_moment_all.m
 - 입력: position_eating_day_{1..15}.mat (EthoVision 좌표, cm, 경기장 중심 원점; 섭식 시작부터 시행 끝까지)
 - 매 프레임 6마리 위치에 DBSCAN(eps 6 cm, minPts 4, 자기 자신 포함), 핵심점 ≥ 2 이면 섭식 군집 프레임,
   군집 중심 = 핵심점 평균. 연속 3 s(99 프레임, fs 33) 이상인 구간만 유지 (원 코드와 같음).
 - 각 군집 프레임의 중심 반지름 < 17 cm(안쪽 실린더 34 cm) 이면 inner, 아니면 outer.
 - 원 파이프라인의 '영상 대조 수동 확인' 단계는 재현하지 않음.
출력: data/fig1c_eating_location_per_trial.csv (day, trial_in_day, eat_frames, inner_frames, inner_pct, outer_pct)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, numpy as np, pandas as pd, scipy.io as sio
SRC = BEHAV + 'eating data (DBScan)'      # position_eating_day_*.mat (Dryad 03_tracking)
OUT = _os.path.join(FIGDATA,
                   'fig1c_eating_location_per_trial.csv')
EPS, MINPTS, MINCORE, MINLEN, R_IN = 6.0, 4, 2, 3 * 33, 34.0 / 2

def frame_cluster(xy):
    ok = np.isfinite(xy).all(1)
    p = xy[ok]
    if len(p) < MINPTS: return None
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    core = (d <= EPS).sum(1) >= MINPTS
    if core.sum() < MINCORE: return None
    return p[core].mean(0)

rows = []
for day in range(1, 16):
    m = sio.loadmat(os.path.join(SRC, f'position_eating_day_{day}.mat'), squeeze_me=True, struct_as_record=False)
    pos = m['pos']; nt = len(np.atleast_1d(pos[0]))
    for tr in range(nt):
        X = np.column_stack([np.asarray(np.atleast_1d(pos[k])[tr].x, float) for k in range(6)])
        Y = np.column_stack([np.asarray(np.atleast_1d(pos[k])[tr].y, float) for k in range(6)])
        T = X.shape[0]
        mark = np.zeros(T, bool); cen = np.full((T, 2), np.nan)
        for t in range(T):
            c = frame_cluster(np.column_stack([X[t], Y[t]]))
            if c is not None: mark[t] = True; cen[t] = c
        keep = np.zeros(T, bool); t = 0
        while t < T:
            if mark[t]:
                s = t
                while t < T and mark[t]: t += 1
                if t - s >= MINLEN: keep[s:t] = True
            else: t += 1
        r = np.hypot(cen[keep, 0], cen[keep, 1])
        n_e, n_in = int(keep.sum()), int((r < R_IN).sum())
        rows.append(dict(day=day, trial_in_day=tr + 1, eat_frames=n_e, inner_frames=n_in,
                         inner_pct=100 * n_in / n_e if n_e else np.nan,
                         outer_pct=100 * (n_e - n_in) / n_e if n_e else np.nan))
    print('day', day, 'trials', nt)
D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
v = D.dropna()
print('trials', len(D), 'with eating', len(v), '| pooled inner % of eating frames',
      round(100 * v.inner_frames.sum() / v.eat_frames.sum(), 2), '| mean per-trial inner %', round(v.inner_pct.mean(), 2),
      '| trials with any inner', int((v.inner_frames > 0).sum()))
