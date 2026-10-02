# -*- coding: utf-8 -*-
"""코호트 A 의 θ tilt 를 구성 변경 자료(Day 21–45)에서 재현 시도.
정의는 돌리기 전에 고정: 동료 중 첫 진입 정렬, 창 0.25–2.25 s,
tilt = (theta2 − theta1) 의 실제 PFC·NAc 평균(log10, 기저 미차감),
자기 진입이 사건 +3.25 s 안이면 제외, 관찰자는 그날 provider 상태가 아닌 개체.
채널: 원시 텍스트 추출은 index0=실제 PFC, 1=NAc, 2=실제 BLA (CHANNEL_ORDER.md)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import numpy as np, csv, os, re, glob, collections

D = (LFP + 'member_change')
OUT = (WORK + 'member_change')
GRID = (FIGDATA[:-1] + '/fig3a_daily_grid.csv')
STEP = 0.125
IDX_PFC, IDX_NAC = 0, 1
TH1, TH2, BE2 = 0, 1, 3
WIN = (0.25, 2.25); GUARD = 3.25
RIG = {'A1': '1_2_3_7', 'A2': '1_2_3_7', 'A3': '1_2_3_7',
       'A4': '4_5_0_8', 'A5': '4_5_0_8', 'A6': '4_5_0_8'}
SUBG = {'active': {'A2', 'A5', 'A6'}, 'passive': {'A1', 'A3', 'A4'}}
EXCL = {(29, 7)}

grid = collections.defaultdict(dict)
for r in csv.DictReader(open(GRID)):
    if r['cohort'] == 'A':
        grid[int(r['day'])][r['mouse']] = (int(r['retrievals']), int(r['n_trials']))
state = {}
for ex, dd in grid.items():
    for mo, (ret, n) in dd.items():
        if n:
            rate = ret / float(n)
            state[(ex + 5, mo)] = ('provider' if rate >= 0.4 else
                                   'withdrawn' if rate <= 0.1 else 'mid', rate)
EV = [r for r in csv.DictReader(open(os.path.join(D, 'timing_record_events.csv')))
      if r['dataset'] == 'member_change']
trials = collections.defaultdict(list)
for r in EV:
    trials[(int(r['day']), int(r['lfp_trial']), r['subgroup'])].append(
        (r['mouse'], r['event'], float(r['time_s'])))
qual = {}
for r in csv.DictReader(open(os.path.join(D, 'bandpower_whole.csv'))):
    qual[(int(r['day']), r['segment'], int(r['trial']), r['mouse'])] = (
        int(float(r['frozen'])), float(r['artifact_frac']), float(r['seconds']))
index = {}
for f in sorted(glob.glob(os.path.join(D, 'spec_Day*_*.npz'))):
    m = re.match(r'spec_Day(\d+)_(.+)_p(\d+)\.npz$', os.path.basename(f))
    with np.load(f, allow_pickle=True) as z:
        for k in z.files:
            if k.endswith('_band'):
                mm = re.match(r'(\w+?)_T(-?\d+)_(A\d)_band$', k)
                if mm:
                    index[(int(m.group(1)), m.group(2), mm.group(1),
                           int(mm.group(2)), mm.group(3))] = (f, k)
print('npz entries', len(index), flush=True)
cache = {}


def series(day, mouse, seg, trial):
    key = (day, RIG[mouse], seg, trial, mouse)
    if key not in index:
        return None
    f, k = index[key]
    if f not in cache:
        cache.clear(); cache[f] = np.load(f, allow_pickle=True)
    return cache[f][k]


def wm(s, t0, t1):
    i0 = max(0, int(round(t0 / STEP))); i1 = min(len(s), int(round(t1 / STEP)))
    if i1 - i0 < 8:
        return np.nan
    v = s[i0:i1]; v = v[np.isfinite(v) & (v > 0)]
    return np.log10(v).mean() if v.size else np.nan


rows = []; seen_days = set()
for (day, lt, sub), evs in sorted(trials.items()):
    if (day, lt) in EXCL:
        continue
    roster = SUBG[sub] if sub else {'A1', 'A2', 'A3', 'A4', 'A5', 'A6'}
    ent = {mo: t for mo, ev, t in evs if ev == 'entry' and mo in roster}
    for M in sorted(roster):
        st = state.get((day, M))
        if st is None or st[0] == 'provider':
            continue
        q = qual.get((day, 'Foraging', lt, M))
        if q is None or q[0] == 1 or q[1] > 0.01:
            continue
        dur = q[2]
        others = [t for mo, t in ent.items() if mo != M]
        if not others:
            continue
        te = min(others); own = ent.get(M)
        if own is not None and (own <= te or own < te + GUARD):
            continue          # 자기가 먼저 들어갔거나, 창이 자기 진입에 오염되는 경우만 제외
        if te + WIN[0] < 0 or te + WIN[1] > dur:
            continue
        a = series(day, M, 'Foraging', lt)
        if a is None:
            continue
        vals = {}
        for nm, ri in (('PFC', IDX_PFC), ('NAc', IDX_NAC)):
            v1 = wm(a[ri, TH1], te + WIN[0], te + WIN[1])
            v2 = wm(a[ri, TH2], te + WIN[0], te + WIN[1])
            vals[nm] = (v2 - v1) if (np.isfinite(v1) and np.isfinite(v2)) else np.nan
        if not (np.isfinite(vals['PFC']) and np.isfinite(vals['NAc'])):
            continue
        hb = wm(a[IDX_PFC, BE2], te + WIN[0], te + WIN[1])
        rows.append((day, lt, sub or '', M, st[0], round(st[1], 3),
                     1 if own is not None else 0, round(te, 2),
                     round((vals['PFC'] + vals['NAc']) / 2.0, 5),
                     round(vals['PFC'], 5), round(vals['NAc'], 5),
                     round(hb, 5) if np.isfinite(hb) else ''))
    if day not in seen_days:
        seen_days.add(day); print('  day %d ... %d rows' % (day, len(rows)), flush=True)
with open(os.path.join(OUT, 'tilt_member_change.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['day', 'trial', 'subgroup', 'mouse', 'state', 'work_rate', 'follow', 'event_t',
                'tilt', 'tilt_PFC', 'tilt_NAc', 'hbeta_PFC'])
    w.writerows(rows)
print('WROTE', len(rows), flush=True)
