# -*- coding: utf-8 -*-
"""구성 변경 단계 CBRAIN 원시 → 분석용 파생 자료 (v3: 단계 이름 일반화, 시간 예산).

`~/mnt/member perturbation` 안의 `Day<N>_<phase>_CBRAIN_<rig>` 폴더를 모두 처리한다.
같은 산출 형식으로 이어 붙이므로 여러 번 나누어 실행해도 됨 (이미 처리한 구간은 건너뜀).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, re, sys, csv, time
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, welch, spectrogram
sys.path.insert(0, (WORK[:-1]))
from solitary_lfp import read_fields, device_signals, FS

T0 = time.time()
BUDGET = float(os.environ.get('BUDGET', '150'))
SRC = (RAW_LFP + 'member_perturbation')
OUT = (LFP + 'member_change')
REGION = ['BLA', 'NAc', 'PFC']   # 주의: 실제 대응은 ch1=PFC, ch2=NAc, ch3=BLA (../CHANNEL_ORDER.md). 기존 산출물과의 호환을 위해 이름은 유지함
BANDS = [('theta1', 4, 8), ('theta2', 8, 12), ('beta1', 18, 24),
         ('beta2', 24, 32), ('gamma1', 35, 50), ('gamma2', 70, 90)]
RIGMAP = {'1_2_3_7': {1: 'A3', 2: 'A1', 3: 'A2'},
          '4_5_0_8': {1: 'A4', 2: 'A5', 3: 'A6'}}
PAT = re.compile(
    r'Day(\d+)_.*?_(Baseline|Foraging)_'
    r'(?:(?:(Active|Passive)_)?Trial(\d+)_CBRAIN_[\d_]+|Individual_#(\d)(?:_CBRAIN_[\d_]+)?)'
    r'_file(\d+)\.txt$')
TTLPAT = re.compile(r'Day(\d+)_.*?_TTL_CBRAIN_([\d_]+)_file(\d+)\.txt$')
SUB = {None: 0, 'Active': 100, 'Passive': 200}          # 분리 단계 하위집단은 시행번호에 가산

bh, ah = butter(4, 1/(FS/2), 'highpass')
bl, al = butter(4, 300/(FS/2), 'lowpass')
bn, an = iirnotch(60, 30, FS)
prep = lambda x: filtfilt(bn, an, filtfilt(bl, al, filtfilt(bh, ah, x.astype(np.float64))))

groups, ttls, phase_of = {}, {}, {}
for d in sorted(os.listdir(SRC)):
    if not d.startswith('Day'):
        continue
    rig = d.split('CBRAIN_')[-1]
    day0 = int(re.match(r'Day(\d+)', d).group(1))
    phase_of[day0] = d.split('_CBRAIN')[0].split('_', 1)[1]
    for f in sorted(os.listdir(os.path.join(SRC, d))):
        if '_Food_' in f or f.endswith('_para.txt'):
            continue
        p = os.path.join(SRC, d, f)
        m = PAT.search(f)
        if m:
            day, seg, sub, tri, ind, part = m.groups()
            t = int(tri) + SUB[sub] if tri else -int(ind) - 1
            groups.setdefault((int(day), rig, seg, t), []).append((int(part), p))
            continue
        t = TTLPAT.search(f)
        if t:
            ttls.setdefault((int(t.group(1)), rig), []).append((int(t.group(3)), p))

keys = sorted(groups)
print('폴더 단계: %s' % phase_of, flush=True)
print('구간 %d개, TTL %d개' % (len(keys), len(ttls)), flush=True)

bp_path, ix_path, dg_path = (os.path.join(OUT, x) for x in
                             ('bandpower_whole.csv', 'trial_index.csv', 'dig_events.csv'))
hdr = (['day', 'rig', 'segment', 'trial', 'device', 'mouse', 'n', 'seconds', 'frozen',
        'artifact_frac', 'sd_ch1', 'sd_ch2', 'sd_ch3']
       + ['%s_%s' % (g_, b) for g_ in REGION for b, _, _ in BANDS])
newbp = not os.path.exists(bp_path)
fbp = open(bp_path, 'a', newline=''); wbp = csv.writer(fbp)
if newbp:
    wbp.writerow(hdr)
fix = open(ix_path, 'a', newline=''); wix = csv.writer(fix)
fdg = open(dg_path, 'a', newline=''); wdg = csv.writer(fdg)

done = set()
if not newbp:
    with open(bp_path) as fh:
        for r in csv.DictReader(fh):
            done.add((int(r['day']), r['rig'], r['segment'], int(r['trial'])))
print('이미 처리 %d' % len(done), flush=True)

store, sbytes, part, cur = {}, 0, {}, None


def flush(tag):
    global store, sbytes
    if not store or tag is None:
        return
    part[tag] = part.get(tag, 0) + 1
    while os.path.exists(os.path.join(OUT, 'spec_%s_p%d.npz' % (tag, part[tag]))):
        part[tag] += 1
    np.savez_compressed(os.path.join(OUT, 'spec_%s_p%d.npz' % (tag, part[tag])), **store)
    print('   저장 spec_%s_p%d.npz (%d키, %.0f MB)' % (tag, part[tag], len(store),
                                                     sbytes/1e6), flush=True)
    store, sbytes = {}, 0


for k in keys:
    day, rig, seg, tri = k
    tag = 'Day%d_%s' % (day, rig)
    if cur != tag:
        flush(cur); cur = tag
    if (day, rig, seg, tri) in done:
        continue
    if time.time() - T0 > BUDGET:
        print('시간 예산 소진', flush=True); break
    try:
        F, hdr0 = [], None
        for _, p in sorted(groups[k]):
            f_, h_ = read_fields(p)
            F.append(f_)
            if hdr0 is None and h_:
                hdr0 = ' '.join(h_[0])
        F = np.vstack(F)
        wix.writerow([day, rig, seg, tri, hdr0, round(len(F)/FS, 1), len(groups[k])]); fix.flush()
        for dev in range(3):
            lfp, dig, idx = device_signals(F, dev)
            ch = dig.astype(np.int16)
            for jj in np.flatnonzero(np.diff(ch) != 0)[:3000]:
                wdg.writerow([day, rig, seg, tri, dev+1, round(float(jj+1)/FS, 4), int(ch[jj+1])])
            frozen = float((np.diff(idx) < 0).mean())
            arts, wrow, specs, b8s = [], [], [], []
            for c in range(3):
                x = prep(lfp[c])
                sd = 1.4826*np.median(np.abs(x - np.median(x)))
                arts.append(float((np.abs(x - np.median(x)) > 5*sd).mean()))
                f_, t_, S = spectrogram(x, fs=FS, window='hamming', nperseg=256,
                                        noverlap=128, detrend=False, scaling='density',
                                        mode='psd')
                e = np.searchsorted(f_, np.arange(0.5, 121, 1.0))
                Sb = np.add.reduceat(S, e[:-1], axis=0) / np.maximum(np.diff(e), 1)[:, None]
                b8 = np.stack([S[(f_ >= lo) & (f_ < hi)].mean(0) for _, lo, hi in BANDS])
                if seg == 'Baseline':
                    m_ = (Sb.shape[1]//8)*8
                    Sb = Sb[:, :m_].reshape(Sb.shape[0], -1, 8).mean(2)
                    b8 = b8[:, :m_].reshape(b8.shape[0], -1, 8).mean(2)
                specs.append(np.log10(np.maximum(Sb, 1e-20)).astype(np.float16))
                b8s.append(b8.astype(np.float32))
                fw, Pw = welch(x, fs=FS, nperseg=FS*2, noverlap=FS, detrend='constant')
                wrow += [float(np.trapezoid(Pw[(fw >= lo) & (fw < hi)], fw[(fw >= lo) & (fw < hi)]))
                         for _, lo, hi in BANDS]
            mouse = RIGMAP[rig][dev+1]
            wbp.writerow([day, rig, seg, tri, dev+1, mouse, len(idx), round(len(idx)/FS, 1),
                          round(frozen, 4), round(float(np.mean(arts)), 4)]
                         + [round(float(lfp[c].std()), 5) for c in range(3)]
                         + ['%.6g' % v for v in wrow])
            a, b = np.stack(specs), np.stack(b8s)
            store['%s_T%d_%s_spec' % (seg, tri, mouse)] = a
            store['%s_T%d_%s_band' % (seg, tri, mouse)] = b
            sbytes += a.nbytes + b.nbytes
            del lfp, specs, b8s
        fbp.flush(); fdg.flush(); del F
        if sbytes > 90e6:
            flush(cur)
        print('ok %s %s trial %d' % (tag, seg, tri), flush=True)
    except Exception as e:
        print('FAIL %s : %s' % (str(k), e), flush=True)
flush(cur)

tp = os.path.join(OUT, 'ttl_events.csv')
have = set()
if os.path.exists(tp):
    with open(tp) as fh:
        for r in csv.DictReader(fh):
            have.add((int(r['day']), r['rig']))
with open(tp, 'a', newline='') as fh:
    w = csv.writer(fh)
    if os.path.getsize(tp) == 0:
        w.writerow(['day', 'rig', 'device', 'time_s', 'level'])
    for (day, rig), ps in sorted(ttls.items()):
        if (day, rig) in have or time.time() - T0 > BUDGET + 40:
            continue
        try:
            F = np.vstack([read_fields(p)[0] for _, p in sorted(ps)])
            for dev in range(4):
                _l, dig, _i = device_signals(F, dev)
                ch = dig.astype(np.int16)
                for jj in np.flatnonzero(np.diff(ch) != 0)[:20000]:
                    w.writerow([day, rig, dev+1, round(float(jj+1)/FS, 4), int(ch[jj+1])])
            del F
            print('TTL Day%d %s' % (day, rig), flush=True)
        except Exception as e:
            print('TTL FAIL %d %s : %s' % (day, rig, e), flush=True)
fbp.close(); fix.close(); fdg.close()
print('DONE', flush=True)
