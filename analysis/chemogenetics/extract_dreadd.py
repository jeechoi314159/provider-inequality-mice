# -*- coding: utf-8 -*-
"""DREADD (PFC inhibition) CBRAIN 원시 → 분석용 파생 자료.
구성 변경 단계 추출기와 같은 형식으로 내보내므로 기존 분석 코드를 그대로 쓸 수 있음.

입력: dreadd/raw/PFC Inhibition/Group{E,F,G,H}/LFP/{CNO,Saline}/*.txt
  * 코호트와 조건은 **폴더 이름**을 기준으로 함(각 군 README: 파일 안의 군 표기가 한 칸 밀려 있음).
출력: dreadd/data/ 의 bandpower_whole.csv, trial_index.csv, dig_events.csv, spec_*.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
import os, re, sys, csv, time
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, welch, spectrogram
# --- 이 기록은 파일마다 필드 수가 다름(소자 2개 = 13필드, 3개 = 19필드). 형식을 파일에서 읽음.
FS = int(round(32768/32))          # 1024 Hz
SCALE = 0.0001529
_LUT = np.full(256, -1, np.int16)
for _i, _c in enumerate(b'0123456789'): _LUT[_c] = _i
for _i, _c in enumerate(b'ABCDEF'):     _LUT[_c] = 10+_i
for _i, _c in enumerate(b'abcdef'):     _LUT[_c] = 10+_i

T0 = time.time(); BUDGET = float(os.environ.get('BUDGET', '1e9'))
BASE = (CHEMO[:-1])
SRC, OUT = os.path.join(BASE, 'raw'), os.path.join(BASE, 'data')
BANDS = [('theta1', 4, 8), ('theta2', 8, 12), ('beta1', 18, 24),
         ('beta2', 24, 32), ('gamma1', 35, 50), ('gamma2', 70, 90)]
REGION = ['BLA', 'NAc', 'PFC']                      # ch1/ch2/ch3, 기존 파이프라인 가정을 물려받음
PAT = re.compile(r'Group(\w)/LFP/(CNO|Saline)/Day(\d+)_Group\w+_(?:CNO|saline|Saline)_'
                 r'(Baseline|Foraging)_Trial(\d+)_CBRAIN_([\d_]+)_file(\d+)\.txt$')


def read_fields(path):
    """반환: (F, hdr, ndev). 마지막 필드는 가변 길이 십진수이므로 앞의 6*ndev 개 4자리 필드만 읽음."""
    buf = np.fromfile(path, dtype=np.uint8)
    nl = np.flatnonzero(buf == 10)
    if len(nl) < 3:
        return np.zeros((0, 0), np.uint16), [], 0
    starts = np.concatenate(([0], nl[:-1] + 1))
    e = np.where(buf[np.maximum(nl-1, 0)] == 13, nl-1, nl)
    dl = (e - starts).astype(np.int64)
    modal = int(np.bincount(np.maximum(dl, 0)).argmax())
    i0 = int(np.flatnonzero(dl == modal)[0])
    ntok = len(buf[starts[i0]:e[i0]].tobytes().decode('latin1').split())
    ndev = (ntok - 1)//6
    if ndev < 1:
        return np.zeros((0, 0), np.uint16), [], 0
    nfix = 6*ndev
    lo, hi = 5*nfix - 1, 5*nfix + 6
    hdr = [buf[starts[i]:e[i]].tobytes().decode('latin1').split()
           for i in np.flatnonzero((dl > hi) & (dl < 200))]
    keep = np.flatnonzero((dl >= lo) & (dl <= hi))
    if not len(keep):
        return np.zeros((0, nfix), np.uint16), hdr, ndev
    off = (np.arange(nfix)*5)[None, :, None] + np.arange(4)[None, None, :]
    out = np.empty((len(keep), nfix), np.uint16)
    STEP = 200_000
    for a_ in range(0, len(keep), STEP):
        s_ = starts[keep[a_:a_+STEP]][:, None, None]
        d = _LUT[buf[s_ + off]]
        bad = (d < 0).any(2)
        v = ((d[..., 0].astype(np.int32) << 12) | (d[..., 1].astype(np.int32) << 8)
             | (d[..., 2].astype(np.int32) << 4) | d[..., 3].astype(np.int32))
        v[bad] = 0
        out[a_:a_+STEP] = v.astype(np.uint16)
    return out, hdr, ndev


def device_signals(F, dev, ndev):
    """필드 배치: ch_k = k*ndev + dev (k=0,1,2), ttl = 3*ndev+dev,
    표본번호 상위 = 4*ndev+dev, 하위 = 5*ndev+dev."""
    idx = F[:, 4*ndev+dev].astype(np.int64)*65536 + F[:, 5*ndev+dev].astype(np.int64)
    new = np.concatenate(([True], np.diff(idx) != 0))
    lfp = np.stack([(F[new, k*ndev+dev].astype(np.float32) - 32767)*SCALE for k in range(3)])
    ttl = F[new, 3*ndev+dev].astype(np.uint16)
    return lfp, ttl, idx[new]


bh, ah = butter(4, 1/(FS/2), 'highpass')
bl, al = butter(4, 300/(FS/2), 'lowpass')
bn, an = iirnotch(60, 30, FS)
prep = lambda x: filtfilt(bn, an, filtfilt(bl, al, filtfilt(bh, ah, x.astype(np.float64))))

groups = {}
for root, _dirs, fs in os.walk(SRC):
    for f in fs:
        if not f.endswith('.txt') or f.endswith('_para.txt'):
            continue
        p = os.path.join(root, f)
        m = PAT.search(p.replace(os.sep, '/'))
        if not m:
            continue
        coh, cond, day, seg, tri, rig, part = m.groups()
        groups.setdefault((coh, cond, int(day), rig, seg, int(tri)), []).append((int(part), p))
keys = sorted(groups)
print('구간 %d개' % len(keys), flush=True)
if not keys:
    print('raw 폴더가 비어 있음. PFC Inhibition 폴더를 dreadd/raw/ 로 복사할 것.', flush=True); raise SystemExit

bp_path, ix_path, dg_path = (os.path.join(OUT, x) for x in
                             ('bandpower_whole.csv', 'trial_index.csv', 'dig_events.csv'))
done = set()
if os.path.exists(bp_path):
    for r in csv.DictReader(open(bp_path)):
        done.add((r['cohort'], r['condition'], r['segment'], int(r['trial'])))
new = not os.path.exists(bp_path)
fbp, fix, fdg = (open(p, 'a', newline='') for p in (bp_path, ix_path, dg_path))
wbp, wix, wdg = csv.writer(fbp), csv.writer(fix), csv.writer(fdg)
if new:
    wbp.writerow(['cohort', 'condition', 'day', 'rig', 'segment', 'trial', 'device', 'n', 'seconds',
                  'frozen', 'artifact_frac', 'sd_ch1', 'sd_ch2', 'sd_ch3']
                 + ['%s_%s' % (g, b) for g in REGION for b, _, _ in BANDS])
    wix.writerow(['cohort', 'condition', 'day', 'rig', 'segment', 'trial', 'header', 'seconds', 'n_files', 'nfield', 'ndev'])
    wdg.writerow(['cohort', 'condition', 'segment', 'trial', 'device', 'time_s', 'level'])

store, cur, sbytes = {}, None, 0
def flush(tag):
    global store, sbytes
    if store and tag:
        i = 1
        while os.path.exists(os.path.join(OUT, 'spec_%s_p%d.npz' % (tag, i))):
            i += 1
        np.savez_compressed(os.path.join(OUT, 'spec_%s_p%d.npz' % (tag, i)), **store)
        print('  saved spec_%s_p%d.npz (%d개)' % (tag, i, len(store)), flush=True)
    store, sbytes = {}, 0

for k in keys:
    coh, cond, day, rig, seg, tri = k
    tag = '%s_%s' % (coh, cond)
    if cur != tag:
        flush(cur); cur = tag
    if (coh, cond, seg, tri) in done:
        continue
    if time.time() - T0 > BUDGET:
        print('시간 예산 소진', flush=True); break
    try:
        F, hdr0, nfs = [], None, set()
        for _, p in sorted(groups[k]):
            f_, h_, nd_ = read_fields(p)
            if f_.shape[0] == 0:
                continue
            F.append(f_); nfs.add(nd_)
            if hdr0 is None and h_:
                hdr0 = ' '.join(h_[0])
        if not F or len(nfs) != 1:
            raise ValueError('소자 수 불일치 또는 빈 파일: %s' % sorted(nfs))
        F = np.vstack(F)
        NDEV = nfs.pop(); nfield = 6*NDEV + 1
        wix.writerow([coh, cond, day, rig, seg, tri, hdr0, round(len(F)/FS, 1), len(groups[k]), nfield, NDEV]); fix.flush()
        for dev in range(NDEV):
            lfp, dig, idx = device_signals(F, dev, NDEV)
            if len(idx) < FS*5:          # 빈 소자 슬롯은 건너뜀
                continue
            ch = dig.astype(np.int16)
            for jj in np.flatnonzero(np.diff(ch) != 0)[:3000]:
                wdg.writerow([coh, cond, seg, tri, dev+1, round(float(jj+1)/FS, 4), int(ch[jj+1])])
            frozen = float((np.diff(idx) < 0).mean())
            arts, wrow, specs, b8s = [], [], [], []
            for c in range(3):
                x = prep(lfp[c])
                sd = 1.4826*np.median(np.abs(x - np.median(x)))
                arts.append(float((np.abs(x - np.median(x)) > 5*sd).mean()))
                f_, t_, S = spectrogram(x, fs=FS, window='hamming', nperseg=256, noverlap=128,
                                        detrend=False, scaling='density', mode='psd')
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
            wbp.writerow([coh, cond, day, rig, seg, tri, dev+1, len(idx), round(len(idx)/FS, 1),
                          round(frozen, 4), round(float(np.mean(arts)), 4)]
                         + [round(float(lfp[c].std()), 5) for c in range(3)]
                         + ['%.6g' % v for v in wrow])
            a, b = np.stack(specs), np.stack(b8s)
            store['%s_T%d_dev%d_spec' % (seg, tri, dev+1)] = a
            store['%s_T%d_dev%d_band' % (seg, tri, dev+1)] = b
            sbytes += a.nbytes + b.nbytes
            del lfp, specs, b8s
        fbp.flush(); fdg.flush(); del F
        if sbytes > 90e6:
            flush(cur)
        print('ok %s %s trial %d' % (tag, seg, tri), flush=True)
    except Exception as e:
        print('FAIL %s : %s' % (str(k), e), flush=True)
flush(cur)
print('완료 (%.0f s)' % (time.time()-T0), flush=True)
