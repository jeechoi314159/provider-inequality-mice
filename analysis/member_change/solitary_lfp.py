# -*- coding: utf-8 -*-
"""단독 채집 원시 LFP 파서 (CBRAIN 무선, main30 3ch sync)."""
import os, sys, json
import numpy as np

FS      = int(round(32768/32))          # 1024 Hz
SCALE   = 0.0001529
NDEV    = 4
NFIELD  = 24
BANDS   = {'low_theta':(4,8), 'high_theta':(8,12), 'low_beta':(12,20),
           'high_beta':(20,32), 'low_gamma':(32,50), 'high_gamma':(51,100)}

_LUT = np.full(256, -1, np.int16)
for i, c in enumerate(b'0123456789'):   _LUT[c] = i
for i, c in enumerate(b'ABCDEF'):       _LUT[c] = 10+i
for i, c in enumerate(b'abcdef'):       _LUT[c] = 10+i


def read_fields(path):
    buf = np.fromfile(path, dtype=np.uint8)
    nl = np.flatnonzero(buf == 10)
    if len(nl) < 2:
        return np.zeros((0, NFIELD), np.uint16), []
    starts = np.concatenate(([0], nl[:-1] + 1))
    lens   = nl - starts
    hdr    = [buf[starts[i]:nl[i]].tobytes().decode('latin1').split()
              for i in np.flatnonzero((lens > 40) & (lens < 120))]
    keep   = np.flatnonzero((lens >= 120) & (lens <= 132))
    if not len(keep):
        return np.zeros((0, NFIELD), np.uint16), hdr
    off = (np.arange(NFIELD)*5)[None, :, None] + np.arange(4)[None, None, :]
    out = np.empty((len(keep), NFIELD), np.uint16)
    STEP = 200_000
    for a in range(0, len(keep), STEP):
        s = starts[keep[a:a+STEP]][:, None, None]
        d = _LUT[buf[s + off]]
        bad = (d < 0).any(2)
        v = ((d[..., 0].astype(np.int32) << 12) | (d[..., 1].astype(np.int32) << 8)
             | (d[..., 2].astype(np.int32) << 4) | d[..., 3].astype(np.int32))
        v[bad] = 0
        out[a:a+STEP] = v.astype(np.uint16)
    return out, hdr


def device_signals(F, dev):
    idx = F[:, 16+dev].astype(np.int64)*65536 + F[:, 20+dev].astype(np.int64)
    new = np.concatenate(([True], np.diff(idx) != 0))
    lfp = np.stack([(F[new, 4*k+dev].astype(np.float32) - 32767)*SCALE for k in range(3)])
    ttl = F[new, 12+dev].astype(np.uint16)
    return lfp, ttl, idx[new]


def band_power(x, fs=FS, nper=None):
    from scipy.signal import welch
    nper = nper or fs*2
    if len(x) < nper:
        return {b: np.nan for b in BANDS}
    f, P = welch(x, fs=fs, nperseg=nper, noverlap=nper//2, detrend='constant')
    return {b: float(np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))
            for b, (lo, hi) in BANDS.items()}


def band_series(x, fs=FS):
    """8 Hz 간격 대역파워 시계열 (0.5 s 창)."""
    from scipy.signal import spectrogram
    win, step = fs//2, fs//8
    f, t, S = spectrogram(x, fs=fs, nperseg=win, noverlap=win-step, detrend='constant')
    return t, {b: S[(f >= lo) & (f < hi)].mean(0) for b, (lo, hi) in BANDS.items()}


def qc(path):
    F, hdr = read_fields(path)
    print(os.path.basename(path)[:78])
    print('  headers %d %s   rows %d  (%.1f s)' % (len(hdr), hdr[:2], len(F), len(F)/FS))
    for d in range(NDEV):
        lfp, ttl, idx = device_signals(F, d)
        g = np.diff(idx)
        print('   dev%d n=%6d (%5.1f%%) idx %10d..%10d  step med %d  gap>1 %5d  '
              'sd %.4f/%.4f/%.4f  flat %.1f%%  TTL uniq %s  trans %d'
              % (d+1, len(idx), 100*len(idx)/max(len(F), 1), idx[0], idx[-1],
                 int(np.median(g)) if len(g) else 0, int((g > 1).sum()),
                 lfp[0].std(), lfp[1].std(), lfp[2].std(),
                 100*np.mean(np.abs(np.diff(lfp[0])) < 1e-9),
                 np.unique(ttl)[:6].tolist(), int((np.diff(ttl.astype(int)) != 0).sum())))
    for d in range(NDEV):
        lfp, ttl, idx = device_signals(F, d)
        for k in range(3):
            bp = band_power(lfp[k])
            print('   dev%d ch%d' % (d+1, k+1),
                  ' '.join('%s=%.4f' % (b[:2]+b.split('_')[1][:2], v) for b, v in bp.items()))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        qc(p)
