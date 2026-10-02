# -*- coding: utf-8 -*-
"""fig. S10 — 추종자(follower)의 사건 정렬 스펙트로그램: 누가 먼저 들어갔는가에 따른 반응.
자료: data/edspec/ (2026-09-01 ED_spectro 추출; Powerspect_for_foraging.mat 의 Start_act/Act_ride 단계, 1–100 Hz, 8 bins/s)
  ed_spectro_{PFC,NAc,BLA}.csv + ed_spectro_trials_meta.csv       : 동료(선두) 진입 정렬, −2…+3 s
  ed_spectro_own_{PFC,NAc,BLA}.csv + ed_spectro_own_trials_meta.csv : 추종자 자기 진입 정렬, −2…+2 s
재분석(본 연구 규약): 시행별 가우스 평활(σ 1.5 bin, 2 Hz) → −2…−1 s 기저 차감 → 군집 기반 순열 검정
  (군집 형성 양측 p < 0.05, 군집 수준 α 0.05, 2,000 순열; 가족별 보정 = 한 행(세 영역)의 최대 군집 질량).
  행 1–3 부호 뒤집기 일표본, 행 4 라벨 섞기 이표본. 실선 = 가족별 P < 0.05; 점선 = 자기 지도 안에서만 P < 0.05.
출력: png/pdf figS10, data/figS10_follower_clusters.csv"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats, ndimage
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
ED = os.path.join(D, 'edspec')
CM = F.CM
DIV = LinearSegmentedColormap.from_list('div', ['#1f4e8c', F.GAM, '#ffffff', F.PROV, '#8f3c0c'])
REG = ['PFC', 'NAc', 'BLA']
ROWS = [('P follows NP', 'Provider follows another mouse (one animal'), ('NP follows P', 'Another mouse follows the provider (five mice'),
        ('NP follows NP', 'Another mouse follows another mouse (five mice')]
NPERM = int(os.environ.get('NPERM', '2000')); ALPHA = 0.05; PFORM = 0.05
SIG_T, SIG_F = 1.5, 2.0
rng = np.random.default_rng(20261001)


def load_block(prefix):
    meta = pd.read_csv(os.path.join(ED, prefix + 'trials_meta.csv'))
    X = {}
    for reg in REG:
        M = pd.read_csv(os.path.join(ED, prefix + reg + '.csv'))
        t = np.sort(M.t.unique()); nT = len(t); n = int(M.idx.max())
        A = np.full((n, nT, 100), np.nan)
        M = M.sort_values(['idx', 't'])
        vals = M.iloc[:, 2:].values.reshape(n, nT, 100)
        A[:] = vals
        # 평활 (시행별) 후 기저(−2…−1 s) 차감
        A = ndimage.gaussian_filter(A, sigma=(0, SIG_T, SIG_F), mode='nearest')
        bl = (t >= -2.0) & (t < -1.0)
        A = A - A[:, bl, :].mean(axis=1, keepdims=True)
        X[reg] = A
    return meta, t, X


def tmap_one(A):
    m = A.mean(0); s = A.std(0, ddof=1); n = A.shape[0]
    return m / (s / np.sqrt(n) + 1e-12)


def tmap_two(A, B):
    return stats.ttest_ind(A, B, axis=0, equal_var=False).statistic


def clusters(tm, thr):
    """양·음 군집 라벨과 질량."""
    out = []
    for sign in (1, -1):
        mask = (sign * tm) > thr
        lab, k = ndimage.label(mask)
        for i in range(1, k + 1):
            m_ = lab == i; out.append((sign, m_, float(np.abs(tm[m_]).sum())))
    return out


def maxmass(tm, thr):
    best = 0.0
    for sign in (1, -1):
        mask = (sign * tm) > thr
        if not mask.any(): continue
        lab, k = ndimage.label(mask)
        if k:
            masses = ndimage.sum(np.abs(tm), lab, index=np.arange(1, k + 1))
            best = max(best, float(np.max(masses)))
    return best


def perm_one(A, thr, nperm, batch=200):
    """부호 뒤집기 일표본 순열 (행렬곱으로 일괄 계산). A: n × T × F."""
    n, T_, F_ = A.shape; Af = A.reshape(n, -1).astype(np.float64); sumsq = (Af ** 2).sum(0)
    mx = np.zeros(nperm); done = 0
    while done < nperm:
        b = min(batch, nperm - done)
        S = rng.choice([-1.0, 1.0], size=(b, n))
        M = S @ Af / n                                   # b × cells
        var = (sumsq[None, :] - n * M ** 2) / (n - 1)
        tm = M / (np.sqrt(np.maximum(var, 1e-12) / n))
        for i in range(b):
            mx[done + i] = maxmass(tm[i].reshape(T_, F_), thr)
        done += b
    return mx


def perm_two(A, B, thr, nperm, batch=200):
    """라벨 섞기 이표본 순열 (Welch t), 행렬곱 일괄."""
    AB = np.concatenate([A, B], 0); n = AB.shape[0]; na = A.shape[0]; nb = n - na
    T_, F_ = AB.shape[1:]; Xf = AB.reshape(n, -1).astype(np.float64); X2 = Xf ** 2
    tot = Xf.sum(0); tot2 = X2.sum(0)
    mx = np.zeros(nperm); done = 0
    while done < nperm:
        b = min(batch, nperm - done)
        I = np.zeros((b, n))
        for i in range(b):
            I[i, rng.permutation(n)[:na]] = 1.0
        s1 = I @ Xf; q1 = I @ X2; m1 = s1 / na; v1 = (q1 - na * m1 ** 2) / (na - 1)
        s2 = tot[None, :] - s1; q2 = tot2[None, :] - q1; m2 = s2 / nb; v2 = (q2 - nb * m2 ** 2) / (nb - 1)
        tm = (m1 - m2) / np.sqrt(np.maximum(v1, 1e-12) / na + np.maximum(v2, 1e-12) / nb)
        for i in range(b):
            mx[done + i] = maxmass(tm[i].reshape(T_, F_), thr)
        done += b
    return mx


def analyse(meta, t, X, test_from=-0.5):
    """행(4) × 영역(3) 지도와 군집. 가족별 보정은 한 행의 세 영역 null 최대값 합성(각 순열에서 세 영역 최대)."""
    case = meta.Case.values
    tsel = t >= test_from
    res = {}
    for ri, (key, lab) in enumerate(ROWS + [('diff', 'difference')]):
        tmaps, cls, nulls = {}, {}, {}
        for reg in REG:
            A = X[reg][:, tsel, :]
            if key != 'diff':
                S = A[case == key]
                n = S.shape[0]; dfree = n - 1
                thr = stats.t.ppf(1 - PFORM / 2, dfree)
                tm = tmap_one(S); nulls[reg] = perm_one(S, thr, NPERM)
            else:
                P_ = A[case == 'NP follows P']; N_ = A[case == 'NP follows NP']
                dfree = min(P_.shape[0], N_.shape[0]) - 1
                thr = stats.t.ppf(1 - PFORM / 2, dfree)
                tm = tmap_two(P_, N_); nulls[reg] = perm_two(P_, N_, thr, NPERM)
            tmaps[reg] = tm; cls[reg] = clusters(tm, thr)
        fam = np.max(np.column_stack([nulls[r] for r in REG]), axis=1)   # 행 전체(세 영역) 가족별 null
        rows = []
        for reg in REG:
            for sign, m_, mass in cls[reg]:
                p_map = (np.sum(nulls[reg] >= mass) + 1) / (NPERM + 1)
                p_fam = (np.sum(fam >= mass) + 1) / (NPERM + 1)
                tt = t[tsel]; ti = np.where(m_.any(1))[0]; fi = np.where(m_.any(0))[0]
                rows.append(dict(row=key, region=reg, sign='+' if sign > 0 else '−', mass=mass, P_map=p_map, P_family=p_fam,
                                 t_start=tt[ti.min()], t_end=tt[ti.max()], f_lo=fi.min() + 1, f_hi=fi.max() + 1, mask=m_))
        res[key] = dict(tmaps=tmaps, clusters=rows)
    return res


def mean_maps(meta, X):
    case = meta.Case.values; out = {}
    for key, _ in ROWS:
        out[key] = {reg: X[reg][case == key].mean(0) for reg in REG}
    out['diff'] = {reg: X[reg][case == 'NP follows P'].mean(0) - X[reg][case == 'NP follows NP'].mean(0) for reg in REG}
    out['n'] = {key: int((case == key).sum()) for key, _ in ROWS}
    return out


def build():
    blocks = [('ed_spectro_', 'Aligned to the first entrant’s entry', 'Time from first entrant’s entry (s)', 0.3, 3.0),
              ('ed_spectro_own_', 'Aligned to the follower’s own entry', 'Time from own entry (s)', 0.5, 2.0)]
    allrows = []
    fig = F.figure(F.W3, 19.6 * CM)
    G_ = F.Grid(fig, 4, 7, left=0.06, right=0.10, top=1.05, bottom=1.95, wgap=0.22, hgap=0.55,
                wratios=[1, 1, 1, 0.55, 1, 1, 1])
    for b, (prefix, title, xlab, vmax, tmax) in enumerate(blocks):
        meta, t, X = load_block(prefix)
        res = analyse(meta, t, X); mm = mean_maps(meta, X)
        tsel = t >= -0.5; tt = t[tsel]
        c0 = 0 if b == 0 else 4
        for ri, (key, lab) in enumerate(ROWS + [('diff', 'Others: follows the provider − follows another')]):
            for c, reg in enumerate(REG):
                ax = G_.ax(ri, c0 + c, padl=1.15 if c == 0 else 0.30, padb=1.05 if ri == 3 else 0.20, padr=0.08, padt=0.55)
                ax.set_label('<sub b%dr%dc%d>' % (b, ri, c))
                Z = mm[key][reg]      # nT × 100
                ax.imshow(Z.T, aspect='auto', origin='lower', cmap=DIV, vmin=-vmax, vmax=vmax,
                          extent=[t[0], t[-1] + 0.125, 1, 100], interpolation='nearest')
                # 군집 윤곽
                for r_ in res[key]['clusters']:
                    if r_['region'] != reg: continue
                    if r_['P_map'] >= ALPHA: continue
                    full = np.zeros((len(t), 100), bool); full[np.ix_(tsel, np.arange(100))] = r_['mask']
                    T_, F_ = np.meshgrid(t + 0.0625, np.arange(1, 101), indexing='ij')
                    ax.contour(T_, F_, full.astype(float), levels=[0.5], colors=F.INK, linewidths=0.8,
                               linestyles='solid' if r_['P_family'] < ALPHA else 'dotted')
                    allrows.append(dict(alignment=title, row=key, region=reg, n=mm['n'].get(key, 0), **{k: v for k, v in r_.items() if k not in ('mask', 'row', 'region')}))
                ax.axvline(0, color=F.INK, lw=0.6, ls='--')
                ax.set_xlim(-2, tmax); ax.set_ylim(1, 100); ax.set_yticks([20, 60, 100])
                if c == 0:
                    ax.set_ylabel('Frequency (Hz)', labelpad=1.0)
                else:
                    ax.set_yticklabels([])
                if ri == 3:
                    ax.set_xticks([-2, 0, 2] if tmax > 2.5 else [-2, 0, 2]); ax.set_xticklabels(['−2', '0', '2'])
                    if c == 1: ax.set_xlabel(xlab, labelpad=1.5)
                else:
                    ax.set_xticks([])
                if ri == 0:
                    ax.set_title(reg, fontsize=7.5, pad=2.5)
                for s_ in ax.spines.values():
                    s_.set_visible(True); s_.set_linewidth(0.5)
                ax.tick_params(axis='y', pad=1.5)
            x, y, w, h = G_.cell_cm(ri, c0)
            nlab = ('; %d trials)' % mm['n'][key]) if key != 'diff' else (' (%d vs %d trials)' % (mm['n']['NP follows P'], mm['n']['NP follows NP']))
            fig.text((x + 1.15) / G_.W, (y + h - 0.02) / G_.H, '%s%s' % (lab, nlab), ha='left', va='bottom', fontsize=7.0, color=F.INK)
        # 블록 제목·패널 문자·색막대
        x, y, w, h = G_.cell_cm(0, c0, cs=3)
        fig.text((x + 1.15) / G_.W, (y + h + 0.40) / G_.H, title, ha='left', va='bottom', fontsize=8.0, color=F.INK)
        G_.label(0, c0, 'A' if b == 0 else 'B')
        x, y, w, h = G_.cell_cm(3, c0, cs=3)
        cax = fig.add_axes([(x + 1.15 + 0.6) / G_.W, (y - 0.85) / G_.H, (w - 1.15 - 1.4) / G_.W, 0.16 / G_.H], label='<sub-cbar>')
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap=DIV), cax=cax, orientation='horizontal', ticks=[-vmax, 0, vmax])
        cb.outline.set_linewidth(0.4); cax.tick_params(length=1.6, pad=1.0, labelsize=6.5)
        cb.set_ticklabels(['−%g' % vmax, '0', '%g' % vmax])
        cax.set_xlabel('Change in z-scored power from −2 to −1 s', fontsize=6.5, labelpad=1.5)
        # 저장용 수치
        pd.DataFrame(allrows).drop(columns=[c for c in ('mask',) if c in pd.DataFrame(allrows).columns]).to_csv(os.path.join(D, 'figS10_follower_clusters.csv'), index=False)
    # 범례(실선/점선)
    fig.text(0.5, 0.10 / G_.H, 'Solid contour, cluster with family-wise P < 0.05 over the three regions of its row; dotted, P < 0.05 within its own map only',
             ha='center', va='bottom', fontsize=6.5, color=F.INK)
    return F.save(fig, 'figS10', PNG, PDF)


if __name__ == '__main__':
    print('saved', build())
    T = pd.read_csv(os.path.join(D, 'figS10_follower_clusters.csv'))
    print(T[['alignment', 'row', 'region', 'n', 'sign', 't_start', 't_end', 'f_lo', 'f_hi', 'mass', 'P_map', 'P_family']].to_string())
