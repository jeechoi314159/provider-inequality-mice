# -*- coding: utf-8 -*-
"""fig. S8 · fig. S9 — 코호트 A LFP 종합 그림 (원시 사건 배열 S8–S11 대체).
자료: Nature 2025-02 투고 Source data 의 평균 스펙트로그램(J. Lee, 2025-02; data/nat2025_*.mat|xlsx)과
이 프로젝트의 군집 검정 출력(data/event_clusters.csv, pair_clusters.csv).
figS08: 상태(회수·진입·불참)별 시간정규화 스펙트로그램(A), 진입 구간 PSD(B–D), 대역×구간 상태 차이(E–F).
figS09: 진입 정렬(A 회수 개체, B 진입만 한 개체)·회수 정렬(C) 스펙트로그램, 군집 검정 요약(D)."""
import os, sys, warnings
import numpy as np, pandas as pd
import scipy.io as sio
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
CM = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
TIGHTY = dict(axis='y', pad=1.5)
# 발산 색: 파랑(GAM) – 흰색 – 주황(PROV); 적록 병용 없음
DIV = LinearSegmentedColormap.from_list('div', ['#1f4e8c', F.GAM, '#ffffff', F.PROV, '#8f3c0c'])
REG = ['PFC', 'NAc', 'BLA']
STATE = [('Working', 'Retrieved'), ('Participating', 'Entered'), ('Freeriding', 'Stayed out')]
SCOL = {'Retrieved': F.PROV, 'Entered': F.OTH, 'Stayed out': F.INK}
PHASE = ['Pre', 'Entry', 'Mount', 'Carry', 'Eat']
PH_KEY = ['Pre_act', 'Act', 'Get', 'Delivery', 'Eat']
BANDS = [('low_beta', 'β 18–24'), ('high_beta', 'β 24–32'), ('low_gamma', 'γ 35–50'), ('high_gamma', 'γ 70–90')]


SPEC_H = 2.50                 # 스펙트로그램 축 높이 (cm), fig. S8·S9 공통 (2026-10-02)
SPEC_YLIM, SPEC_YT = (1, 100), [20, 60, 100]   # 주파수 축 공통


def mat(name):
    return sio.loadmat(os.path.join(D, name), squeeze_me=True, struct_as_record=False)


def cbar(fig, rect, vmin, vmax, label, ticks, norm=None):
    cax = fig.add_axes(rect, label='<sub-cbar>')
    cb = fig.colorbar(ScalarMappable(norm=norm or Normalize(vmin, vmax), cmap=DIV), cax=cax, orientation='vertical', ticks=ticks)
    cb.outline.set_linewidth(0.4); cax.tick_params(length=1.6, pad=1.0, labelsize=6.5)
    cax.set_ylabel(label, fontsize=7.0, labelpad=1.5)
    return cax


# ============================================================================ fig. S8
def figS08():
    S = mat('nat2025_state_spectrograms.mat')['Time_norm_Data']
    fs = np.asarray(S.fs); keep = fs <= 120
    # 2026-10-02 red notes: 스펙트로그램 축 높이를 모든 행에서 SPEC_H(2.50 cm)로 같게(BLA 행 포함), 주파수 축 1–100 Hz 를 fig. S9 와 공통으로.
    # 셀 높이 = SPEC_H + padt(0.42) + padb(0.18, 아래 행 1.20). 열지도 행 높이는 이전과 같음(4.80 cm).
    hr = [SPEC_H + 0.60, SPEC_H + 0.60, SPEC_H + 1.62, 4.803]
    fig = F.figure(F.W3, (sum(hr) + 0.60 + 3 * 0.30) * CM)
    # 격자: 위 3행(스펙트로그램 3열 + PSD 1열), 아래 1행(차이 열지도 2개)
    G_ = F.Grid(fig, 4, 4, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.30, hgap=0.30,
                wratios=[1, 1, 1, 1.15], hratios=hr)
    NORM = TwoSlopeNorm(vcenter=0.0, vmin=-0.3, vmax=0.8)
    for r, reg in enumerate(REG):
        for c, (key, lab) in enumerate(STATE):
            ax = G_.ax(r, c, padl=1.25 if c == 0 else 0.35, padb=1.20 if r == 2 else 0.18, padr=1.05 if c == 2 else 0.10, padt=0.42)
            ax.set_label('<sub r%dc%d>' % (r, c))
            Z = np.asarray(getattr(getattr(S, key), reg))[keep, :]
            ax.imshow(Z, aspect='auto', origin='lower', cmap=DIV, norm=NORM,
                      extent=[0, 10, fs[keep].min(), fs[keep].max()], interpolation='nearest')
            for xb in (2, 4, 6, 8):
                ax.axvline(xb, color=F.INK, lw=0.5)
            ax.set_xlim(0, 10); ax.set_ylim(*SPEC_YLIM); ax.set_yticks(SPEC_YT)
            if c == 0:
                ax.set_ylabel('%s\nFrequency (Hz)' % reg, labelpad=1.0)
            else:
                ax.set_yticklabels([])
            if r == 2:
                ax.set_xticks([1, 3, 5, 7, 9]); ax.set_xticklabels(PHASE, fontsize=5.8)
                ax.tick_params(axis='x', length=0, pad=2.0)
                if c == 1:
                    ax.set_xlabel('Trial phase (time-normalised)', labelpad=1.5)
            else:
                ax.set_xticks([])
            if r == 0:
                ax.set_title(lab, fontsize=7.5, color=SCOL[lab], pad=2.5)
            ax.tick_params(**TIGHTY)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(True)
            for s_ in ax.spines.values():
                s_.set_linewidth(0.5)
    # 색막대 (스펙트로그램 오른쪽)
    x, y, w, h = G_.cell_cm(0, 2, rs=3)
    cbar(fig, [(x + w - 0.92) / G_.W, (y + 1.20 + 1.0) / G_.H, 0.16 / G_.W, (h - 1.20 - 0.42 - 2.0) / G_.H],
         -0.3, 0.8, 'Power (within-trial z)', [-0.3, 0, 0.4, 0.8], norm=NORM)
    # ---- B–D PSD (진입 구간)
    X = pd.ExcelFile(os.path.join(D, 'nat2025_psd_act.xlsx'))
    for r, (reg, sh) in enumerate(zip(REG, ['b', 'c', 'd'])):
        ax = G_.ax(r, 3, padl=1.45, padb=1.20 if r == 2 else 0.18, padr=0.12, padt=0.42)
        d = X.parse(sh, header=None).iloc[2:, :6].astype(float).values
        f = np.arange(1, d.shape[0] + 1)
        for k, (key, lab) in enumerate(STATE):
            m, se = d[:, 2 * k], d[:, 2 * k + 1]
            ax.plot(f, m, color=SCOL[lab], lw=0.9, label=lab if r == 0 else None)
            ax.fill_between(f, m - se, m + se, color=SCOL[lab], alpha=0.18, lw=0)
        ax.set_xlim(1, 120); ax.set_ylim(-0.1, 0.65); ax.set_yticks([0, 0.3, 0.6])
        ax.axhline(0, color=F.GRID, lw=0.5)
        F.tidy(ax, 'Frequency (Hz)' if r == 2 else None, 'Power, entry to mount\n(within-trial z)' if r == 1 else None)
        if r < 2:
            ax.set_xticklabels([])
        F.note(ax, reg, x=0.97, y=0.95, ha='right', va='top', size=7.0)
        ax.tick_params(**TIGHTY)
        if r == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.30, 1.02), fontsize=6.5, borderpad=0.1, labelspacing=0.2, handlelength=1.0)
    # ---- E–F 대역 × 구간 차이 열지도 (시행 합산, 서술형)
    B = mat('nat2025_band_by_phase.mat')['DATA']
    rows = [(reg, bk, bl) for reg in REG for bk, bl in BANDS]
    def diff_map(a, b):
        M = np.zeros((len(rows), 5)); Pv = np.ones((len(rows), 5))
        for i, (reg, bk, bl) in enumerate(rows):
            for j, ph in enumerate(PH_KEY):
                node = getattr(getattr(B, ph), reg)
                va = np.asarray(getattr(getattr(node, a), bk), float); vb = np.asarray(getattr(getattr(node, b), bk), float)
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                M[i, j] = va.mean() - vb.mean()
                Pv[i, j] = stats.mannwhitneyu(va, vb, alternative='two-sided').pvalue if len(va) > 2 and len(vb) > 2 else np.nan
        return M, Pv
    VM2 = 0.3
    for c, (a, b, ttl, let) in enumerate((('w', 'f', 'Retrieved − stayed out', 'E'), ('p', 'f', 'Entered − stayed out', 'F'))):
        ax = G_.ax(3, 2 * c, cs=2, padl=1.75, padb=1.05, padr=1.25 if c == 1 else 0.35, padt=0.50)
        ax.set_label('<sub heat%d>' % c)
        M, Pv = diff_map(a, b)
        ax.imshow(M, aspect='auto', cmap=DIV, vmin=-VM2, vmax=VM2, interpolation='nearest')
        for i in range(len(rows)):
            for j in range(5):
                if np.isfinite(Pv[i, j]) and Pv[i, j] < 0.001:
                    ax.text(j, i, '•', ha='center', va='center', fontsize=6.0, color=F.INK)
        ax.set_xticks(range(5)); ax.set_xticklabels([p.replace('\n', ' ') for p in PHASE], fontsize=6.0, rotation=0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(['%s %s' % (reg, bl) for reg, bk, bl in rows], fontsize=6.0)
        for yb in (3.5, 7.5):
            ax.axhline(yb, color=F.INK, lw=0.5)
        ax.tick_params(length=0, pad=1.5)
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_linewidth(0.5)
        ax.set_title(ttl, fontsize=7.5, pad=2.5)
        G_.label(3, 2 * c, let)
    x, y, w, h = G_.cell_cm(3, 2, cs=2)
    cbar(fig, [(x + w - 1.10) / G_.W, (y + 1.05 + 0.3) / G_.H, 0.16 / G_.W, (h - 1.05 - 0.50 - 0.6) / G_.H],
         -VM2, VM2, 'Difference in mean power (z)', [-0.3, 0, 0.3])
    G_.label(0, 0, 'A'); G_.label(0, 3, 'B'); G_.label(1, 3, 'C'); G_.label(2, 3, 'D')
    return F.save(fig, 'figS08', PNG, PDF)


# ============================================================================ fig. S9
def figS09():
    E = mat('nat2025_event_spectrograms.mat')['Differential_spectrogram']
    C = load('event_clusters.csv'); P = load('pair_clusters.csv'); Q = load('event_clusters_pre.csv')
    # 2026-10-02: 스펙트로그램 축 높이 SPEC_H 로 fig. S8 과 같게(C 행 포함), D 행 높이는 이전과 같음(3.94 cm).
    hr = [SPEC_H + 0.60, SPEC_H + 0.60, SPEC_H + 1.47, 3.941]
    fig = F.figure(F.W3, (sum(hr) + 0.60 + 3 * 0.40) * CM)
    G_ = F.Grid(fig, 4, 3, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.30, hgap=0.40,
                hratios=hr)
    NORM = TwoSlopeNorm(vcenter=0.0, vmin=-0.2, vmax=0.6)
    t = (np.arange(79) - 39) * 0.1266          # 126.6 ms 간격, 0 = 사건
    # 2026-10-02 red note: 각 행의 시간 0 사건을 명시 (C = 회수 시각: 먹이가 로봇에서 떨어진 첫 프레임, Methods 의 retrieval time)
    rowspec = [('Act', 'W', 'A', 'Time 0 = the mouse’s own entry into the robot zone; trials on which it retrieved'),
               ('Act', 'P', 'B', 'Time 0 = the mouse’s own entry into the robot zone; trials on which it entered without retrieving'),
               ('Get', 'W', 'C', 'Time 0 = the grab: first frame with the snack off the robot, taken by the retrieving mouse')]
    for r, (ev, role, let, ttl) in enumerate(rowspec):
        for c, reg in enumerate(REG):
            ax = G_.ax(r, c, padl=1.25 if c == 0 else 0.35, padb=1.05 if r == 2 else 0.18, padr=1.25 if c == 2 else 0.10, padt=0.42)
            ax.set_label('<sub r%dc%d>' % (r, c))
            Z = np.asarray(getattr(getattr(getattr(E, ev), reg), role))      # 100 freq × 79 time
            ax.imshow(Z, aspect='auto', origin='lower', cmap=DIV, norm=NORM,
                      extent=[t[0] - 0.0633, t[-1] + 0.0633, 1, 100], interpolation='nearest')
            ax.axvline(0, color=F.INK, lw=0.6)
            ax.set_xlim(-5, 5); ax.set_ylim(*SPEC_YLIM); ax.set_yticks(SPEC_YT)
            if c == 0:
                ax.set_ylabel('Frequency (Hz)', labelpad=1.0)
            else:
                ax.set_yticklabels([])
            if r == 2:
                ax.set_xticks([-5, -2.5, 0, 2.5, 5]); ax.set_xticklabels(['−5', '−2.5', '0', '2.5', '5'])
                ax.set_xlabel('Time from the grab (s)', labelpad=1.5)
            else:
                ax.set_xticks([])
            if r == 0:
                ax.set_title(reg, fontsize=7.5, pad=2.5)
            ax.tick_params(**TIGHTY)
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_linewidth(0.5)
        G_.label(r, 0, let)
        xx, yy, ww, hh = G_.cell_cm(r, 1)
        fig.text((xx + ww / 2) / G_.W, (yy + hh + 0.06) / G_.H, ttl, ha='center', va='bottom', fontsize=7.0, color=F.INK)
    x, y, w, h = G_.cell_cm(0, 2, rs=3)
    cbar(fig, [(x + w - 1.10) / G_.W, (y + 1.05 + 1.5) / G_.H, 0.16 / G_.W, (h - 1.05 - 0.42 - 3.0) / G_.H],
         -0.2, 0.6, 'Power (within-trial z)', [-0.2, 0, 0.3, 0.6], norm=NORM)
    # ---- D 군집 검정 요약: 행 = 정렬·집단, 열 = 18 영역×대역; 색 = 부호 있는 군집 질량 (가족별 P < 0.05)
    bands = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
    blab = ['θ 4–8', 'θ 8–12', 'β 18–24', 'β 24–32', 'γ 35–50', 'γ 70–90']
    cols = [(reg, b) for reg in REG for b in bands]
    spec = [('C', 'own_entry', 'All mice', 'Own entry · all mice'),
            ('C', 'own_entry', 'Others', 'Own entry · others'),
            ('Q', 'retrieval_pre', 'All mice', 'Grab, −3 to +0.5 s · all'),
            ('C', 'retrieval', 'All mice', 'Grab, −2 to +3 s · all'),
            ('C', 'cagemate_all', 'All mice', 'Cagemate entry · all observers'),
            ('C', 'cagemate_all', 'Others', 'Cagemate entry · others'),
            ('C', 'cagemate_all', 'Provider', 'Cagemate entry · provider'),
            ('C', 'cagemate_nofollow', 'Others', 'Cagemate entry · stayed out'),
            ('P', 'follower_first', 'All mice', 'Cagemate entry · first follower'),
            ('P', 'nonfollower', 'All mice', 'Cagemate entry · never followed')]
    M = np.full((len(spec), len(cols)), np.nan); ann = {}
    ns = []
    for i, (src, arr, row, lab) in enumerate(spec):
        T = {'C': C, 'P': P, 'Q': Q}[src]; T = T[(T.array == arr) & (T.row == row)]
        ns.append(int(T.n.iloc[0]) if len(T) else 0)
        for j, (reg, b) in enumerate(cols):
            s = T[(T.region == reg) & (T.band == b)]
            sig = s[s.P_family < 0.05]
            if len(sig):
                k = sig.mass.idxmax(); r_ = sig.loc[k]
                M[i, j] = (1 if str(r_.sign).strip() in ('+', '＋') else -1) * r_.mass
                ann[(i, j)] = r_.P_family
            else:
                M[i, j] = 0.0
    ax = G_.ax(3, 0, cs=3, padl=4.40, padb=1.35, padr=1.25, padt=0.30)
    vm = np.nanmax(np.abs(M)); vm = max(vm, 1.0)
    ax.imshow(M, aspect='auto', cmap=DIV, vmin=-vm, vmax=vm, interpolation='nearest')
    for (i, j), p in ann.items():
        ax.text(j, i, '%.3f' % p if p >= 0.001 else '<.001', ha='center', va='center', fontsize=5.2, color=F.INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(blab * 3, rotation=90, fontsize=6.0)
    ax.set_yticks(range(len(spec))); ax.set_yticklabels(['%s (n = %d)' % (lab, n) for (_, _, _, lab), n in zip(spec, ns)], fontsize=6.2)
    for xb in (5.5, 11.5):
        ax.axvline(xb, color=F.INK, lw=0.6)
    for yb in (3.5, 7.5):
        ax.axhline(yb, color=F.INK, lw=0.5)
    for k, reg in enumerate(REG):
        ax.text(2.5 + 6 * k, -0.9, reg, ha='center', va='bottom', fontsize=7.0, color=F.INK)
    ax.tick_params(length=0, pad=1.5)
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.5)
    x, y, w, h = G_.cell_cm(3, 2)
    cbar(fig, [(x + w - 1.10) / G_.W, (y + 1.35) / G_.H, 0.16 / G_.W, (h - 1.35 - 0.30) / G_.H],
         -vm, vm, 'Signed cluster mass', [-round(vm), 0, round(vm)])
    G_.label(3, 0, 'D')
    return F.save(fig, 'figS09', PNG, PDF)


if __name__ == '__main__':
    print('saved', figS08()); print('saved', figS09())
