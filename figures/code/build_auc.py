# -*- coding: utf-8 -*-
"""진입 반응을 시행별 AUC(0…2 s 적분) 로 요약했을 때, 조건 의존성과 남들의 물러남과의 관계."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
CM = F.CM
REG = ['PFC', 'NAc', 'BLA']; COL = {'PFC': F.INK, 'NAc': F.GAM, 'BLA': F.PROV}
BANDS = ['ltheta', 'htheta', 'lbeta', 'hbeta', 'lgamma', 'hgamma']
BL = [r'$\theta$ 4–8', r'$\theta$ 8–12', r'$\beta$ 18–24', r'$\beta$ 24–32',
      r'$\gamma$ 35–50', r'$\gamma$ 70–90']
FE = ['%s_%s' % (g, b) for g in REG for b in BANDS]
XS = np.array([gi * 6.8 + bi for gi in range(3) for bi in range(6)], float)
KEY6 = ['PFC_ltheta', 'PFC_lbeta', 'NAc_lbeta', 'BLA_lbeta', 'BLA_hbeta', 'BLA_lgamma']
K6L = ['PFC ' + BL[0], 'PFC ' + BL[2], 'NAc ' + BL[2], 'BLA ' + BL[2], 'BLA ' + BL[3], 'BLA ' + BL[4]]


def feataxis(ax, top=False):
    ax.set_xticks(XS)
    ax.set_xticklabels(BL * 3, rotation=90)
    ax.tick_params(axis='x', pad=1.5, labelsize=6.6)
    ax.set_xlim(-0.9, 2 * 6.8 + 5.9)
    for gi, g in enumerate(REG):
        ax.annotate(g, xy=(gi * 6.8 + 2.5, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -34), textcoords='offset points', ha='center', va='top',
                    fontsize=7.5, color=COL[g])


def fig():
    M = pd.read_csv(os.path.join(D, 'auc_mean.csv')).set_index('feature').loc[FE]
    I = pd.read_csv(os.path.join(D, 'auc_by_inside.csv'))
    B = pd.read_csv(os.path.join(D, 'auc_beta.csv'))
    # 2026-10-02 red notes: A·B 의 주석을 축 위로 올려 선과 겹치지 않게(윗줄 padt 0.50 → 1.05, 그만큼 그림 높이 증가),
    # A 유의 표시 = 이 연구의 진입 군집 검정(fig. S9D, own entry · all mice)에서 family-wise P < 0.05 인 특징.
    fig = F.figure(F.W3, 12.15 * CM)
    G_ = F.Grid(fig, 2, 2, left=0.10, right=0.12, top=0.52, bottom=0.12,
                wgap=1.55, hgap=0.95, hratios=[5.555, 5.005])
    CL = pd.read_csv(os.path.join(D, 'event_clusters.csv'))
    CL = CL[(CL.array == 'own_entry') & (CL.row == 'All mice') & (CL.P_family < 0.05)]
    SIG = set(CL.region + '_' + CL.band)

    a = G_.ax(0, 0, padl=1.42, padb=1.52, padr=0.12, padt=1.05)
    cols = [COL[f.split('_')[0]] for f in FE]
    a.bar(XS, M['mean'].values, 0.74, yerr=M['sem'].values, color=cols, lw=0,
          error_kw=dict(lw=0.6, ecolor=F.INK, capsize=0))
    a.axhline(0, color=F.INK, lw=0.5)
    for x, f, m_, e_ in zip(XS, FE, M['mean'].values, M['sem'].values):
        if f in SIG:
            up = m_ >= 0
            a.text(x, m_ + e_ + 0.012 if up else m_ - e_ - 0.012, '*', ha='center', va='bottom' if up else 'top',
                   fontsize=8.0, color=F.INK, fontweight='bold')
    feataxis(a); a.set_ylim(-0.30, 0.42)
    a.set_ylabel('Entry AUC, 0–2 s (z·s)')
    F.note(a, 'mean ± SEM, 555 entries\n* entry cluster at family-wise P < 0.05 (fig. S9D)',
           x=0.0, y=1.03, ha='left', va='bottom', size=6.5)

    b_ = G_.ax(0, 1, padl=1.42, padb=1.52, padr=0.12, padt=1.05)
    xs = np.arange(3)
    for f, lb in zip(KEY6, K6L):
        s = I[I.feature == f].sort_values('n_before')
        hi = (f == 'PFC_ltheta')
        b_.errorbar(xs, s['mean'], yerr=s['sem'], color=COL[f.split('_')[0]],
                    lw=1.3 if hi else 0.7, marker='o', ms=3.4 if hi else 2.4,
                    alpha=1.0 if hi else 0.45, capsize=0, zorder=3 if hi else 2)
    b_.axhline(0, color=F.GRID, lw=0.5)
    b_.set_xticks(xs); b_.set_xticklabels(['0', '1', '2 or more'])
    b_.set_xlim(-0.35, 2.35); b_.set_ylim(-0.30, 0.42)
    b_.set_xlabel('Cagemates already inside at this entry')
    b_.set_ylabel('Entry AUC, 0–2 s (z·s)')
    n = I[I.feature == 'PFC_ltheta'].sort_values('n_before').n.values
    for x, v in zip(xs, n):
        b_.text(x, -0.29, 'n = %d' % v, ha='center', va='bottom', fontsize=6.2, color=F.OTH)
    bb = B[(B.panel == 'n_inside') & (B.feature_id == 'PFC_ltheta')].iloc[0]
    # 2026-10-02 저자 결정: B 의 별표 삭제 (P 0.007 은 보정 전, family-wise 0.09)
    F.note(b_, 'PFC $\\theta$ 4–8 (solid) deepens with the number inside:\n'
           '$\\beta$ = %+.3f/SD, P = %.3f (uncorrected), family-wise P = %.2f\n'
           'faded, the five other entry features'
           % (bb.beta, bb.P, bb.P_family),
           x=0.0, y=1.03, ha='left', va='bottom', size=6.5)
    for t_ in b_.texts[-1:]:
        t_.set_text(t_.get_text().replace('= -', '= \u2212'))     # 음수 부호는 minus sign

    for j, (pan, ttl) in enumerate((('next_wait', 'How long the next mouse then waits'),
                                    ('never_enter', 'How many cagemates never enter'))):
        ax = G_.ax(1, j, padl=1.42, padb=1.52, padr=0.12, padt=0.50)
        s = B[B.panel == pan].set_index('feature_id').loc[FE]
        bd = float(s.detect_bound.iloc[0])
        ax.axhspan(-bd, bd, color=F.GRID, alpha=0.45, lw=0, zorder=0)
        ax.axhline(0, color=F.INK, lw=0.5, zorder=1)
        for x, f, v in zip(XS, FE, s.beta.values):
            ax.plot([x, x], [0, v], color=COL[f.split('_')[0]], lw=0.8, zorder=2)
        ax.scatter(XS, s.beta.values, s=13, color=cols, lw=0, zorder=3)
        feataxis(ax); ax.set_ylim(-0.19, 0.19)
        ax.set_yticks([-0.15, 0, 0.15])
        ax.set_ylabel('Effect of the AUC on\n%s (per SD)' % ('the wait' if j == 0 else 'the count'))
        F.note(ax, '%s\nn = %d; nothing outside the band' % (ttl, int(s.n.iloc[0])),
               x=0.02, y=0.97, ha='left', va='top', size=6.5)
        if j == 0:
            F.note(ax, 'grey band: effects too small\nto detect (family-wise 5%)',
                   x=0.98, y=0.03, ha='right', va='bottom', size=6.2, color=F.OTH)
    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 1), ('C', 1, 0), ('D', 1, 1)):
        G_.label(r_, c_, lab_)
    return F.save(fig, 'EvI_entry_auc', PNG, PDF)


if __name__ == '__main__':
    print(fig())
