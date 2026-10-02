# -*- coding: utf-8 -*-
"""보충 그림 세 장: (1) DREADD 조작 검증 세 영역 전체, (2) 해독기 변형 AUC, (3) θ tilt 재현(코호트 A 대 Day 21–45)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
CM = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
DB = ['theta1', 'theta2', 'beta1', 'beta2', 'gamma1', 'gamma2']
BL = [r'$\theta$ 4–8', r'$\theta$ 8–12', r'$\beta$ 18–24', r'$\beta$ 24–32', r'$\gamma$ 35–50', r'$\gamma$ 70–90']


# ------------------------------------------------ (1) DREADD 조작 검증, 세 영역
def figS_dreadd_manip():
    DP = load('fig5g_dreadd_power.csv')
    fig = F.figure(F.W3, 6.4 * CM)
    G_ = F.Grid(fig, 1, 3, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.35, hgap=0.8)
    xs = np.arange(6)
    for k, reg in enumerate(('PFC', 'NAc', 'BLA')):
        ax = G_.ax(0, k, padl=1.60 if k == 0 else 0.55, padb=1.30, padr=0.12, padt=0.62)
        dp = DP[DP.region == reg]
        for j, (seg, col) in enumerate((('Baseline', F.OTH), ('Foraging', F.PROV))):
            s = dp[dp.segment == seg].set_index('band')
            for i, b in enumerate(DB):
                v = float(s.loc[b, 'effect_log10']); p = float(s.loc[b, 'P'])
                x = xs[i] + (j - 0.5) * 0.38
                ax.bar(x, v, 0.34, color=col, lw=0, label=seg if (i == 0 and k == 0) else None)
                if p < 0.05:
                    ax.text(x, v + (0.010 if v > 0 else -0.010), '*', ha='center',
                            va='bottom' if v > 0 else 'top', fontsize=8, color=col)
        ax.axhline(0, color=F.INK, lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(BL, rotation=90); ax.tick_params(axis='x', pad=1.5, labelsize=6.6)
        ax.set_xlim(-0.6, 5.6); ax.set_ylim(-0.23, 0.26); ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        if k > 0: ax.set_yticklabels([])
        else: ax.set_ylabel('CNO − saline\n(log$_{10}$ power)')
        coh = dp.cohorts.iloc[0]; nseg = int(dp.n_segments.iloc[0])
        ax.set_title('%s  (%s, %d + %d segments)' % (reg, ', '.join(coh), nseg, int(dp[dp.segment == 'Foraging'].n_segments.iloc[0])),
                     pad=2, fontsize=7.5)
        if k == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1.03), ncol=1, handlelength=0.9, borderpad=0.1, labelspacing=0.2)
            F.note(ax, '* P < 0.05', x=0.02, y=0.03, ha='left', va='bottom', size=6.5)
        G_.label(0, k, 'ABC'[k])
    return F.save(fig, 'figS_dreadd_manipulation_all', PNG, PDF)


# ------------------------------------------------ (2) 해독기 변형 AUC
def figS_decoder():
    V = load('decoder_variants.csv')
    order = ['theta only', 'PFC only', 'NAc only', 'all 18 features', 'PFC high β alone (no fitting)',
             'BLA only', 'beta only', 'gamma only', '5 principal components', '3 principal components',
             '2 principal components', '1 principal component']
    lab = {'theta only': r'$\theta$ only (6)', 'PFC only': 'PFC only (6)', 'NAc only': 'NAc only (6)',
           'all 18 features': 'all 18 features', 'PFC high β alone (no fitting)': r'PFC high $\beta$ alone, no fitting',
           'BLA only': 'BLA only (6)', 'beta only': r'$\beta$ only (6)', 'gamma only': r'$\gamma$ only (6)',
           '5 principal components': '5 principal components', '3 principal components': '3 principal components',
           '2 principal components': '2 principal components', '1 principal component': '1 principal component'}
    fig = F.figure(F.W2, 7.4 * CM)
    G_ = F.Grid(fig, 1, 1, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.2, hgap=0.8)
    ax = G_.ax(0, 0, padl=4.3, padb=1.05, padr=0.25, padt=0.62)
    ys = np.arange(len(order))[::-1]
    ax.axvline(0.5, color=F.INK, lw=0.5)
    for y, v in zip(ys, order):
        for samp, mk, fc in (('post only (n=616)', 'o', F.BET), ('post+pre required (n=327)', 'o', '#ffffff')):
            r = V[(V['variant'] == v) & (V['sample'] == samp)]
            if r.empty: continue
            r = r.iloc[0]
            ax.scatter([r.auc], [y], s=18, marker=mk, facecolor=fc, edgecolor=F.BET, lw=0.8, zorder=3)
            if samp.startswith('post only') and pd.notna(r.P_family) and r.P_family < 0.05:
                ax.text(r.auc + 0.006, y, '‡' if r.P_family < 0.01 else '†', ha='left', va='center', fontsize=7, color=F.BET)
    ax.set_yticks(ys); ax.set_yticklabels([lab[v] for v in order]); ax.tick_params(axis='y', length=0, pad=1.5, labelsize=6.8)
    ax.set_xlim(0.47, 0.705); ax.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7]); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel('Held-out AUC (leave one mouse out)')
    ax.spines['left'].set_visible(False)
    F.note(ax, 'filled: n = 616, post-entry\nopen: n = 327, both windows\n'
           '† P < 0.05, ‡ P < 0.01\n(family-wise, 11 variants)', x=0.98, y=0.03, ha='right', va='bottom', size=6.3)
    return F.save(fig, 'figS_decoder_variants', PNG, PDF)


# ------------------------------------------------ (3) θ tilt 재현: 코호트 A 대 Day 21–45
def figS_replication():
    A = load('theta_tilt_tests.csv'); Mc = load('tilt_member_change_tests.csv')
    PA = load('fig5e_tilt_per_mouse.csv'); PM = load('tilt_member_change_per_mouse.csv')
    rows = [('Cohort A, both windows (n = 327)', float(A[(A['sample'] == 'n=327') & (A.model == 'θ tilt alone')].beta.iloc[0]),
             float(A[(A['sample'] == 'n=327') & (A.model == 'θ tilt alone')].P.iloc[0]), 'A'),
            ('Cohort A, post-entry window (n = 616)', float(A[(A['sample'] == 'n=616') & (A.model == 'θ tilt alone')].beta.iloc[0]),
             float(A[(A['sample'] == 'n=616') & (A.model == 'θ tilt alone')].P.iloc[0]), 'A'),
            ('Cohort A, movement controlled (n = 616)', float(A[(A['sample'] == 'n=616') & (A.model == 'θ tilt + baseline kinetic energy')].beta.iloc[0]),
             float(A[(A['sample'] == 'n=616') & (A.model == 'θ tilt + baseline kinetic energy')].P.iloc[0]), 'A'),
            ('Days 21–45, all five mice (n = 761)', float(Mc[Mc.analysis.str.startswith('primary')].beta.iloc[0]),
             float(Mc[Mc.analysis.str.startswith('primary')].P.iloc[0]), 'M'),
            ('Days 21–45, A5 excluded (n = 720)', float(Mc[Mc.analysis.str.startswith('exclude A5 (')].beta.iloc[0]),
             float(Mc[Mc.analysis.str.startswith('exclude A5 (')].P.iloc[0]), 'M'),
            ('Days 21–45, PFC component only (n = 720)', float(Mc[Mc.analysis.str.startswith('PFC component')].beta.iloc[0]),
             float(Mc[Mc.analysis.str.startswith('PFC component')].P.iloc[0]), 'M')]
    fig = F.figure(F.W3, 6.4 * CM)
    G_ = F.Grid(fig, 1, 12, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.22, hgap=0.8)
    a = G_.ax(0, 0, cs=7, padl=4.6, padb=1.05, padr=1.55, padt=0.62)
    ys = np.arange(len(rows))[::-1]
    a.axvline(0, color=F.INK, lw=0.5)
    for y, (lab, b, p, kind) in zip(ys, rows):
        col = F.BET
        a.plot([0, b], [y, y], color=col, lw=0.8, zorder=2)
        a.scatter([b], [y], s=16, facecolor=col if kind == 'A' else '#ffffff', edgecolor=col, lw=0.8, zorder=3)
        a.text(0.108, y, ('P < 0.001' if p < 0.001 else 'P = %.2g' % p), ha='left', va='center', fontsize=6.4)
    a.set_yticks(ys); a.set_yticklabels([r[0] for r in rows]); a.tick_params(axis='y', length=0, pad=1.5, labelsize=6.6)
    a.set_xlim(-0.012, 0.104); a.set_xticks([0, 0.05, 0.10]); a.set_ylim(-0.7, len(rows) - 0.3)
    a.set_xlabel(r'Change in joining per SD of $\theta$ tilt'); a.spines['left'].set_visible(False)
    F.note(a, 'filled, cohort A; open, Days 21–45', x=0.98, y=0.97, ha='right', va='top', size=6.3)
    b_ = G_.ax(0, 7, cs=5, padl=1.55, padb=1.05, padr=0.12, padt=0.62)
    pa = PA[PA['sample'] == 'n616'].sort_values('mouse'); pm = PM.sort_values('mouse')
    xs = np.arange(len(pa))
    b_.axhline(0, color=F.INK, lw=0.5)
    b_.scatter(xs - 0.15, pa.beta_tilt, s=16, color=F.BET, lw=0, zorder=3, label='cohort A (n = 616)')
    b_.scatter(xs + 0.15, pm.beta, s=16, facecolor='#ffffff', edgecolor=F.BET, lw=0.8, zorder=3, label='Days 21–45')
    b_.set_xticks(xs); b_.set_xticklabels(pa.mouse); b_.set_xlim(-0.6, len(pa) - 0.4)
    b_.set_ylim(-0.08, 0.21); b_.set_yticks([0, 0.1, 0.2])
    b_.set_ylabel(r'Per-mouse slope' + '\n' + r'(joining per SD of $\theta$ tilt)')
    b_.legend(loc='upper right', bbox_to_anchor=(1.02, 1.04), handlelength=0.9, borderpad=0.1, labelspacing=0.2)
    F.note(b_, 'A5 channel artefact in Days 21–45', x=0.02, y=0.03, ha='left', va='bottom', size=6.2, color=F.OTH)
    G_.label(0, 0, 'A'); G_.label(0, 7, 'B')
    return F.save(fig, 'figS_tilt_replication', PNG, PDF)


if __name__ == '__main__':
    for f in (figS_dreadd_manip, figS_decoder, figS_replication):
        print(f())
