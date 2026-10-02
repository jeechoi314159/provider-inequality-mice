# -*- coding: utf-8 -*-
"""Fig. 4 (LFP) 생성. data/ 의 파일만 읽는다."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

HERE = os.path.dirname(os.path.abspath(__file__))
SUB  = os.path.dirname(HERE)
D    = os.path.join(SUB, 'data')
PNG  = os.path.join(SUB, 'png')
PDF  = os.path.join(SUB, 'pdf')
CM   = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
TIGHTY = dict(axis='y', labelsize=7.0, length=2.0, pad=1.0)
TAGS = {'fig4': ''}
RCOL = {'Retrieved': F.PROV, 'Entered': F.OTHL, 'Stayed out': F.INK2}


def fig4():
    fig = F.figure(F.W3, 16.5 * CM)
    # 공통 격자 3행 12열. 1행 A·B·C(각 4열), 2행 D(4열)·E(8열), 3행 F·G·H(각 4열)
    G_ = F.Grid(fig, 3, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[1.10, 0.95, 1.10])

    # ------------------------------------------- A 역할별 high β 궤적 (PFC·NAc·BLA)
    TR = load('fig4a_band_traces.csv')
    TR = TR[TR.band == 'hbeta']
    NPT = TR.x.max() // 2 + 1
    _ax0, _ay0, _aw, _ah = G_.cell_cm(0, 0, cs=4)
    _pl, _pb, _pr, _pt, _gp = 1.38, 1.25, 0.12, 0.90, 0.22
    _sh = (_ah - _pb - _pt - 2 * _gp) / 3.0
    for gi, g in enumerate(('PFC', 'NAc', 'BLA')):
        ax = fig.add_axes([(_ax0 + _pl) / G_.W,
                           (_ay0 + _pb + (2 - gi) * (_sh + _gp)) / G_.H,
                           (_aw - _pl - _pr) / G_.W, _sh / G_.H], label='<sub%d>' % gi)
        for role in ('Stayed out', 'Entered', 'Retrieved'):
            s = TR[(TR.region == g) & (TR.role == role)].sort_values('x')
            ax.plot(s.x, s['mean'], color=RCOL[role], lw=0.9, zorder=3)
            ax.fill_between(s.x, s['mean'] - s.se, s['mean'] + s.se, color=RCOL[role],
                            alpha=0.18, lw=0, zorder=2)
        ax.axvline(NPT - 0.5, color=F.INK, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(0, 2 * NPT - 1)
        ax.set_xticks([NPT / 2, NPT * 1.5])
        if gi == 2:
            ax.set_xticklabels(['Approach', 'Retrieval'], fontsize=7.0)
            ax.tick_params(axis='x', length=0, pad=1.5)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)
        F.tidy(ax, None, 'High-β power (z)' if gi == 1 else None)
        ax.yaxis.labelpad = 1.0; ax.tick_params(**TIGHTY)
        F.note(ax, g, x=0.985, y=0.95, ha='right', va='top', size=7.0)
        if gi == 2:
            ax.set_xlabel('Normalised time within epoch', fontsize=7.0)
            ax.xaxis.labelpad = 1.5
        if gi == 0:
            for role in ('Retrieved', 'Entered', 'Stayed out'):
                ax.plot([], [], color=RCOL[role], lw=1.0, label=role)
            ax.legend(loc='upper left', fontsize=7.0, frameon=False, borderpad=0.1,
                      labelspacing=0.18, handlelength=1.0, handletextpad=0.35, ncol=2,
                      columnspacing=0.7, bbox_to_anchor=(-0.02, 1.42))

    # ------------------------------------------- B 회수 대 진입을 가르는 특징
    b = G_.ax(0, 4, cs=4, padl=1.92, padb=1.25, padr=0.12, padt=0.90)
    L = load('fig4b_lda_coefficients.csv').head(10)
    yy = np.arange(len(L))[::-1]
    cols = [F.BET if f == 'PFC_hbeta' else F.OTH for f in L.feature]
    b.barh(yy, L['abs'], color=cols, height=0.62, lw=0)
    lab = [f.replace('_ltheta', ' θ low').replace('_htheta', ' θ high')
            .replace('_lbeta', ' β low').replace('_hbeta', ' β high')
            .replace('_lgamma', ' γ low').replace('_hgamma', ' γ high') for f in L.feature]
    b.set_yticks(yy); b.set_yticklabels(lab, fontsize=7.0)
    b.set_ylim(-0.7, len(L) - 0.3)
    F.tidy(b, '|LDA coefficient|', None)
    b.xaxis.labelpad = 1.5; b.tick_params(**TIGHTY)
    F.note(b, '{} retrieved vs {} entered trials'.format(int(L.n_retrieved.iloc[0]),
                                                         int(L.n_entered.iloc[0])),
           x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    # ------------------------------------------- C 비용–편익 가중치 지도
    c = G_.ax(0, 8, cs=4, padl=1.08, padb=1.25, padr=0.12, padt=0.90)
    W = load('fig4c_enet_weights.csv')
    BORD = ['low_theta', 'high_theta', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']
    M = W.pivot_table(index='region', columns='band', values='weight').reindex(
        index=['PFC', 'NAc', 'BLA'], columns=BORD)
    vmax = np.nanmax(np.abs(M.values))
    from matplotlib.colors import LinearSegmentedColormap
    CMAP = LinearSegmentedColormap.from_list('bc', [F.BET, '#ffffff', F.PROV])
    c.imshow(M.values, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            c.text(j, i, '{:.1f}'.format(v).replace('-', '−'),
                   ha='center', va='center', fontsize=7.0,
                   color='white' if abs(v) > 0.62 * vmax else F.INK)
    c.set_xticks(range(6))
    c.set_xticklabels(['θ\nlow', 'θ\nhigh', 'β\nlow', 'β\nhigh', 'γ\nlow', 'γ\nhigh'],
                      fontsize=7.0, linespacing=1.1)
    c.set_yticks(range(3)); c.set_yticklabels(['PFC', 'NAc', 'BLA'], fontsize=7.0)
    c.tick_params(length=1.4, pad=1.0)
    for sp in c.spines.values():
        sp.set_visible(False)
    rho = load('fig4c_loco_stability.csv').rho.median()
    F.note(c, 'Held-out cohort ρ = {:.2f}'.format(rho),
           x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    # ------------------------------------------- D 편익–비용 평면과 표본 밖 AUC
    d = fig.add_axes(G_.rect(1, 0, cs=5, padl=1.38, padb=1.30, padr=3.49, padt=0.55))
    BC = load('fig4d_benefit_cost.csv')
    lim = max(abs(BC.benefit).max(), abs(BC.cost).max()) * 1.18
    d.plot([-lim, lim], [-lim, lim], color=F.GRID, lw=0.6)
    for r_, col, lab_ in (('P', F.PROV, 'Provider'), ('NP', F.OTH, 'Others')):
        s = BC[BC.Role == r_]
        d.scatter(s.benefit, s.cost, s=9, color=col, lw=0, label=lab_)
    d.set_xlim(-lim, lim); d.set_ylim(-lim, lim)
    F.tidy(d, 'Benefit', 'Cost')
    d.yaxis.labelpad = 1.0; d.xaxis.labelpad = 1.5; d.tick_params(**TIGHTY)
    d.legend(loc='upper left', fontsize=7.0, frameon=False, borderpad=0.1, labelspacing=0.2,
             handletextpad=0.3)
    F.note(d, '18 mice; P = 4 $\\times$ 10$^{-4}$', x=0.0, y=1.02,
           ha='left', va='bottom', size=7.0)
    d2 = fig.add_axes(G_.rect(1, 0, cs=5, padl=4.53, padb=1.30, padr=1.49, padt=0.55), label='<subD2>')
    AU = load('fig4d_heldout_auc.csv')
    vals = list(AU.auc); pvs = list(AU.perm_P)
    d2.bar(range(len(vals)), vals, color=F.OTH, width=0.62, lw=0)
    for k, (v, pv) in enumerate(zip(vals, pvs)):
        d2.text(k, v + 0.025, '{:.2f}'.format(v), ha='center', va='bottom',
                fontsize=7.0, color=F.INK)
    d2.axhline(0.5, color=F.INK, lw=0.6, ls=(0, (3, 2)))
    d2.set_xticks(range(len(vals)))
    d2.set_xticklabels([str(x).capitalize() for x in AU.left_out], fontsize=7.0)
    d2.set_ylim(0, 1.0); d2.set_yticks([0, 0.5, 1.0])
    d2.set_yticklabels(['0', '.5', '1'], fontsize=7.0)
    F.tidy(d2, None, 'Identity AUC, held out')
    d2.yaxis.label.set_size(7.0)
    d2.yaxis.set_label_position('right'); d2.yaxis.tick_right()
    d2.yaxis.labelpad = 1.0; d2.tick_params(**TIGHTY); d2.tick_params(axis='x', length=1.4, pad=1.0)


    # ------------------------------------------- E 동료 진입 정렬 t 곡선
    TC = load('fig4e_timecourses.csv')
    CL = load('fig4e_clusters.csv')
    for pi, (key, col, panel) in enumerate((('PFC γ (51–100 Hz)', F.GAM, 'gamma'),
                                            ('PFC high β (24–32 Hz)', F.BET, 'beta'))):
        _eo = (1.38, 6.425)[pi]
        ax = fig.add_axes(G_.rect(1, 5, cs=7, padl=_eo, padb=1.30,
                                  padr=10.50 - _eo - 3.945, padt=0.55), label='<subE%d>' % pi)
        cl = CL[CL.panel == panel].iloc[0]
        ax.add_patch(Rectangle((cl.t_start, -3.6), cl.t_end - cl.t_start, 7.2,
                               fc='#f2f1ee', ec='none', zorder=0))
        for ana, ls, lab_ in (('wait', '-', 'Waits longer'), ('follow', (0, (3, 2)), 'Follows')):
            s = TC[(TC['key'] == key) & (TC.analysis == ana)].sort_values('t')
            yv = s.tstat.rolling(5, center=True, min_periods=1).mean()
            ax.plot(s.t, yv, color=col, lw=1.0, ls=ls, label=lab_, zorder=3)
        ax.axhline(0, color=F.INK, lw=0.5)
        ax.axvline(0, color=F.INK, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(-6, 3); ax.set_ylim(-3.6, 3.6)
        F.tidy(ax, 'Time from the cagemate’s entry (s)',
               'Within-mouse t' if pi == 0 else None)
        ax.yaxis.labelpad = 1.0; ax.xaxis.labelpad = 1.5; ax.tick_params(**TIGHTY)
        if pi == 1:
            ax.set_yticklabels([])
        ax.legend(loc='lower left', fontsize=7.0, frameon=False, borderpad=0.1,
                  labelspacing=0.2, handlelength=1.2, handletextpad=0.35)
        F.note(ax, key, x=0.5, y=1.02, ha='center', va='bottom', size=7.0, color=col)
        F.note(ax, 'cluster P = {:.3f}'.format(cl.P_family), x=0.98, y=0.98,
               ha='right', va='top', size=7.0)

    # ------------------------------------------- F 결합 모형의 두 부분 효과
    f = G_.ax(2, 0, cs=4, padl=1.95, padb=1.30, padr=0.35, padt=1.15)
    JM = load('fig4f_joint_model.csv')
    piv = JM.pivot_table(index='outcome', columns='signal', values='beta').reindex(
        index=['Waits longer', 'Follows'], columns=['γ before entry', 'β after entry'])
    pv = JM.pivot_table(index='outcome', columns='signal', values='P').reindex(
        index=piv.index, columns=piv.columns)
    from matplotlib.colors import LinearSegmentedColormap
    CM2 = LinearSegmentedColormap.from_list('eff', ['#ffffff', '#cfcdc7'])
    vm = np.nanmax(np.abs(piv.values))
    f.imshow(np.abs(piv.values), cmap=CM2, vmin=0, vmax=vm, aspect='auto')
    for i in range(2):
        for j in range(2):
            strong = pv.values[i, j] < 0.05
            f.text(j, i - 0.10, '{:+.3f}'.format(piv.values[i, j]).replace('-', '−'),
                   ha='center', va='center', fontsize=7.0,
                   fontweight='bold' if strong else 'normal',
                   color=(F.GAM if j == 0 else F.BET) if strong else F.INK)
            f.text(j, i + 0.16, 'P = {:.3f}'.format(pv.values[i, j]), ha='center', va='center',
                   fontsize=7.0, fontweight='bold' if strong else 'normal', color=F.INK)
    f.set_xticks([0, 1]); f.set_xticklabels(['γ before', 'β after'], fontsize=7.0)
    f.set_yticks([0, 1]); f.set_yticklabels(['Waits\nlonger', 'Follows'], fontsize=7.0)
    f.tick_params(length=0, pad=2.0)
    for sp in f.spines.values():
        sp.set_visible(False)

    # ------------------------------------------- G 견고성
    g = G_.ax(2, 4, cs=4, padl=2.98, padb=1.30, padr=0.12, padt=1.15)
    RB = load('fig4g_robustness.csv')
    order = list(dict.fromkeys(RB.check))[::-1]
    for _, r in RB.iterrows():
        yv = order.index(r.check) + (0.16 if r.signal.startswith('γ') else -0.16)
        col = F.GAM if r.signal.startswith('γ') else F.BET
        face = col if r.P < 0.05 else 'white'
        g.plot([r.beta], [yv], marker='o', ms=3.4, lw=0, markerfacecolor=face,
               markeredgecolor=col, markeredgewidth=0.7, clip_on=False, zorder=3)
        _lab = 'P = {:.3f}'.format(r.P) if r.P >= 0.001 else 'P < 0.001'
        g.text(r.beta + (0.006 if r.beta >= 0 else -0.006), yv, _lab, fontsize=7.0,
               va='center', ha='left' if r.beta >= 0 else 'right', color=F.INK)
    g.axvline(0, color=F.INK, lw=0.6, ls=(0, (3, 2)))
    g.set_yticks(range(len(order))); g.set_yticklabels(order, fontsize=7.0)
    g.set_ylim(-0.6, len(order) - 0.4); g.set_xlim(-0.325, 0.190)
    F.tidy(g, 'Effect per SD of power', None)
    g.xaxis.labelpad = 1.5; g.tick_params(**TIGHTY)
    hs = [mlines.Line2D([], [], marker='o', ms=3.4, lw=0, markerfacecolor=F.GAM,
                        markeredgecolor=F.GAM, label='γ before entry → waits longer'),
          mlines.Line2D([], [], marker='o', ms=3.4, lw=0, markerfacecolor=F.BET,
                        markeredgecolor=F.BET, label='β after entry → follows'),
          mlines.Line2D([], [], marker='o', ms=3.4, lw=0, markerfacecolor='white',
                        markeredgecolor=F.INK, label='P ≥ 0.05')]
    g.legend(handles=hs, loc='lower left', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.2, handletextpad=0.3,
             bbox_to_anchor=(-2.98 / 2.83, 1.01), ncol=1)

    # ------------------------------------------- H 쥐별 계수
    h = G_.ax(2, 8, cs=4, padl=3.00, padb=1.30, padr=0.12, padt=1.15)
    PM = load('fig4h_per_mouse.csv')
    mice = sorted(PM.mouse.unique())[::-1]
    for _, r in PM.iterrows():
        yv = mice.index(r.mouse)
        if r.outcome == 'Waits longer':
            h.plot([r.beta_gamma], [yv + 0.16], marker='o', ms=3.2, lw=0, color=F.GAM,
                   markeredgecolor='none', zorder=3)
        else:
            h.plot([r.beta_beta], [yv - 0.16], marker='o', ms=3.2, lw=0, color=F.BET,
                   markeredgecolor='none', zorder=3)
    h.axvline(0, color=F.INK, lw=0.6, ls=(0, (3, 2)))
    h.set_yticks(range(len(mice))); h.set_yticklabels(mice, fontsize=7.0)
    h.set_ylim(-0.6, len(mice) - 0.4)
    F.tidy(h, 'Effect per SD', 'Mouse (cohort A)')
    h.yaxis.labelpad = 1.0; h.xaxis.labelpad = 1.5; h.tick_params(**TIGHTY)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 4), ('C', 0, 8),
                         ('D', 1, 0), ('E', 1, 5),
                         ('F', 2, 0), ('G', 2, 4), ('H', 2, 8)):
        G_.label(r_, c_, TAGS['fig4'] + lab_)
    return F.save(fig, 'figS13', PNG, PDF)


if __name__ == '__main__':
    print('saved', fig4())
