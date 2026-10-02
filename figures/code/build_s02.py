# -*- coding: utf-8 -*-
"""fig. S2 v2 (2026-10-01): 옛 A(46마리 단독 대 집단) 는 Fig. 1C 로 이동, 옛 B(18마리 잠복기) 는 fig. S5B 와
중복이라 제외. 사용자 메모(2026-09-30) 반영: 결정 경계 표기를 점선 오른쪽으로 옮기고 '=' 와 산출 방법을 적음,
Lorenz 소격자에 축 라벨, 폐쇄 칸 삽화 확대. 고정 먹이 패널은 '회수' 가 아닌 '먹이 획득(로봇 위 섭식)' 으로 표기."""
import os, sys, warnings
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import build_r12 as R
warnings.filterwarnings('ignore')
CM, load, COH, TIGHTY, PNG, PDF = F.CM, R.load, R.COH, R.TIGHTY, R.PNG, R.PDF
img, girect, strip, dunn = R.img, R.girect, R.strip, R.dunn
import matplotlib.image as mpimg


def imgfit(G_, r, c, cs, name, padl=0.0, padb=1.20, padt=0.62):
    """삽화를 셀 안에 플롯과 같은 높이(위·아래 기준선 일치)로 놓고 폭은 비율대로."""
    im = mpimg.imread(os.path.join(R.ART, name)); ar = im.shape[1] / im.shape[0]
    x, y, w, h = G_.cell_cm(r, c, cs=cs)
    h2 = h - padb - padt; w2 = min(h2 * ar, w - padl - 0.05)
    h2 = w2 / ar
    return [(x + padl) / G_.W, (y + padb) / G_.H, w2 / G_.W, h2 / G_.H]


def figS02():
    # 2026-10-02 red note: G 삽화 확대 — 2행(G·H)만 높임(행 단위 3.89 cm 유지, 2행 ×1.3)
    fig = F.figure(F.W3, 20.2 * CM)
    # 4행 12열. 0행 A 히스토그램(5)·B(3)·C Lorenz 소격자(4) / 1행 D(3)·E(3)·F 삽화(2)+막대(4)
    # 2행 G 삽화(3)+도넛(3)·H(6) / 3행 I(3)·J(3)·K(4)·L(2)
    G_ = F.Grid(fig, 4, 12, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.22, hgap=0.95,
                hratios=[1, 1, 1.3, 1])
    rng = np.random.default_rng(11)
    ccol = lambda role: F.PROV if role == 'provider' else F.OTH
    PB, PT = 1.20, 0.62

    # ---------------------------------------------------- A 회수율 이봉 분포 (결정 경계 표기 수정)
    a = G_.ax(0, 0, cs=5, padl=1.42, padb=PB, padr=0.12, padt=PT)
    RR = load('figS02b_retrieval_rate_per_mouse.csv')
    bins = np.linspace(0, 1, 21)
    a.hist(RR[RR.role == 'other'].retrieval_rate, bins=bins, color=F.OTH, lw=0)
    a.hist(RR[RR.role == 'provider'].retrieval_rate, bins=bins, color=F.PROV, lw=0)
    a.axvline(0.43, color=F.INK, ls='--', lw=0.6)
    F.tidy(a, 'Retrieval rate (fraction of trials)', 'Mice (n)')
    a.xaxis.labelpad = 1.5; a.yaxis.labelpad = 1.0; a.tick_params(**TIGHTY)
    F.note(a, f'{len(RR)} mice\ndecision boundary = 0.43\n(k-means, k = 2; midpoint\nof centroids 0.04 and 0.82)',
           x=0.98, y=0.97, ha='right', va='top', size=6.6)

    # ---------------------------------------------------- B 제공자 대 나머지 몫
    b = G_.ax(0, 5, cs=3, padl=1.42, padb=PB, padr=0.12, padt=PT)
    SH = load('figS02c_share_provider_vs_others.csv')
    for _, r in SH.iterrows():
        b.plot([0, 1], [r.provider_share, r.others_mean_share], color=F.GRID, lw=0.6)
    b.scatter(np.zeros(len(SH)), SH.provider_share, s=10, color=F.PROV, lw=0, zorder=3)
    b.scatter(np.ones(len(SH)), SH.others_mean_share, s=10, color=F.OTH, lw=0, zorder=3)
    b.set_xticks([0, 1]); b.set_xticklabels(['Provider', 'Others\n(mean)'], fontsize=7.0)
    for _t, _c in zip(b.get_xticklabels(), (F.PROV, F.OTH)): _t.set_color(_c)
    b.set_xlim(-0.45, 1.45); b.set_ylim(0, 1.28)
    F.tidy(b, None, 'Fraction of retrievals'); b.yaxis.labelpad = 1.0; b.tick_params(**TIGHTY)
    F.note(b, '8 cohorts (descriptive)', x=0.5, y=0.97, ha='center', va='top', size=7.0)

    # ---------------------------------------------------- C cohort별 Lorenz (축 라벨 추가)
    L = load('fig1c_lorenz_points.csv'); G = load('fig1c_gini_by_cohort.csv')
    _mc = G_.cell_cm(0, 8, cs=4)
    _x0, _y0 = _mc[0] + 1.20, _mc[1] + PB
    _mw = (_mc[2] - 1.20 - 0.12) / 4.0
    _mh = (_mc[3] - PB - PT) / 2.0
    for k, coh in enumerate(COH):
        ax = fig.add_axes([(_x0 + (k % 4) * _mw) / G_.W, (_y0 + (1 - k // 4) * _mh) / G_.H,
                           (_mw - 0.16) / G_.W, (_mh - 0.36) / G_.H])
        s = L[L.cohort == coh]
        ax.plot([0, 1], [0, 1], color=F.GRID, lw=0.5)
        ax.plot(s.cum_frac_mice, s.cum_share_retrievals, color=F.PROV, lw=0.8, marker='o', ms=1.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['0', '1'] if k // 4 == 1 else []); ax.set_yticklabels(['0', '1'] if k % 4 == 0 else [])
        ax.tick_params(labelsize=5.8, length=1.2, pad=0.8)
        ax.text(0.06, 0.92, f'{coh}  {G[G.cohort == coh].gini.iloc[0]:.2f}', transform=ax.transAxes, fontsize=6.6, va='top')
    fig.text((_x0 + 2 * _mw - 0.06) / G_.W, (_y0 - 0.62) / G_.H, 'Cumulative fraction of mice', ha='center', va='center', fontsize=7.0)
    fig.text((_x0 - 0.78) / G_.W, (_y0 + _mh - 0.06) / G_.H, 'Cumulative fraction of retrievals', ha='center', va='center',
             fontsize=7.0, rotation=90)
    fig.text((_x0 + 2 * _mw - 0.06) / G_.W, (_y0 + 2 * _mh - 0.36 + 0.22) / G_.H, 'Cohort and Gini index', ha='center', va='bottom', fontsize=7.0)

    # ---------------------------------------------------- D·E 역할별 섭식
    EP = load('fig1c_eating_per_mouse_published.csv')
    cmap = {'Providing': F.PROV, 'Venturing': F.OTHL, 'Freeriding': F.INK}
    cols = [cmap[r] for r in EP.dominant_role]; x = np.arange(len(EP))
    d = G_.ax(1, 0, cs=3, padl=1.42, padb=PB, padr=0.12, padt=PT)
    e = G_.ax(1, 3, cs=3, padl=1.42, padb=PB, padr=0.12, padt=PT)
    for ax, val, sd, ylab, pv in ((d, EP.eating_duration_mean_s, EP.eating_duration_sd_s, 'Eating time (s)', EP.p_duration.iloc[0]),
                                  (e, EP.latency_to_eat_mean_s, EP.latency_to_eat_sd_s, 'Latency to eat (s)', EP.p_latency.iloc[0])):
        ax.bar(x, val, 0.72, color=cols, lw=0)
        ax.errorbar(x, val, yerr=sd, fmt='none', ecolor=F.INK, elinewidth=0.6, capsize=1.2)
        ax.set_xticks(x); ax.set_xticklabels(EP.mouse, fontsize=7.0, rotation=90)
        ax.set_ylim(0, (val + sd).max() * 1.30)
        F.tidy(ax, 'Mouse', ylab); ax.yaxis.labelpad = 1.0; ax.xaxis.labelpad = 1.5; ax.tick_params(**TIGHTY)
        F.note(ax, f'P = {pv:.2f}', x=0.5, y=0.97, ha='center', va='top', size=7.0)
    RLAB = {'Providing': 'Retrieved', 'Venturing': 'Entered', 'Freeriding': 'Stayed out'}
    _c0 = G_.cell_cm(1, 0, cs=6); _yl = (_c0[1] + _c0[3] - 0.30) / G_.H
    xo = (_c0[0] + 1.42 + 0.9) / G_.W
    for r_, col in cmap.items():
        fig.patches.append(Rectangle((xo, _yl - 0.006), 0.011, 0.013, transform=fig.transFigure, fc=col, ec='none'))
        fig.text(xo + 0.015, _yl, RLAB[r_], fontsize=7.0, va='center', color=F.INK)
        xo += 0.030 + 0.0052 * len(RLAB[r_])

    # ---------------------------------------------------- F 고정 먹이 (로봇 위 섭식)
    f1 = F.panel(fig, girect(G_, 1, 9, 3, 'fixed_snack.png', padt=PT + 0.12, padl=0.35, padr=0.10))
    img(f1, 'fixed_snack.png')
    f1.text(0.576, 1.03, 'Snack fixed', transform=f1.transAxes, ha='center', va='bottom', fontsize=7.0, color=F.INK)
    f2 = G_.ax(1, 6, cs=3, padl=1.42, padb=PB, padr=0.12, padt=PT)
    FX = load('figS02h_fixed_vs_moving.csv')
    for _, r in FX.iterrows():
        f2.plot([0, 1], [r.rate_moving, r.rate_fixed], color=F.GRID, lw=0.6, zorder=1)
        f2.scatter([0, 1], [r.rate_moving, r.rate_fixed], s=9, color=ccol(r.role), lw=0, zorder=3)
    f2.set_xticks([0, 1]); f2.set_xticklabels(['Removable\n(carried\nout)', 'Fixed\n(eaten\non robot)'], fontsize=7.0)
    f2.set_xlim(-0.45, 1.45); f2.set_ylim(-0.04, 1.18)
    F.tidy(f2, None, 'Trials with snack obtained\n(fraction)'); f2.yaxis.labelpad = 1.0; f2.tick_params(**TIGHTY)
    wfx = stats.wilcoxon(FX.rate_fixed, FX.rate_moving)
    F.note(f2, f'{len(FX)} mice, P = {wfx.pvalue:.3f}', x=0.5, y=0.99, ha='center', va='top', size=7.0)

    # ---------------------------------------------------- G 닫힌 칸 (삽화 확대)
    # 2026-10-02 red note: 삽화 크게(3열, 행 높이 거의 전부), 밖으로 가지고 나간 3시행의 개체 표기
    g1 = F.panel(fig, imgfit(G_, 2, 0, 3, 'closed_chamber.png', padl=0.10, padb=0.05, padt=0.40))
    img(g1, 'closed_chamber.png')
    # 2026-10-02 저자 지시: 삽화 지시선 끝에 이름표(원 삽화의 흐린 흰 글자 자리, 좌표 = 원본 화소 990 × 1458)
    for x_, y_, t_, va_ in ((748, 532, 'Transparent\nbox', 'center'),       # 오른쪽 위 지시선
                            (822, 918, 'One-way\ndoor', 'top'),             # 오른쪽 아래 지시선
                            (22, 684, 'Closed\ndoor', 'top')):              # 왼쪽 지시선
        g1.text(x_, y_, t_, ha='left', va=va_, fontsize=6.5, color=F.INK, linespacing=1.0, clip_on=False, zorder=5,
                bbox=dict(boxstyle='square,pad=0.10', fc='white', ec='none', alpha=0.9))
    g2 = F.panel(fig, G_.rect(2, 3, cs=3, padl=0.45, padb=0.55, padr=0.25, padt=1.70))
    CO = load('figS02i_chamber_outcome.csv').set_index('outcome')
    vals = [float(CO.loc['Eat alone', 'n_trials']), float(CO.loc['Bring out', 'n_trials'])]; tot = sum(vals)
    g2.pie(vals, colors=[F.OTHL, F.PROV], startangle=90, counterclock=False, radius=1.0,
           wedgeprops=dict(width=0.42, lw=0.6, edgecolor='white'))
    g2.set_aspect('equal')
    g2.text(0, 0, f'Ate\ninside\n{vals[0] / tot * 100:.0f}%', ha='center', va='center', fontsize=7.0, color=F.INK)
    CR = load('figS02g_carried_out_trials.csv')
    who = ', '.join('%s ×%d' % (m, k) for m, k in CR.groupby('mouse').size().items())
    rl = 'non-providers' if (CR.role == 'other').all() else 'see table'
    g2.annotate(f'Carried out to the group:\n{int(vals[1])} trials ({vals[1] / tot * 100:.0f}%), by\n{who} ({rl})',
                xy=(0.40, 1.02), xytext=(0.0, 1.80),
                fontsize=7.0, color=F.INK, ha='center', va='center', linespacing=1.15,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=F.PROV, shrinkA=1, shrinkB=2))
    F.note(g2, f'{int(tot)} trials', x=0.5, y=-0.02, ha='center', va='top', size=7.0)

    # ---------------------------------------------------- H 상태별 운동 에너지
    h = G_.ax(2, 6, cs=6, padl=1.52, padb=PB, padr=0.12, padt=PT)
    KE = load('figS02h_kinetic_energy.csv')
    order = ['retrieved', 'entered', 'stayed out']
    vals_ke = [KE[KE.state == r].ke_foraging.dropna().values for r in order]
    strip(h, order, vals_ke, [F.PROV, F.OTHL, F.INK2], rng=rng)
    h.set_yscale('log'); h.set_ylim(1e-9, 3e-1)
    h.set_xticks(range(3)); h.set_xticklabels([o.capitalize() for o in order], fontsize=7.0)
    F.tidy(h, None, 'Kinetic energy (a.u.)'); h.yaxis.labelpad = 1.0; h.tick_params(**TIGHTY)
    pairs = [(0, 1), (0, 2), (1, 2)]; adj = dunn(vals_ke, pairs); kw_all = stats.kruskal(*vals_ke)
    F.note(h, 'Kruskal–Wallis P < 0.001' if kw_all.pvalue < 1e-3 else f'Kruskal–Wallis P = {kw_all.pvalue:.3f}',
           x=0.5, y=0.04, ha='center', va='bottom', size=7.0)
    for (i_, j_), pv, yy in zip(pairs, adj, (3e-3, 2.2e-1, 2.6e-2)):
        txt = 'P < 0.001' if pv < 1e-3 else f'P = {pv:.3f}'
        h.plot([i_, i_, j_, j_], [yy * 0.45, yy, yy, yy * 0.45], color=F.INK, lw=0.5)
        h.text((i_ + j_) / 2, yy * 1.25, txt, ha='center', va='bottom', fontsize=7.0, color=F.INK)

    # ---------------------------------------------------- I 세션 내 집중도
    i1 = G_.ax(3, 0, cs=3, padl=1.42, padb=1.55, padr=0.12, padt=PT)
    Q = load('figS03c_hhi_by_quintile.csv'); piv = Q.pivot_table(index='q', columns='cohort', values='HHI')
    i1.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=F.PROV, alpha=0.20, lw=0)
    i1.plot(piv.index, piv.mean(axis=1), color=F.PROV, lw=1.2)
    i1.set_xticks(list(piv.index)); i1.set_ylim(0, 1.05)
    F.tidy(i1, 'Session quintile', 'HHI'); i1.yaxis.labelpad = 1.0; i1.xaxis.labelpad = 1.5; i1.tick_params(**TIGHTY)
    rho, pv = stats.spearmanr(Q.q, Q.HHI)
    F.note(i1, f'ρ = {rho:.2f}, P = {pv:.3f}', x=0.02, y=0.05, ha='left', va='bottom', size=7.0)

    # ---------------------------------------------------- J 연속 회수 구간
    j1 = G_.ax(3, 3, cs=3, padl=1.42, padb=1.55, padr=0.12, padt=PT)
    BT = load('figS03d_retrieval_bouts.csv')
    edges = np.arange(0.5, 21.5, 1.0); ctr = (edges[:-1] + edges[1:]) / 2
    for role_, col in (('provider', F.PROV), ('other', F.OTH)):
        v = BT[BT.role == role_].bout_len.values
        hgt = np.histogram(v, bins=edges)[0] / max(1, len(v))
        j1.fill_between(ctr, hgt, step='mid', color=col, alpha=0.30, lw=0)
        j1.step(ctr, hgt, where='mid', color=col, lw=1.0, label='Providers' if role_ == 'provider' else 'Others')
    j1.set_xlim(0.5, 20.5)
    F.tidy(j1, 'Uninterrupted run (trials)', 'Fraction of runs'); j1.yaxis.labelpad = 1.0; j1.xaxis.labelpad = 1.5; j1.tick_params(**TIGHTY)
    mp = BT[BT.role == 'provider'].bout_len.mean(); mo = BT[BT.role == 'other'].bout_len.mean()
    j1.legend(loc='upper right', fontsize=7.0, borderpad=0.1, labelspacing=0.25, handletextpad=0.4, frameon=False)
    F.note(j1, f'Mean {mp:.1f} versus\n{mo:.1f} trials', x=0.62, y=0.52, ha='center', va='top', size=7.0)

    # ---------------------------------------------------- K 사전 형질
    k1 = G_.ax(3, 6, cs=4, padl=1.42, padb=1.55, padr=0.12, padt=PT)
    TR = load('fig1e_trait_table.csv')
    cols_m = [('tube_rank_pct', 'Tube\nrank'), ('oft_center_pct', 'Open\nfield'), ('epm_open_pct', 'EPM\nopen'),
              ('ymaze_alternation_pct', 'Y-maze\nalt.'), ('rotarod_pct', 'Rotarod')]
    rng5 = np.random.default_rng(5)
    for kk, (col, lab_) in enumerate(cols_m):
        for jj, (role_, colr) in enumerate([('other', F.OTH), ('provider', F.PROV)]):
            v = pd.to_numeric(TR.loc[TR.role == role_, col], errors='coerce').dropna().values
            xc = kk + (-0.17 if jj == 0 else 0.17)
            k1.scatter(np.full(v.size, xc) + rng5.uniform(-0.07, 0.07, v.size), v, s=3.2, color=colr, lw=0)
            if v.size: k1.plot([xc - 0.12, xc + 0.12], [np.median(v)] * 2, color=F.INK, lw=0.9)
    k1.set_xticks(range(len(cols_m))); k1.set_xticklabels([c[1] for c in cols_m], fontsize=7.0)
    k1.set_xlim(-0.5, len(cols_m) - 0.5); k1.set_ylim(-3, 116)
    F.tidy(k1, None, 'Trait score (%)'); k1.yaxis.labelpad = 1.0; k1.tick_params(**TIGHTY)
    k1.scatter([], [], s=6, color=F.OTH, label='Others'); k1.scatter([], [], s=6, color=F.PROV, label='Providers')
    k1.legend(loc='upper center', fontsize=7.0, ncol=2, frameon=False, borderpad=0.1, handletextpad=0.3, columnspacing=1.0)

    # ---------------------------------------------------- L 활동 전후 서열
    l1 = G_.ax(3, 10, cs=2, padl=1.42, padb=1.55, padr=0.12, padt=PT)
    DM = load('figS02n_dominance_before_after.csv')
    for _, r in DM.iterrows():
        l1.plot([0, 1], [r.dominance_before_pct, r.dominance_after_pct], color=F.GRID, lw=0.5, zorder=1)
    l1.scatter(np.zeros(len(DM)), DM.dominance_before_pct, s=7, color=[ccol(r) for r in DM.role], lw=0, zorder=3)
    l1.scatter(np.ones(len(DM)), DM.dominance_after_pct, s=7, color=[ccol(r) for r in DM.role], lw=0, zorder=3)
    l1.set_xticks([0, 1]); l1.set_xticklabels(['Before', 'After'], fontsize=7.0)
    l1.set_xlim(-0.45, 1.45); l1.set_ylim(-4, 118)
    F.tidy(l1, 'Test weeks', 'Dominance score (%)'); l1.xaxis.labelpad = 1.5; l1.yaxis.labelpad = 1.0; l1.tick_params(**TIGHTY)
    wd = stats.wilcoxon(DM.dominance_before_pct, DM.dominance_after_pct)
    F.note(l1, f'{len(DM)} mice\nP = {wd.pvalue:.2f}', x=0.5, y=0.97, ha='center', va='top', size=7.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 5), ('C', 0, 8),
                         ('D', 1, 0), ('E', 1, 3), ('F', 1, 6),
                         ('G', 2, 0), ('H', 2, 6),
                         ('I', 3, 0), ('J', 3, 3), ('K', 3, 6), ('L', 3, 10)):
        G_.label(r_, c_, lab_)
    return F.save(fig, 'figS02', PNG, PDF)


if __name__ == '__main__':
    print('saved', figS02())
