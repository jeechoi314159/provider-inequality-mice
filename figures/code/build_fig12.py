# -*- coding: utf-8 -*-
"""Fig. 1 (노동·보상·단독) 과 Fig. 2 (초기 역할·형질) — 2026-10-01 재구성.
옛 Fig. 1 A–H 를 두 그림으로 나누고, 옛 fig. S2A(46마리 단독 대 집단)를 Fig. 1C 로,
비순환 검증(10시행 선두 → 11시행 이후 회수 비율)을 Fig. 2C 로 신설. 사용자 메모(2026-09-30) 반영:
A 의 '(Mouse only)' 라벨을 삽화 밖으로, raster 의 'No retrieval' 범례 삭제, 패널 문자를 플롯 옆으로,
HHI 패널 y축을 위 행 패널의 y축과 정렬, 'P = 0.016' 앞에 '8 cohorts'."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Patch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import build_r12 as R
warnings.filterwarnings('ignore')

CM, load, COH, TIGHTY, ART, PNG, PDF = F.CM, R.load, R.COH, R.TIGHTY, R.ART, R.PNG, R.PDF
PADL = 1.32          # 같은 열의 y축 기준선을 맞추기 위한 공통 왼쪽 패딩


# ======================================================================= Fig. 1
def fig1():
    # 2026-10-02 v5 (저자 지시: v3 패널 그대로, 정보량이 적은 B–D 를 윗줄로 모아 면적당 정보량 균형)
    #   0행 A 삽화(5) · B 과제 의존(2) · C 먹는 위치(2) · D 상태별 섭식 시간(3)
    #   1행 E 단독 누적 회수(4) · F 단독 대 집단(4) · G Lorenz(4)
    # B–D 는 막대 + 자료 점, 역할 색 구분 없음 (사용자 지시).
    H0, H1 = 5.45, 5.55
    fig = F.figure(F.W3, (0.50 + H0 + 0.95 + H1 + 0.10) * CM)
    G_ = F.Grid(fig, 2, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[H0, H1])
    PT0, PB0 = 0.95, 1.75          # 0행 B·C·D
    PT1, PB1 = 0.95, 1.45          # 1행 E–G
    DOT = dict(s=1.6, color=F.OTH, lw=0, alpha=0.55, zorder=2)
    BAR = dict(color='white', ec=F.INK, lw=0.6, zorder=1)
    ERR = dict(fmt='none', ecolor=F.INK, elinewidth=0.7, capsize=1.6, zorder=3)
    rng = np.random.default_rng(7)

    def jit(n, w=0.24):
        return rng.uniform(-w, w, n)

    def title(ax, s):
        F.note(ax, s, x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    # --- A 삽화
    a = fig.add_axes(G_.rect(0, 0, cs=5, padb=0.10, padt=0.05))
    im = mpimg.imread(os.path.join(ART, 'arena_closed.png'))
    IH, IW = im.shape[0], im.shape[1]
    a.imshow(im, extent=[0, IW, IH, 0]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
    a.set_xlim(0, IW); a.set_ylim(IH, 0)
    arr = lambda c='k': dict(arrowstyle='->', lw=0.6, color=c, shrinkA=1, shrinkB=2)
    lab = dict(fontsize=7.0, color=F.INK, ha='center', va='center')
    a.text(330, 300, 'Robot zone', **lab)
    a.text(300, 925, 'Living zone', **lab)
    a.annotate('Snack', xy=(852, 222), xytext=(598, 40), arrowprops=arr(F.PROV), **lab)
    a.annotate('Gate 6 × 10 cm, mouse only', xy=(1104, 613), xytext=(1330, 40),
               arrowprops=arr(F.PROV), annotation_clip=False, **lab)

    # --- B 과제 의존: 먹이가 있을 때만 robot zone 에 들어감 (cohort A)
    #     먹이 있음 = 시행 시작부터 그 쥐의 섭식 시작까지(시행값 954, 2026 원 자료 Fig1c 'Foraging' 구역).
    #     먹이 없음 = 로봇이 먹이 없이 움직인 시행 간 구간 — 원 시행값 없음, 2026-05 원고의 평균 ± SEM 만 (점 없음).
    b = G_.ax(0, 5, cs=2, padl=PADL, padb=PB0, padr=0.06, padt=PT0)
    FB = load('fig1b_robot_zone_food_per_trial.csv'); NB = load('fig1b_robot_zone_nofood_summary.csv').iloc[0]
    v = FB.robot_zone_pct.values
    b.bar(0, v.mean(), 0.62, **BAR); b.errorbar(0, v.mean(), yerr=stats.sem(v), **ERR)
    b.scatter(jit(v.size), v, **DOT)
    b.bar(1, NB.mean_pct, 0.62, **BAR); b.errorbar(1, NB.mean_pct, yerr=NB.sem_pct, **ERR)
    yb = 106
    b.plot([0, 0, 1, 1], [yb - 3, yb, yb, yb - 3], color=F.INK, lw=0.6)
    b.text(0.5, yb + 1.5, f'P {NB.p}', ha='center', va='bottom', fontsize=7.0, color=F.INK)
    b.set_xticks([0, 1]); b.set_xticklabels(['Food', 'No\nfood'], fontsize=7.0)
    b.set_xlim(-0.55, 1.55); b.set_ylim(0, 126); b.set_yticks([0, 50, 100])
    F.tidy(b, 'Cohort A,\n954 mouse-trials', 'Time in robot zone (%)')
    b.yaxis.labelpad = 1.0; b.xaxis.labelpad = 1.5; b.tick_params(**TIGHTY)
    title(b, 'Task dependence')

    # --- C 먹는 위치: 섭식 군집 시간 중 robot zone 안/밖 (cohort A, 시행값; 공사일 제외)
    c = G_.ax(0, 7, cs=2, padl=PADL, padb=PB0, padr=0.06, padt=PT0)
    EL = load('fig1c_eating_location_per_trial.csv')
    EL = EL[(EL.day != 9) & EL.eat_frames.gt(0)]
    for xi, col in ((0, 'inner_pct'), (1, 'outer_pct')):
        w = EL[col].values
        c.bar(xi, w.mean(), 0.62, **BAR)
        c.scatter(xi + jit(w.size), w, **DOT)
    c.set_xticks([0, 1]); c.set_xticklabels(['Robot', 'Living'], fontsize=7.0)
    c.set_xlim(-0.55, 1.55); c.set_ylim(0, 126); c.set_yticks([0, 50, 100])
    F.tidy(c, f'Zone, cohort A\n({len(EL)} trials)', 'Eating time (%)')
    c.yaxis.labelpad = 1.0; c.xaxis.labelpad = 1.5; c.tick_params(**TIGHTY)
    pin = 100 * EL.inner_frames.sum() / EL.eat_frames.sum()
    c.text(0, max(pin, 0) + 4, f'{pin:.2f}%', ha='center', va='bottom', fontsize=7.0, color=F.INK)
    title(c, 'Eating location')

    # --- D 시행 상태별 섭식 시간 (cohort A, 954 mouse-trials; 혼합모형 P = 저자 통계표)
    d1 = G_.ax(0, 9, cs=3, padl=1.10, padb=PB0, padr=0.04, padt=PT0)
    ED = load('fig1d_eat_duration_by_state.csv')
    ST = ['Retrieved', 'Entered', 'Stayed out']
    for xi, st in enumerate(ST):
        w = ED[ED.state == st].eat_duration_s.values
        d1.bar(xi, w.mean(), 0.62, **BAR); d1.errorbar(xi, w.mean(), yerr=stats.sem(w), **ERR)
        d1.scatter(xi + jit(w.size), w, **DOT)
    d1.set_xticks(range(3)); d1.set_xticklabels(['Retrieved', 'Entered', 'Stayed\nout'], fontsize=7.0)
    d1.set_xlim(-0.48, 2.48); d1.set_ylim(0, 230); d1.set_yticks([0, 100, 200])
    F.tidy(d1, 'State on the trial', 'Eating time per trial (s)')
    d1.yaxis.labelpad = 1.0; d1.xaxis.labelpad = 1.5; d1.tick_params(**TIGHTY)
    SD1 = load('fig1d_eat_duration_stats.csv').iloc[0]
    F.note(d1, f'Cohort A, {len(ED)} mouse-trials\nMixed model, P = {SD1.p:.2f}', x=0.5, y=1.02, ha='center',
           va='bottom', size=7.0)

    # --- E 혼자 시행: 시행 시작부터 회수까지의 누적 비율 (cohort A, 108 시행 전체 + 개체별)
    from matplotlib.lines import Line2D
    c1 = G_.ax(1, 0, cs=4, padl=PADL, padb=PB1, padr=0.12, padt=PT1)
    sol = load('fig1b_solitary_latency.csv')
    mice = sorted(sol.mouse.unique())
    def _ecdf(v):
        v = np.sort(np.asarray(v, float)); y = np.arange(1, v.size + 1) / v.size
        return np.r_[v[0], v], np.r_[0.0, y]
    for m in mice:
        xs, ys = _ecdf(sol[sol.mouse == m].total_latency_s)
        c1.step(xs, ys, where='post', color=F.OTHL, lw=0.6, zorder=1)
    xs, ys = _ecdf(sol.total_latency_s)
    c1.step(xs, ys, where='post', color=F.INK, lw=1.2, zorder=3)
    c1.axvline(1200, color=F.OTH, lw=0.6, ls=(0, (2, 2)), zorder=0)
    c1.text(1050, 0.62, 'Trial\nlimit\n(20 min)', ha='right', va='center', fontsize=7.0, color=F.INK,
            linespacing=1.0)
    c1.set_xscale('log'); c1.set_xlim(2, 2000); c1.set_ylim(0, 1.04)
    c1.set_xticks([10, 100, 1000]); c1.set_xticklabels(['10', '100', '1000'])
    c1.set_yticks([0, 0.5, 1.0])
    F.tidy(c1, 'Time from trial start to retrieval (s)', 'Solitary trials retrieved\n(cumulative fraction)')
    c1.yaxis.labelpad = 1.0; c1.xaxis.labelpad = 1.5; c1.tick_params(**TIGHTY)
    kw = stats.kruskal(*[sol[sol.mouse == m].total_latency_s.values for m in mice])
    c1.text(0.88, 0.04, f'Between mice\nP = {kw.pvalue:.2f}', transform=c1.transAxes, ha='right',
            va='bottom', fontsize=7.0, color=F.INK, linespacing=1.15)
    _hc = [Line2D([], [], color=F.INK, lw=1.2, label=f'All {len(sol)} trials'),
           Line2D([], [], color=F.OTHL, lw=0.6, label=f'Each of {len(mice)} mice')]
    lg = c1.legend(handles=_hc, loc='lower left', bbox_to_anchor=(-0.02, 1.0), fontsize=7.0, ncol=2,
                   title=f'Cohort A, alone ({int(sol.retrieved.sum())}/{len(sol)} retrieved)', title_fontsize=7.0,
                   frameon=False, borderpad=0.1, labelspacing=0.18, handlelength=1.2, handletextpad=0.45,
                   columnspacing=0.9)
    lg._legend_box.align = 'left'

    # --- F 단독 대 집단 회수 시행 비율 (46마리)
    c = G_.ax(1, 4, cs=4, padl=PADL, padb=PB1, padr=0.12, padt=PT1)
    SV = load('figS02a_solitary_vs_group_per_mouse.csv')
    rng2 = np.random.default_rng(3); jt = rng2.uniform(-0.085, 0.085, len(SV))
    ccol = lambda role: F.PROV if role == 'provider' else F.OTH
    for (_, r), j in zip(SV.iterrows(), jt):
        c.plot([j, 1 + j], [r.rate_solitary, r.rate_group], color=F.GRID, lw=0.5, zorder=1)
        c.scatter([j, 1 + j], [r.rate_solitary, r.rate_group], s=9, color=ccol(r.role), lw=0, zorder=3)
    c.set_xticks([0, 1]); c.set_xticklabels(['Alone', 'In group'], fontsize=7.0)
    c.set_xlim(-0.45, 1.5); c.set_ylim(-0.04, 1.16); c.set_yticks([0, 0.5, 1.0])
    F.tidy(c, None, 'Trials with retrieval\n(fraction)')
    c.set_xlabel('8 cohorts, 46 mice', fontsize=7.0); c.xaxis.labelpad = 1.5
    c.yaxis.labelpad = 1.0; c.tick_params(**TIGHTY)
    c.add_patch(Rectangle((0.52, SV.equal_share.min()), 1.0,
                          SV.equal_share.max() - SV.equal_share.min(),
                          fc='#eceae6', ec='none', zorder=0))
    c.text(0.47, 0.5 * (SV.equal_share.min() + SV.equal_share.max()), 'Equal\ncontribution (1/n)',
           fontsize=6.5, ha='right', va='center', color=F.INK, linespacing=1.0)
    F.note(c, '{}/{} lower in group\n{} never retrieved'.format(
        int((SV.rate_group < SV.rate_solitary).sum()), len(SV), int((SV.rate_group == 0).sum())),
        x=0.5, y=1.02, ha='center', va='bottom', size=7.0)
    for lab_, col in (('Provider', F.PROV), ('Others', F.OTH)):
        c.scatter([], [], s=9, color=col, lw=0, label=lab_)
    c.legend(loc='center left', fontsize=7.0, borderpad=0.1, labelspacing=0.25,
             handletextpad=0.3, frameon=False, bbox_to_anchor=(-0.02, 0.55))

    # --- G Lorenz (평균 선 + min–max 띠)
    d = G_.ax(1, 8, cs=4, padl=PADL, padb=PB1, padr=0.12, padt=PT1)
    L = load('fig1c_lorenz_points.csv'); G = load('fig1c_gini_by_cohort.csv')
    xg = np.linspace(0, 1, 101)
    Y = np.vstack([np.interp(xg, s.cum_frac_mice, s.cum_share_retrievals) for _, s in L.groupby('cohort')])
    d.plot([0, 1], [0, 1], color=F.GRID, lw=0.6)
    d.fill_between(xg, Y.min(0), Y.max(0), color=F.PROV, alpha=0.22, lw=0)
    d.plot(xg, Y.mean(0), color=F.PROV, lw=1.2)
    d.set_xlim(0, 1); d.set_ylim(0, 1); d.set_xticks([0, 0.5, 1]); d.set_yticks([0, 0.5, 1])
    F.tidy(d, 'Cumulative fraction of mice', 'Cumulative fraction\nof retrievals')
    d.yaxis.labelpad = 1.0; d.xaxis.labelpad = 1.5; d.tick_params(**TIGHTY)
    d.plot([], [], color=F.PROV, lw=1.2, label='Mean, 8 cohorts')
    d.fill_between([], [], [], color=F.PROV, alpha=0.22, label='Min–max')
    d.plot([], [], color=F.GRID, lw=0.6, label='Equality')
    d.legend(loc='upper left', fontsize=7.0, borderpad=0.1, labelspacing=0.25, frameon=False)
    F.note(d, f'Gini = {G.gini.mean():.2f} ± {G.gini.std(ddof=1):.2f}', x=0.05, y=0.40, ha='left', size=7.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 5), ('C', 0, 7), ('D', 0, 9), ('E', 1, 0), ('F', 1, 4), ('G', 1, 8)):
        G_.label(r_, c_, lab_)
    return F.save(fig, 'Fig1', PNG, PDF)


# ======================================================================= Fig. 2
def fig2():
    fig = F.figure(F.W3, 11.4 * CM)
    # 2행 12열. 0행 A(3)·B raster(6)·C(3) / 1행 D(3)·E forest(9)
    G_ = F.Grid(fig, 2, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[1.0, 1.0])

    # --- A 첫 10시행 진입 잠복기 (39마리)
    a = G_.ax(0, 0, cs=3, padl=PADL, padb=1.20, padr=0.12, padt=0.62)
    PE = load('fig1b_early_entry_per_mouse.csv')
    R.strip(a, ['other', 'provider'],
            [PE[PE.role == 'other'].entry_latency_median_s.values,
             PE[PE.role == 'provider'].entry_latency_median_s.values],
            [F.OTH, F.PROV], width=0.26)
    a.set_yscale('log'); a.set_ylim(0.3, 1000)
    a.set_yticks([1, 10, 100]); a.set_yticklabels(['1', '10', '100'])
    a.set_xticks([0, 1]); a.set_xticklabels(['Others', 'Provider'], fontsize=7.0)
    for _t, _c in zip(a.get_xticklabels(), (F.OTH, F.PROV)): _t.set_color(_c)
    a.set_xlim(-0.45, 1.45)
    F.tidy(a, None, 'Entry latency (s)'); a.yaxis.labelpad = 1.0; a.tick_params(**TIGHTY)
    u = stats.mannwhitneyu(PE[PE.role == 'provider'].entry_latency_median_s,
                           PE[PE.role == 'other'].entry_latency_median_s)
    F.note(a, f'39 mice by trial 10\nP = {u.pvalue:.2f}', x=0.5, y=0.98, ha='center', size=7.0)
    a.set_xlabel('8 cohorts, group', fontsize=7.0)

    # --- B 회수자 raster (첫 40시행); 'No retrieval' 범례 삭제(사용자 메모)
    b = G_.ax(0, 3, cs=6, padl=PADL, padb=1.20, padr=0.12, padt=0.62)
    RR = load('fig1f_retrieval_raster.csv')
    CST = {'provider': F.PROV, 'other': F.GRID, 'none': '#ffffff'}
    for i, coh in enumerate(COH):
        for _, r in RR[RR.cohort == coh].iterrows():
            b.add_patch(Rectangle((r.trial - 1.0, i - 0.38), 1, 0.76, fc=CST[r.retrieved_by], ec='none'))
    b.axvline(10, color=F.INK, ls='--', lw=0.6)
    b.set_xlim(0, 40); b.set_ylim(7.6, -0.6)
    b.set_yticks(range(8)); b.set_yticklabels(COH, fontsize=7.0)
    F.tidy(b, 'Trial (first 40 shown)', 'Cohort'); b.yaxis.labelpad = 1.0; b.tick_params(**TIGHTY)
    b.legend(handles=[Patch(fc=CST['provider'], ec='none', label='Final provider retrieved'),
                      Patch(fc=CST['other'], ec='none', label='Another mouse retrieved')],
             loc='lower left', bbox_to_anchor=(-0.005, 1.005), ncol=2, fontsize=7.0, frameon=False,
             borderpad=0.1, handlelength=1.0, handleheight=0.9, handletextpad=0.35, columnspacing=1.2)

    # --- C 10시행 선두의 11시행 이후 회수 비율 (비순환 검증)
    c = G_.ax(0, 9, cs=3, padl=PADL, padb=1.20, padr=0.12, padt=0.62)
    LD = load('fig2c_leader10_prediction.csv')
    LD7 = LD[LD.leader10_ties == 1].reset_index(drop=True)          # 단독 선두 7 cohort
    for _, r in LD7.iterrows():
        c.plot([0, 1], [r.frac_after10_by_leader10, r.frac_after10_by_best_other], color=F.GRID, lw=0.6)
    c.scatter(np.zeros(len(LD7)), LD7.frac_after10_by_leader10, s=10, color=F.PROV, lw=0, zorder=3)
    c.scatter(np.ones(len(LD7)), LD7.frac_after10_by_best_other, s=10, color=F.OTH, lw=0, zorder=3)
    c.set_xticks([0, 1]); c.set_xticklabels(['Leader\nat trial 10', 'Best\nother mouse'], fontsize=7.0)
    for _t, _c in zip(c.get_xticklabels(), (F.PROV, F.OTH)): _t.set_color(_c)
    c.set_xlim(-0.45, 1.45); c.set_ylim(-0.04, 1.16); c.set_yticks([0, 0.5, 1.0])
    F.tidy(c, None, 'Fraction of retrievals\nfrom trial 11 onward'); c.yaxis.labelpad = 1.0; c.tick_params(**TIGHTY)
    n_match = int((LD7.leader10 == LD7.provider_after10).sum())
    F.note(c, f'{n_match}/{len(LD7)} cohorts\nled both periods', x=0.98, y=0.93, ha='right', size=7.0)
    c.set_xlabel('7 cohorts with a sole leader', fontsize=7.0)

    # --- D 첫 10시행 HHI (8 cohorts; y축은 A 와 정렬)
    d = G_.ax(1, 0, cs=3, padl=PADL, padb=1.20, padr=0.12, padt=0.62)
    H = load('fig1d_hhi_first10.csv')
    for _, r in H.iterrows():
        d.plot([0, 1], [r.hhi_observed, r.hhi_equal_expected], color=F.GRID, lw=0.6)
    d.scatter(np.zeros(len(H)), H.hhi_observed, s=9, color=F.PROV, lw=0, zorder=3)
    d.scatter(np.ones(len(H)), H.hhi_equal_expected, s=9, color=F.OTH, lw=0, zorder=3)
    d.set_xticks([0, 1]); d.set_xticklabels(['Observed', 'Equal\npropensity'], fontsize=7.0)
    d.set_xlim(-0.45, 1.45); d.set_ylim(0, 1.22); d.set_yticks([0, 0.5, 1.0])
    F.tidy(d, None, 'HHI, first 10 trials'); d.yaxis.labelpad = 1.0; d.tick_params(**TIGHTY)
    w = stats.wilcoxon(H.hhi_observed, H.hhi_equal_expected)
    F.note(d, f'8 cohorts, P = {w.pvalue:.3f}', x=0.5, y=0.98, ha='center', size=7.0)

    # --- E 효과 크기 forest (95% CI)
    e = G_.ax(1, 3, cs=9, padl=4.45, padb=1.20, padr=0.12, padt=0.62)
    S = load('fig1e_effect_sizes.csv').iloc[::-1].reset_index(drop=True)
    for i, r in S.iterrows():
        col = F.PROV if r.p < 0.05 else F.OTH
        e.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.0)
        e.scatter([r.g], [i], s=13, color=col, lw=0, zorder=3)
    e.axvline(0, color=F.INK, lw=0.5, ls='--')
    e.set_yticks(np.arange(len(S))); e.set_yticklabels(S.measure, fontsize=7.0)
    e.set_ylim(-0.9, len(S) - 0.3); e.set_xlim(-2.3, 6.3)
    e.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6])
    F.tidy(e, 'Provider minus others (Hedges g, 95% CI)', None); e.tick_params(axis='y', length=0, pad=1.5)
    e.spines['left'].set_visible(False)
    e.scatter([], [], s=13, color=F.PROV, label='P < 0.05 (uncorrected)'); e.scatter([], [], s=13, color=F.OTH, label='P ≥ 0.05')
    e.legend(loc='upper right', fontsize=7.0, borderpad=0.1, labelspacing=0.25, frameon=False, bbox_to_anchor=(1.0, 0.86))

    # 패널 문자: B·E 는 플롯 왼쪽 가장자리에 붙인다(사용자 메모 'Move F/H to the right').
    for lab_, r_, c_, dx in (('A', 0, 0, 0.0), ('B', 0, 3, PADL - 0.55), ('C', 0, 9, 0.0),
                             ('D', 1, 0, 0.0), ('E', 1, 3, 4.45 - 3.9)):
        G_.label(r_, c_, lab_, dx=dx)
    return F.save(fig, 'Fig2', PNG, PDF)


if __name__ == '__main__':
    print('saved', fig1()); print('saved', fig2())
