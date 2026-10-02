# -*- coding: utf-8 -*-
"""Fig. 1 과 fig. S1–S3 생성. data/ 의 파일만 읽는다."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.image as mpimg
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import arena

HERE = os.path.dirname(os.path.abspath(__file__))
SUB  = os.path.dirname(HERE)
D    = os.path.join(SUB, 'data')
ART  = os.path.join(SUB, 'art')
PNG  = os.path.join(SUB, 'png')
PDF  = os.path.join(SUB, 'pdf')
CM   = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
COH  = list('ABCDEFGH')

# 패널 문자는 플롯 하나당 하나씩 부여한다 (Science 규정: 대문자 A, B, C ...).
TAGS = {'fig1': '', 'figS01': '', 'figS02': '', 'figS03': ''}


TIGHTY = dict(axis='y', labelsize=7.0, length=2.0, pad=1.0)


def img(ax, name):
    ax.imshow(mpimg.imread(os.path.join(ART, name)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def irect(fig, x, y, w, name):
    """이미지 비율에 맞춘 패널 rect (imshow 가 여백을 남기지 않도록)."""
    im = mpimg.imread(os.path.join(ART, name))
    ar = im.shape[1] / im.shape[0]
    fw, fh = fig.get_size_inches()
    return [x, y, w, (w * fw / ar) / fh]


def girect(G_, r, c, cs, name, padl=0.0, padr=0.0, padt=0.0):
    """격자 셀 안에 이미지 비율대로 얹되 위쪽 기준선에 맞춘다."""
    im = mpimg.imread(os.path.join(ART, name))
    ar = im.shape[1] / im.shape[0]
    x, y, w, h = G_.cell_cm(r, c, cs=cs)
    w2 = w - padl - padr
    h2 = w2 / ar
    top = y + h - padt
    return [(x + padl) / G_.W, (top - h2) / G_.H, w2 / G_.W, h2 / G_.H]


def strip(ax, groups, values, color, width=0.24, rng=None):
    rng = rng or np.random.default_rng(3)
    for i, g in enumerate(groups):
        v = np.asarray(values[i], float); v = v[np.isfinite(v)]
        ax.scatter(np.full(v.size, i) + rng.uniform(-width, width, v.size), v,
                   s=2.2, color=color if isinstance(color, str) else color[i], lw=0, alpha=0.75)
        ax.plot([i - width * 1.4, i + width * 1.4], [np.median(v)] * 2, color=F.INK, lw=1.0)


def dunn(groups, pairs):
    """Kruskal-Wallis 후 Dunn 사후검정 (동순위 보정 + Holm)."""
    allv = np.concatenate(groups)
    rk = stats.rankdata(allv)
    N = len(allv)
    _, cnt = np.unique(allv, return_counts=True)
    ties = (cnt ** 3 - cnt).sum()
    sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))
    idx, mr, ns = 0, [], []
    for g in groups:
        mr.append(rk[idx:idx + len(g)].mean()); ns.append(len(g)); idx += len(g)
    raw = []
    for i, j in pairs:
        z = abs(mr[i] - mr[j]) / np.sqrt(sigma2 * (1.0 / ns[i] + 1.0 / ns[j]))
        raw.append(2 * (1 - stats.norm.cdf(z)))
    o = np.argsort(raw); adj = np.empty(len(raw)); run = 0.0
    for rank, k in enumerate(o):
        run = max(run, min(1.0, raw[k] * (len(raw) - rank))); adj[k] = run
    return adj


def bars(ax, labels, mean, err, colors):
    x = np.arange(len(labels))
    ax.bar(x, mean, 0.62, color=colors, lw=0)
    if err is not None and np.any(np.asarray(err, float) > 0):
        ax.errorbar(x, mean, yerr=err, fmt='none', ecolor=F.INK, elinewidth=0.6, capsize=1.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)


# ======================================================================= Fig. 1
def fig1():
    fig = F.figure(F.W3, 15.5 * CM)
    # 공통 격자 3행 12열
    #  1행 A 삽화(6열) · B(3) · C(3)
    #  2행 D(3) · E(3) · F raster(6)
    #  3행 G(4) · H forest(8)
    G_ = F.Grid(fig, 3, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[1.00, 1.15, 1.00])

    # --- A 삽화 (닫힌 벽·좁은 게이트·바깥에 더 많은 쥐)
    a = fig.add_axes(G_.rect(0, 0, cs=6, padb=0.15, padt=0.05))
    im = mpimg.imread(os.path.join(ART, 'arena_closed.png'))
    IH, IW = im.shape[0], im.shape[1]
    a.imshow(im, extent=[0, IW, IH, 0]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
    a.set_xlim(0, IW); a.set_ylim(IH, 0)
    arr = lambda c='k': dict(arrowstyle='->', lw=0.6, color=c, shrinkA=1, shrinkB=2)
    lab = dict(fontsize=7.0, color=F.INK, ha='center', va='center')
    a.text(330, 300, 'Robot zone', **lab)
    a.text(300, 925, 'Living zone', **lab)
    a.annotate('Snack', xy=(852, 222), xytext=(598, 58), arrowprops=arr(F.PROV), **lab)
    a.annotate('Gate 6 × 10 cm\n(Mouse only)', xy=(1104, 613), xytext=(1369, 128),
               arrowprops=arr(F.PROV), **lab)


    # --- B 혼자 시행 잠복기 + 8 cohort 초기 진입
    b = G_.ax(0, 6, cs=3, padl=1.22, padb=1.20, padr=0.12, padt=0.42)
    sol = load('fig1b_solitary_latency.csv')
    mice = sorted(sol.mouse.unique())
    strip(b, mice, [sol[sol.mouse == m].total_latency_s.values for m in mice], F.OTH, width=0.30)
    b.set_yscale('log'); b.set_ylim(5, 10000)
    b.set_yticks([10, 100, 1000]); b.set_yticklabels(['10', '100', '1000'])
    b.set_xticks(range(len(mice))); b.set_xticklabels(mice, fontsize=7.0)
    F.tidy(b, None, 'Retrieval latency (s)'); b.yaxis.labelpad = 1.0; b.tick_params(**TIGHTY)
    import warnings; warnings.filterwarnings('ignore')
    kw = stats.kruskal(*[sol[sol.mouse == m].total_latency_s.values for m in mice])
    F.note(b, f'108/108 trials\nretrieved\nP = {kw.pvalue:.2f}', x=0.5, ha='center', size=7.0)
    b.set_xlabel('Cohort A, alone', fontsize=7.0)

    b2 = G_.ax(0, 9, cs=3, padl=1.22, padb=1.20, padr=0.12, padt=0.42)
    PE = load('fig1b_early_entry_per_mouse.csv')
    strip(b2, ['other', 'provider'],
          [PE[PE.role == 'other'].entry_latency_median_s.values,
           PE[PE.role == 'provider'].entry_latency_median_s.values],
          [F.OTH, F.PROV], width=0.26)
    b2.set_yscale('log'); b2.set_ylim(0.3, 1000)
    b2.set_yticks([1, 10, 100]); b2.set_yticklabels(['1', '10', '100'])
    b2.set_xticks([0, 1]); b2.set_xticklabels(['Others', 'Provider'], fontsize=7.0)
    for _t, _c in zip(b2.get_xticklabels(), (F.OTH, F.PROV)):   # 조건 색을 눈금 글자에
        _t.set_color(_c)
    b2.set_xlim(-0.45, 1.45)
    F.tidy(b2, None, 'Entry latency (s)'); b2.yaxis.labelpad = 1.0; b2.tick_params(**TIGHTY)
    u = stats.mannwhitneyu(PE[PE.role == 'provider'].entry_latency_median_s,
                           PE[PE.role == 'other'].entry_latency_median_s)
    F.note(b2, f'39 mice\nby trial 10\nP = {u.pvalue:.2f}', x=0.5, ha='center', size=7.0)
    b2.set_xlabel('8 cohorts, group', fontsize=7.0)


    # --- C Lorenz (평균 선 + min–max 띠) + 역할별 섭식
    c1 = G_.ax(1, 0, cs=3, padl=1.42, padb=1.45, padr=0.12, padt=0.95)
    L = load('fig1c_lorenz_points.csv'); G = load('fig1c_gini_by_cohort.csv')
    xg = np.linspace(0, 1, 101)
    Y = np.vstack([np.interp(xg, s.cum_frac_mice, s.cum_share_retrievals)
                   for _, s in L.groupby('cohort')])
    c1.plot([0, 1], [0, 1], color=F.GRID, lw=0.6)
    c1.fill_between(xg, Y.min(0), Y.max(0), color=F.PROV, alpha=0.22, lw=0)
    c1.plot(xg, Y.mean(0), color=F.PROV, lw=1.2)
    c1.set_xlim(0, 1); c1.set_ylim(0, 1); c1.set_xticks([0, 0.5, 1]); c1.set_yticks([0, 0.5, 1])
    F.tidy(c1, 'Cumulative fraction\nof mice', 'Cumulative fraction\nof retrievals')
    c1.yaxis.labelpad = 1.0; c1.xaxis.labelpad = 1.5; c1.tick_params(**TIGHTY)
    c1.plot([], [], color=F.PROV, lw=1.2, label='Mean')
    c1.fill_between([], [], [], color=F.PROV, alpha=0.22, label='Min–max')
    c1.plot([], [], color=F.GRID, lw=0.6, label='Equality')
    c1.legend(loc='upper left', fontsize=7.0, borderpad=0.1, labelspacing=0.25)
    F.note(c1, f'Gini = {G.gini.mean():.2f} ± {G.gini.std(ddof=1):.2f}',
           x=0.05, y=0.42, ha='left', size=7.0)

    c2 = G_.ax(1, 3, cs=3, padl=1.28, padb=1.45, padr=0.12, padt=0.95)
    EP = load('fig1c_eating_per_mouse_published.csv')
    cmap = {'Providing': F.PROV, 'Venturing': F.OTHL, 'Freeriding': F.INK}
    x = np.arange(len(EP))
    c2.bar(x, EP.eating_duration_mean_s, 0.72,
           color=[cmap[r] for r in EP.dominant_role], lw=0)
    c2.errorbar(x, EP.eating_duration_mean_s, yerr=EP.eating_duration_sd_s, fmt='none',
                ecolor=F.INK, elinewidth=0.6, capsize=1.2)
    c2.set_xticks(x); c2.set_xticklabels(EP.mouse, fontsize=7.0)
    c2.set_xlim(-0.7, len(EP) - 0.3)
    c2.set_ylim(50, 265)
    c2.set_yticks([50, 100, 150, 200])
    F.tidy(c2, 'Mouse', 'Eating time (s)')
    c2.yaxis.labelpad = 1.0; c2.xaxis.labelpad = 1.5; c2.tick_params(**TIGHTY)
    F.note(c2, f'P = {EP.p_duration.iloc[0]:.2f}', x=0.97, y=0.74, ha='right', va='top', size=7.0)
    from matplotlib.patches import Rectangle as _Rect
    RLAB = {'Providing': 'Retrieved', 'Venturing': 'Entered', 'Freeriding': 'Stayed out'}
    _h = [_Rect((0, 0), 1, 1, fc=col, ec='none', label=RLAB[r]) for r, col in cmap.items()]
    c2.legend(handles=_h, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=7.0, frameon=False, borderpad=0.1, handlelength=1.0,
              handleheight=0.9, handletextpad=0.35, columnspacing=1.0)

    # --- D 선두 raster + 첫 10시행 집중도
    d1 = G_.ax(1, 6, cs=6, padl=1.28, padb=1.45, padr=0.12, padt=0.95)
    from matplotlib.patches import Rectangle
    R = load('fig1f_retrieval_raster.csv')
    CST = {'provider': F.PROV, 'other': F.GRID, 'none': '#ffffff'}
    for i, coh in enumerate(COH):
        for _, r in R[R.cohort == coh].iterrows():
            d1.add_patch(Rectangle((r.trial - 1.0, i - 0.38), 1, 0.76,
                                   fc=CST[r.retrieved_by], ec='none'))
    d1.axvline(10, color=F.INK, ls='--', lw=0.6)
    d1.set_xlim(0, 40); d1.set_ylim(7.6, -0.6)
    d1.set_yticks(range(8)); d1.set_yticklabels(COH, fontsize=7.0)
    F.tidy(d1, 'Trial (first 40 shown)', 'Cohort')
    from matplotlib.patches import Patch
    d1.legend(handles=[Patch(fc=CST['provider'], ec='none', label='Final provider retrieved'),
                       Patch(fc=CST['other'], ec='none', label='Another mouse retrieved'),
                       Patch(fc='#ffffff', ec=F.OTH, lw=0.4, label='No retrieval')],
              loc='lower left', bbox_to_anchor=(-0.005, 1.005), ncol=2, fontsize=7.0,
              frameon=False, borderpad=0.1, handlelength=1.0, handleheight=0.9,
              handletextpad=0.35, columnspacing=1.2)


    d2 = G_.ax(2, 0, cs=4, padl=1.28, padb=1.40, padr=0.12, padt=0.42)
    H = load('fig1d_hhi_first10.csv')
    for _, r in H.iterrows():
        d2.plot([0, 1], [r.hhi_observed, r.hhi_equal_expected], color=F.GRID, lw=0.6)
    d2.scatter(np.zeros(len(H)), H.hhi_observed, s=9, color=F.PROV, lw=0, zorder=3)
    d2.scatter(np.ones(len(H)), H.hhi_equal_expected, s=9, color=F.OTH, lw=0, zorder=3)
    d2.set_xticks([0, 1]); d2.set_xticklabels(['Observed', 'Equal\npropensity'], fontsize=7.0)
    d2.set_xlim(-0.45, 1.45); d2.set_ylim(0, 1.05)
    F.tidy(d2, None, 'HHI, first 10 trials'); d2.yaxis.labelpad = 1.0; d2.tick_params(**TIGHTY)
    w = stats.wilcoxon(H.hhi_observed, H.hhi_equal_expected)
    F.note(d2, f'P = {w.pvalue:.3f}', x=0.5, y=1.02, ha='center', va='bottom')


    # --- E 효과 크기 forest (유의성으로 색 구분)
    e = G_.ax(2, 4, cs=8, padl=4.45, padb=1.40, padr=0.12, padt=0.42)
    S = load('fig1e_effect_sizes.csv').iloc[::-1].reset_index(drop=True)
    for i, r in S.iterrows():
        col = F.PROV if r.p < 0.05 else F.OTH
        e.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.0)
        e.scatter([r.g], [i], s=13, color=col, lw=0, zorder=3)
    e.axvline(0, color=F.INK, lw=0.5, ls='--')
    e.set_yticks(np.arange(len(S))); e.set_yticklabels(S.measure, fontsize=7.0)
    e.set_ylim(-0.9, len(S) - 0.3)
    F.tidy(e, 'Provider vs others (Hedges g)', None)
    e.scatter([], [], s=13, color=F.PROV, label='P < 0.05')
    e.scatter([], [], s=13, color=F.OTH, label='P ≥ 0.05')
    e.legend(loc='upper right', fontsize=7.0, borderpad=0.1, labelspacing=0.25,
             frameon=False, bbox_to_anchor=(1.0, 0.86))
    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 6), ('C', 0, 9),
                         ('D', 1, 0), ('E', 1, 3), ('F', 1, 6),
                         ('G', 2, 0), ('H', 2, 4)):
        G_.label(r_, c_, TAGS['fig1'] + lab_)
    return F.save(fig, 'Fig1', PNG, PDF)


# ======================================================================= fig. S1
def figS01():
    from matplotlib.patches import Rectangle
    fig = F.figure(F.W3, 17.5 * CM)

    a = F.panel(fig, irect(fig, 0.205, 0.752, 0.590, 'task_sequence.png'))
    img(a, 'task_sequence.png')
    F.label(fig, TAGS['figS01'] + 'A', 0.005, 0.988)

    b = F.panel(fig, [0.065, 0.450, 0.300, 0.225])
    S = load('figS01b_cohort_schedule.csv'); yy = np.arange(len(S))
    RM = load('figS01b_retrievals_per_mouse.csv')
    import matplotlib
    SHADE = matplotlib.colors.LinearSegmentedColormap.from_list(
        'share', ['#f6f1ea', '#f3c9a6', '#ef9a5e', F.PROV, '#8f3c0c'])
    for i, r in S.iterrows():
        left = 0.0
        for _, q in RM[RM.cohort == r.cohort].sort_values('rank').iterrows():
            if q.n_retrieved == 0:
                continue
            b.barh(i, q.n_retrieved, 0.62, left=left,
                   color=SHADE(q.share_of_cohort_retrievals), ec='white', lw=0.35)
            left += q.n_retrieved
        b.text(left + 5, i, f'{r.n_mice} mice · {r.n_days} days',
               va='center', fontsize=7.0, color=F.INK)
    b.set_yticks(yy); b.set_yticklabels(S.cohort, fontsize=7.0); b.invert_yaxis()
    b.set_xlim(0, S.n_trials.max() * 1.62)
    F.tidy(b, 'Retrievals (n), one segment per mouse', 'Cohort')
    cax = fig.add_axes([0.248, 0.512, 0.105, 0.009])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=SHADE)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(['0', '0.5', '1'])
    cb.ax.tick_params(labelsize=7.0, length=1.5, pad=1, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label('Fraction of cohort retrievals', fontsize=7.0, labelpad=1.5)
    F.label(fig, TAGS['figS01'] + 'B', 0.005, 0.718)

    c = F.panel(fig, irect(fig, 0.400, 0.437, 0.310, 'arena_top.png'))
    img(c, 'arena_top.png')
    c2 = F.panel(fig, irect(fig, 0.728, 0.437, 0.258, 'arena_side.png'))
    img(c2, 'arena_side.png')
    F.label(fig, TAGS['figS01'] + 'C', 0.382, 0.718)
    F.label(fig, TAGS['figS01'] + 'D', 0.710, 0.718)

    d = F.panel(fig, [0.075, 0.085, 0.390, 0.245])
    Rr = load('figS01d_state_raster_cohortA.csv')
    mice = sorted(Rr.mouse.unique())
    cmap = {'Retrieved': F.PROV, 'Entered': F.OTHL, 'Stayed out': F.OTH}
    for i, m in enumerate(mice):
        for _, r in Rr[Rr.mouse == m].iterrows():
            d.add_patch(Rectangle((r.trial - 0.5, i - 0.42), 1, 0.84, fc=cmap[r.state.capitalize()], ec='none'))
    d.set_xlim(0.5, Rr.trial.max() + 0.5); d.set_ylim(len(mice) - 0.4, -0.6)
    d.set_yticks(range(len(mice))); d.set_yticklabels(mice, fontsize=7.0)
    F.tidy(d, 'Trial', 'Mouse (cohort A)')
    for k, (lab_, col) in enumerate(cmap.items()):
        d.add_patch(Rectangle((0.02 + k * 0.22, 1.05), 0.03, 0.055, transform=d.transAxes,
                              fc=col, ec='none', clip_on=False))
        d.text(0.06 + k * 0.22, 1.078, lab_, transform=d.transAxes, fontsize=7.0,
               va='center', color=F.INK)
    F.label(fig, TAGS['figS01'] + 'E', 0.005, 0.355)

    e1 = F.panel(fig, [0.580, 0.245, 0.300, 0.125]); img(e1, 'recording.png')
    e2 = F.panel(fig, [0.545, 0.030, 0.370, 0.195]); img(e2, 'histology_photos.png')
    F.label(fig, TAGS['figS01'] + 'F', 0.520, 0.395)
    F.label(fig, TAGS['figS01'] + 'G', 0.520, 0.238)
    return F.save(fig, 'figS01', PNG, PDF)


# ======================================================================= fig. S2
def figS02():
    """R1b–R2b 를 뒷받침하는 보조 패널 (본문 서술 순서대로 A–N)."""
    from matplotlib.patches import Rectangle
    fig = F.figure(F.W3, 19.5 * CM)
    # 공통 격자 4행 12열
    #  0행 A(3)·B(3)·C(6) / 1행 D(3)·E 소격자(3)·F(3)·G(3)
    #  2행 H 삽화(2)+막대(2)·I 삽화(1)+도넛(3)·J(4) / 3행 K(3)·L(3)·M(4)·N(2)
    G_ = F.Grid(fig, 4, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95)
    rng = np.random.default_rng(11)
    R1, R2, R3, R4 = 0.805, 0.575, 0.320, 0.055
    HR = 0.155
    ccol = lambda role: F.PROV if role == 'provider' else F.OTH

    # ---------------------------------------------------- A·B 개체별 혼자 대 집단
    # A: 46마리 전원. 단독 채집에서는 모두 회수(20분 제한시간)하므로 혼자 쪽은 1.0
    # B: 회수 소요 시간은 fig. S5B 자료에서 두 조건 모두 있는 개체의 중앙값
    SV = load('figS02a_solitary_vs_group_per_mouse.csv')
    jt = rng.uniform(-0.085, 0.085, len(SV))
    a1 = G_.ax(0, 0, cs=3, padl=1.42, padb=1.20, padr=0.12, padt=0.85)
    a1.add_patch(Rectangle((0.52, SV.equal_share.min()), 1.0,
                           SV.equal_share.max() - SV.equal_share.min(),
                           fc='#eceae6', ec='none', zorder=0))
    a1.text(1.47, SV.equal_share.max() + 0.02, 'Equal share (1/n)',
            fontsize=7.0, ha='right', va='bottom', color=F.INK)
    for (_, r), j in zip(SV.iterrows(), jt):
        a1.plot([j, 1 + j], [r.rate_solitary, r.rate_group], color=F.GRID, lw=0.5, zorder=1)
        a1.scatter([j, 1 + j], [r.rate_solitary, r.rate_group], s=9,
                   color=ccol(r.role), lw=0, zorder=3)
    a1.set_xticks([0, 1]); a1.set_xticklabels(['Alone', 'In group'], fontsize=7.0)
    a1.set_xlim(-0.45, 1.5); a1.set_ylim(-0.04, 1.16)
    F.tidy(a1, None, 'Trials with retrieval (fraction)')
    a1.set_xlabel('8 cohorts, 46 mice', fontsize=7.0); a1.xaxis.labelpad = 1.5
    a1.yaxis.labelpad = 1.0; a1.tick_params(**TIGHTY)
    wr = stats.wilcoxon(SV.rate_solitary, SV.rate_group)
    _m, _e = ('%.0e' % wr.pvalue).split('e')
    F.note(a1, '{}/{} lower in group\nP = {} $\\times$ 10$^{{{}}}$'.format(
        int((SV.rate_group < SV.rate_solitary).sum()), len(SV), _m, int(_e)),
        x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    RL = load('figS05b_retrieval_solitary_vs_group.csv')
    PL = (RL.groupby(['cohort', 'mouse', 'role', 'condition']).retrieval_latency_s.median()
            .unstack('condition').dropna().reset_index())
    jt2 = rng.uniform(-0.085, 0.085, len(PL))
    a2 = G_.ax(0, 3, cs=3, padl=1.42, padb=1.20, padr=0.12, padt=0.85)
    for (_, r), j in zip(PL.iterrows(), jt2):
        a2.plot([j, 1 + j], [r.Solitary, r.Group], color=F.GRID, lw=0.5, zorder=1)
        a2.scatter([j, 1 + j], [r.Solitary, r.Group], s=9,
                   color=ccol(r.role), lw=0, zorder=3)
    a2.set_yscale('log'); a2.set_ylim(1.5, 320)
    a2.set_yticks([10, 100]); a2.set_yticklabels(['10', '100'])
    a2.set_xticks([0, 1]); a2.set_xticklabels(['Alone', 'In group'], fontsize=7.0)
    a2.set_xlim(-0.45, 1.5)
    F.tidy(a2, None, 'Latency to retrieve (s)')
    a2.set_xlabel('8 cohorts, 18 mice', fontsize=7.0); a2.xaxis.labelpad = 1.5
    a2.yaxis.labelpad = 1.0; a2.tick_params(**TIGHTY)
    wl = stats.wilcoxon(PL.Group, PL.Solitary)
    F.note(a2, '{}/{} slower in group\nP = {:.2f}'.format(
        int((PL.Group > PL.Solitary).sum()), len(PL), wl.pvalue),
        x=0.5, y=1.02, ha='center', va='bottom', size=7.0)
    for lab_, col in (('Provider', F.PROV), ('Others', F.OTH)):
        a2.scatter([], [], s=9, color=col, lw=0, label=lab_)
    a2.legend(loc='upper left', fontsize=7.0, borderpad=0.1, labelspacing=0.25,
              handletextpad=0.3, frameon=False, bbox_to_anchor=(-0.02, 1.02))

    # ---------------------------------------------------- C 회수율 이봉 분포
    b = G_.ax(0, 6, cs=6, padl=1.42, padb=1.20, padr=0.12, padt=0.85)
    RR = load('figS02b_retrieval_rate_per_mouse.csv')
    bins = np.linspace(0, 1, 21)
    b.hist(RR[RR.role == 'other'].retrieval_rate, bins=bins, color=F.OTH, lw=0)
    b.hist(RR[RR.role == 'provider'].retrieval_rate, bins=bins, color=F.PROV, lw=0)
    b.axvline(0.43, color=F.INK, ls='--', lw=0.6)
    F.tidy(b, 'Retrieval rate (fraction of trials)', 'Mice (n)')
    b.xaxis.labelpad = 1.5; b.yaxis.labelpad = 1.0; b.tick_params(**TIGHTY)
    F.note(b, f'{len(RR)} mice · decision boundary 0.43', x=0.5, y=0.97,
           ha='center', va='top', size=7.0)

    # ---------------------------------------------------- D 제공자 대 나머지 몫
    c = G_.ax(1, 0, cs=3, padl=1.42, padb=1.40, padr=0.12, padt=0.45)
    SH = load('figS02c_share_provider_vs_others.csv')
    for _, r in SH.iterrows():
        c.plot([0, 1], [r.provider_share, r.others_mean_share], color=F.GRID, lw=0.6)
    c.scatter(np.zeros(len(SH)), SH.provider_share, s=10, color=F.PROV, lw=0, zorder=3)
    c.scatter(np.ones(len(SH)), SH.others_mean_share, s=10, color=F.OTH, lw=0, zorder=3)
    c.set_xticks([0, 1]); c.set_xticklabels(['Provider', 'Others\n(mean)'], fontsize=7.0)
    for _t, _c in zip(c.get_xticklabels(), (F.PROV, F.OTH)):
        _t.set_color(_c)
    c.set_xlim(-0.45, 1.45); c.set_ylim(0, 1.12)
    F.tidy(c, None, 'Fraction of retrievals')
    c.yaxis.labelpad = 1.0; c.tick_params(**TIGHTY)
    w = stats.wilcoxon(SH.provider_share, SH.others_mean_share)
    F.note(c, f'P = {w.pvalue:.3f}', x=0.5, y=0.97, ha='center', va='top', size=7.0)

    # ---------------------------------------------------- E cohort별 Lorenz
    L = load('fig1c_lorenz_points.csv'); G = load('fig1c_gini_by_cohort.csv')
    for k, coh in enumerate(COH):
        _mc = G_.cell_cm(1, 3, cs=3)
        _mw = (_mc[2] - 0.30) / 4.0
        _mh = (_mc[3] - 1.40 - 0.45) / 2.0
        ax = fig.add_axes([(_mc[0] + 0.15 + (k % 4) * _mw) / G_.W,
                           (_mc[1] + 1.40 + (1 - k // 4) * _mh) / G_.H,
                           (_mw - 0.10) / G_.W, (_mh - 0.10) / G_.H])
        s = L[L.cohort == coh]
        ax.plot([0, 1], [0, 1], color=F.GRID, lw=0.5)
        ax.plot(s.cum_frac_mice, s.cum_share_retrievals, color=F.PROV, lw=0.8, marker='o', ms=1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.04, 0.93, f'{coh}  {G[G.cohort == coh].gini.iloc[0]:.2f}',
                transform=ax.transAxes, fontsize=7.0, va='top')

    # ---------------------------------------------------- F·G 역할별 섭식
    EP = load('fig1c_eating_per_mouse_published.csv')
    cmap = {'Providing': F.PROV, 'Venturing': F.OTHL, 'Freeriding': F.INK}
    cols = [cmap[r] for r in EP.dominant_role]; x = np.arange(len(EP))
    e1 = G_.ax(1, 6, cs=3, padl=1.42, padb=1.40, padr=0.12, padt=0.45)
    e2 = G_.ax(1, 9, cs=3, padl=1.42, padb=1.40, padr=0.12, padt=0.45)
    for ax, val, sd, ylab, pv in ((e1, EP.eating_duration_mean_s, EP.eating_duration_sd_s,
                                   'Eating time (s)', EP.p_duration.iloc[0]),
                                  (e2, EP.latency_to_eat_mean_s, EP.latency_to_eat_sd_s,
                                   'Latency to eat (s)', EP.p_latency.iloc[0])):
        ax.bar(x, val, 0.72, color=cols, lw=0)
        ax.errorbar(x, val, yerr=sd, fmt='none', ecolor=F.INK, elinewidth=0.6, capsize=1.2)
        ax.set_xticks(x); ax.set_xticklabels(EP.mouse, fontsize=7.0, rotation=90)
        ax.set_ylim(0, (val + sd).max() * 1.30)
        F.tidy(ax, 'Mouse', ylab)
        ax.yaxis.labelpad = 1.0; ax.xaxis.labelpad = 1.5; ax.tick_params(**TIGHTY)
        F.note(ax, f'P = {pv:.2f}', x=0.5, y=0.97, ha='center', va='top', size=7.0)
    _RLAB = {'Providing': 'Retrieved', 'Venturing': 'Entered', 'Freeriding': 'Stayed out'}
    wid = sum(0.028 + 0.0052 * len(_RLAB[r_]) for r_ in cmap)   # F·G 두 패널의 가운데에 배치
    xo = 0.5 * (0.607 + 0.959) - wid / 2
    for r_, col in cmap.items():
        fig.patches.append(Rectangle((xo, R2 - 0.060), 0.011, 0.013, transform=fig.transFigure,
                                     fc=col, ec='none'))
        fig.text(xo + 0.015, R2 - 0.0535, _RLAB[r_], fontsize=7.0, va='center', color=F.INK)
        xo += 0.028 + 0.0052 * len(_RLAB[r_])

    # ---------------------------------------------------- H 고정 먹이 대조
    h1 = F.panel(fig, girect(G_, 2, 0, 2, 'fixed_snack.png', padt=0.45))
    img(h1, 'fixed_snack.png')
    h1.text(0.576, 1.03, 'Snack fixed', transform=h1.transAxes, ha='center', va='bottom',
            fontsize=7.0, color=F.INK)
    h2 = G_.ax(2, 2, cs=3, padl=1.42, padb=1.20, padr=0.12, padt=0.70)
    FX = load('figS02h_fixed_vs_moving.csv')
    for _, r in FX.iterrows():
        h2.plot([0, 1], [r.rate_moving, r.rate_fixed], color=F.GRID, lw=0.6, zorder=1)
        h2.scatter([0, 1], [r.rate_moving, r.rate_fixed], s=9,
                   color=ccol(r.role), lw=0, zorder=3)
    h2.set_xticks([0, 1]); h2.set_xticklabels(['Moving', 'Fixed'], fontsize=7.0)
    h2.set_xlim(-0.45, 1.45); h2.set_ylim(-0.04, 1.18)
    F.tidy(h2, None, 'Retrieval rate')
    h2.yaxis.labelpad = 1.0; h2.tick_params(**TIGHTY)
    wfx = stats.wilcoxon(FX.rate_fixed, FX.rate_moving)
    F.note(h2, f'{len(FX)} mice\nP = {wfx.pvalue:.3f}', x=0.5, y=0.99,
           ha='center', va='top', size=7.0)

    # ---------------------------------------------------- I 닫힌 칸에서의 결과
    i1 = F.panel(fig, girect(G_, 2, 5, 1, 'closed_chamber.png', padt=0.45))
    img(i1, 'closed_chamber.png')
    i2 = F.panel(fig, G_.rect(2, 6, cs=2, padl=0.45, padb=1.10, padr=0.45, padt=0.95))
    CO = load('figS02i_chamber_outcome.csv').set_index('outcome')
    vals = [float(CO.loc['Eat alone', 'n_trials']), float(CO.loc['Bring out', 'n_trials'])]
    tot = sum(vals)
    i2.pie(vals, colors=[F.OTHL, F.PROV], startangle=90, counterclock=False,
           radius=1.0, wedgeprops=dict(width=0.42, lw=0.6, edgecolor='white'))
    i2.set_aspect('equal')
    i2.text(0, 0, f'Ate\ninside\n{vals[0] / tot * 100:.0f}%', ha='center', va='center',
            fontsize=7.0, color=F.INK)
    i2.annotate(f'Carried out to\nthe group\n{vals[1] / tot * 100:.0f}% ({int(vals[1])} trials)',
                xy=(0.40, 1.02), xytext=(-0.10, 2.05),
                fontsize=7.0, color=F.INK, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.6, color=F.PROV, shrinkA=1, shrinkB=2))
    F.note(i2, f'{int(tot)} trials', x=0.5, y=-0.02, ha='center', va='top', size=7.0)

    # ---------------------------------------------------- J 상태별 운동 에너지
    h = G_.ax(2, 8, cs=4, padl=1.52, padb=1.20, padr=0.12, padt=0.45)
    KE = load('figS02h_kinetic_energy.csv')
    order = ['retrieved', 'entered', 'stayed out']
    vals_ke = [KE[KE.state == r].ke_foraging.dropna().values for r in order]
    strip(h, order, vals_ke, [F.PROV, F.OTHL, F.INK2], rng=rng)
    h.set_yscale('log'); h.set_ylim(1e-9, 3e-1)
    h.set_xticks(range(3)); h.set_xticklabels([o.capitalize() for o in order], fontsize=7.0)
    F.tidy(h, None, 'Kinetic energy (a.u.)')
    h.yaxis.labelpad = 1.0; h.tick_params(**TIGHTY)
    pairs = [(0, 1), (0, 2), (1, 2)]
    adj = dunn(vals_ke, pairs)          # Kruskal-Wallis 후 Dunn 사후검정 (Holm 보정)
    kw_all = stats.kruskal(*vals_ke)
    F.note(h, 'Kruskal-Wallis P < 0.001' if kw_all.pvalue < 1e-3
           else f'Kruskal-Wallis P = {kw_all.pvalue:.3f}', x=0.5, y=0.04,
           ha='center', va='bottom', size=7.0)
    for (i_, j_), pv, yy in zip(pairs, adj, (3e-3, 2.2e-1, 2.6e-2)):
        txt = 'P < 0.001' if pv < 1e-3 else f'P = {pv:.3f}'
        h.plot([i_, i_, j_, j_], [yy * 0.45, yy, yy, yy * 0.45], color=F.INK, lw=0.5)
        h.text((i_ + j_) / 2, yy * 1.25, txt, ha='center', va='bottom', fontsize=7.0, color=F.INK)

    # ---------------------------------------------------- K 세션 내 집중도 (R2a)
    k1 = G_.ax(3, 0, cs=3, padl=1.42, padb=1.55, padr=0.12, padt=0.45)
    Q = load('figS03c_hhi_by_quintile.csv')
    piv = Q.pivot_table(index='q', columns='cohort', values='HHI')
    k1.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=F.PROV, alpha=0.20, lw=0)
    k1.plot(piv.index, piv.mean(axis=1), color=F.PROV, lw=1.2)
    k1.set_xticks(list(piv.index)); k1.set_ylim(0, 1.05)
    F.tidy(k1, 'Session quintile', 'HHI')
    k1.yaxis.labelpad = 1.0; k1.xaxis.labelpad = 1.5; k1.tick_params(**TIGHTY)
    rho, pv = stats.spearmanr(Q.q, Q.HHI)
    F.note(k1, f'ρ = {rho:.2f}, P = {pv:.3f}', x=0.02, y=0.05, ha='left', va='bottom', size=7.0)

    # ---------------------------------------------------- L 연속 회수 구간
    l1 = G_.ax(3, 3, cs=3, padl=1.42, padb=1.55, padr=0.12, padt=0.45)
    BT = load('figS03d_retrieval_bouts.csv')
    edges = np.arange(0.5, 21.5, 1.0); ctr = (edges[:-1] + edges[1:]) / 2
    for role_, col in (('provider', F.PROV), ('other', F.OTH)):
        v = BT[BT.role == role_].bout_len.values
        hgt = np.histogram(v, bins=edges)[0] / max(1, len(v))
        l1.fill_between(ctr, hgt, step='mid', color=col, alpha=0.30, lw=0)
        l1.step(ctr, hgt, where='mid', color=col, lw=1.0,
                label=f'{role_.capitalize()}s' if role_ == 'provider' else 'Others')
    l1.set_xlim(0.5, 20.5)
    F.tidy(l1, 'Uninterrupted bout (trials)', 'Fraction of bouts')
    l1.yaxis.labelpad = 1.0; l1.xaxis.labelpad = 1.5; l1.tick_params(**TIGHTY)
    mp = BT[BT.role == 'provider'].bout_len.mean(); mo = BT[BT.role == 'other'].bout_len.mean()
    l1.legend(loc='upper right', fontsize=7.0, borderpad=0.1, labelspacing=0.25,
              handletextpad=0.4, frameon=False)
    F.note(l1, f'Mean {mp:.1f} versus\n{mo:.1f} trials', x=0.62, y=0.52,
           ha='center', va='top', size=7.0)

    # ---------------------------------------------------- M 사전 형질 (R2b)
    m1 = G_.ax(3, 6, cs=4, padl=1.42, padb=1.55, padr=0.12, padt=0.45)
    TR = load('fig1e_trait_table.csv')
    cols_m = [('tube_rank_pct', 'Tube\nrank'), ('oft_center_pct', 'Open\nfield'),
              ('epm_open_pct', 'EPM\nopen'), ('ymaze_alternation_pct', 'Y-maze\nalt.'),
              ('rotarod_pct', 'Rotarod')]
    rng5 = np.random.default_rng(5)
    for kk, (col, lab_) in enumerate(cols_m):
        for jj, (role_, colr) in enumerate([('other', F.OTH), ('provider', F.PROV)]):
            v = pd.to_numeric(TR.loc[TR.role == role_, col], errors='coerce').dropna().values
            xc = kk + (-0.17 if jj == 0 else 0.17)
            m1.scatter(np.full(v.size, xc) + rng5.uniform(-0.07, 0.07, v.size), v,
                       s=3.2, color=colr, lw=0)
            if v.size:
                m1.plot([xc - 0.12, xc + 0.12], [np.median(v)] * 2, color=F.INK, lw=0.9)
    m1.set_xticks(range(len(cols_m))); m1.set_xticklabels([c[1] for c in cols_m], fontsize=7.0)
    m1.set_xlim(-0.5, len(cols_m) - 0.5); m1.set_ylim(-3, 116)
    F.tidy(m1, None, 'Trait score (%)')
    m1.yaxis.labelpad = 1.0; m1.tick_params(**TIGHTY)
    m1.scatter([], [], s=6, color=F.OTH, label='Others')
    m1.scatter([], [], s=6, color=F.PROV, label='Providers')
    m1.legend(loc='upper center', fontsize=7.0, ncol=2, frameon=False,
              borderpad=0.1, handletextpad=0.3, columnspacing=1.0)

    # ---------------------------------------------------- N 활동 전후 서열
    n1 = G_.ax(3, 10, cs=2, padl=1.42, padb=1.55, padr=0.12, padt=0.45)
    DM = load('figS02n_dominance_before_after.csv')
    for _, r in DM.iterrows():
        n1.plot([0, 1], [r.dominance_before_pct, r.dominance_after_pct],
                color=F.GRID, lw=0.5, zorder=1)
    n1.scatter(np.zeros(len(DM)), DM.dominance_before_pct, s=7,
               color=[ccol(r) for r in DM.role], lw=0, zorder=3)
    n1.scatter(np.ones(len(DM)), DM.dominance_after_pct, s=7,
               color=[ccol(r) for r in DM.role], lw=0, zorder=3)
    n1.set_xticks([0, 1]); n1.set_xticklabels(['Before', 'After'], fontsize=7.0)
    n1.set_xlim(-0.45, 1.45); n1.set_ylim(-4, 118)
    F.tidy(n1, 'Test weeks', 'Dominance score (%)')
    n1.xaxis.labelpad = 1.5
    n1.yaxis.labelpad = 1.0; n1.tick_params(**TIGHTY)
    wd = stats.wilcoxon(DM.dominance_before_pct, DM.dominance_after_pct)
    F.note(n1, f'{len(DM)} mice\nP = {wd.pvalue:.2f}', x=0.5, y=0.97,
           ha='center', va='top', size=7.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 3), ('C', 0, 6),
                         ('D', 1, 0), ('E', 1, 3), ('F', 1, 6), ('G', 1, 9),
                         ('H', 2, 0), ('I', 2, 5), ('J', 2, 8),
                         ('K', 3, 0), ('L', 3, 3), ('M', 3, 6), ('N', 3, 10)):
        G_.label(r_, c_, TAGS['figS02'] + lab_)
    return F.save(fig, 'figS02', PNG, PDF)


if __name__ == '__main__':
    # fig. 1 and fig. S2 are drawn by build_fig12.py and build_s02.py
    print('saved', figS01())
