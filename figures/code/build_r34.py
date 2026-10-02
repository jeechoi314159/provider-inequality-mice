# -*- coding: utf-8 -*-
"""Fig. 3 (구 Fig. 2) 와 fig. S3–S5 생성. data/ 의 파일만 읽는다."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
SUB  = os.path.dirname(HERE)
D    = os.path.join(SUB, 'data')
PNG  = os.path.join(SUB, 'png')
PDF  = os.path.join(SUB, 'pdf')
ART  = os.path.join(SUB, 'art')
CM   = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
COH  = list('ABCDEFGH')
TIGHTY = dict(axis='y', labelsize=7.0, length=2.0, pad=1.0)
TAGS = {'fig2': '', 'figS03': '', 'figS04': '', 'figS05': ''}
CROLE = {'provider': F.PROV, 'other': F.OTH, 'others': F.OTH}



def change_label(name, effect, p):
    """범례용: 유의하면 방향 화살표, 아니면 변화 없음으로 표기."""
    ptxt = 'P < 0.001' if p < 1e-3 else f'P = {p:.2f}'
    mark = '\u2014' if p >= 0.05 else ('\u2193' if effect < 0 else '\u2191')
    return f'{name} {mark}, {ptxt}'

def strip(ax, groups, values, color, width=0.24, rng=None, s=2.2):
    rng = rng or np.random.default_rng(3)
    for i, g in enumerate(groups):
        v = np.asarray(values[i], float); v = v[np.isfinite(v)]
        ax.scatter(np.full(v.size, i) + rng.uniform(-width, width, v.size), v,
                   s=s, color=color if isinstance(color, str) else color[i], lw=0, alpha=0.75)
        ax.plot([i - width * 1.4, i + width * 1.4], [np.median(v)] * 2, color=F.INK, lw=1.0)


def traj_panel(ax, trials, fit, ylab, legend_loc='upper left'):
    """역할별 잠복기 궤적: 시행 점 + LMM 적합선. 범례에 역할별 기울기 유의성 표기."""
    rng = np.random.default_rng(5)
    for role in ('other', 'provider'):
        d = trials[trials.role == role]
        ax.scatter(d.pos + rng.uniform(-0.01, 0.01, len(d)), d.latency_s,
                   s=1.6, color=CROLE[role], lw=0, alpha=0.35)
    xs = np.linspace(0, 1, 50)
    for _, r in fit.iterrows():
        ax.plot(xs, np.exp(r.intercept + r.slope_ln * xs), color=CROLE[r.role], lw=1.3, zorder=4)
    ax.set_yscale('log')
    F.tidy(ax, 'Session position', ylab)
    ax.yaxis.labelpad = 1.0; ax.xaxis.labelpad = 1.5; ax.tick_params(**TIGHTY)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 14)
    for role, lab_ in (('provider', 'Provider'), ('other', 'Others')):
        r = fit[fit.role == role].iloc[0]
        ax.plot([], [], color=CROLE[role], lw=1.3,
                label=change_label(lab_, r.slope_ln, r.slope_p))
    leg = ax.legend(loc=legend_loc, fontsize=7.0, frameon=False, borderpad=0.1,
                    labelspacing=0.25, handlelength=0.0, handletextpad=0.0,
                    title='Change across session', alignment='left')
    leg.get_title().set_fontsize(7.0); leg.get_title().set_color(F.INK)
    for _t, _r in zip(leg.get_texts(), ('provider', 'other')):   # 조건 색을 글자에 실음
        _t.set_color(CROLE[_r])


# ======================================================================= Fig. 2
def ratio_table():
    """개체별 집단/혼자 비 (ln). 세 지표를 한 표로."""
    EN = load('fig2f_entry_solitary_vs_group.csv')
    RT = load('figS05b_retrieval_solitary_vs_group.csv')
    KE = load('fig2g_movement_solitary_vs_group.csv')
    role_of = EN.drop_duplicates('mouse').set_index('mouse').role.to_dict()
    KE = KE.assign(role=KE.mouse.map(role_of)).dropna(subset=['role'])
    out = []
    for name, d, col in (('Entry', EN, 'entry_latency_s'),
                         ('Retrieving', RT, 'retrieval_latency_s'),
                         ('Movement', KE, 'kinetic_energy')):
        w = (d.groupby(['mouse', 'role', 'condition'])[col].median()
             .unstack('condition').dropna().reset_index())
        w = w[(w.Solitary > 0) & (w.Group > 0)]
        w['measure'] = name
        w['lnratio'] = np.log(w.Group / w.Solitary)
        out.append(w[['mouse', 'role', 'measure', 'Solitary', 'Group', 'lnratio']])
    return pd.concat(out, ignore_index=True)


def s_wrap(s, n=26):
    """긴 라벨을 단어 단위로 접는다(삭제하지 않는다)."""
    out, line = [], ''
    for w in s.split(' '):
        if line and len(line) + 1 + len(w) > n:
            out.append(line); line = w
        else:
            line = (line + ' ' + w).strip()
    if line:
        out.append(line)
    return '\n'.join(out)


def per_mouse_pairs(ax, per, ylab, ylim, ptext_y, pooled=False, median_bars=True):
    """혼자 대 집단 개체별 대응 그림 (절대값)."""
    xoff = {'other': 0.0, 'provider': 2.6}
    for _, r in per.iterrows():
        x0 = xoff[r.role]
        ax.plot([x0, x0 + 1], [r.Solitary, r.Group], color=F.GRID, lw=0.5, zorder=1)
        ax.scatter([x0, x0 + 1], [r.Solitary, r.Group], s=8, color=CROLE[r.role], lw=0, zorder=3)
    if median_bars:
        for role in ('other', 'provider'):
            d = per[per.role == role]
            xs_ = [xoff[role], xoff[role] + 1]
            med = [d['Solitary'].median(), d['Group'].median()]
            ax.plot(xs_, med, color=F.INK, lw=1.4, zorder=5, marker='o', ms=4.2,
                    markerfacecolor=F.INK, markeredgecolor='white', markeredgewidth=0.7)
            for k, cnd in enumerate(('Solitary', 'Group')):     # IQR 수염
                ax.plot([xs_[k]] * 2, [d[cnd].quantile(.25), d[cnd].quantile(.75)],
                        color=F.INK, lw=0.9, zorder=4)
    ax.set_yscale('log'); ax.set_ylim(*ylim)
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(['Alone', 'Group', 'Alone', 'Group'], fontsize=7.0)
    ax.set_xlim(-0.65, 4.25)
    F.tidy(ax, None, ylab)
    ax.yaxis.labelpad = 1.0; ax.tick_params(**TIGHTY)
    for i, (role, lab_) in enumerate((('other', 'Others'), ('provider', 'Provider'))):
        d = per[per.role == role]
        ax.text(i * 2.6 + 0.5, ptext_y[0], f'{lab_} ({len(d)})', ha='center', va='bottom',
                fontsize=7.0, color=F.INK)
        if pooled:
            continue
        w = stats.wilcoxon(d.Group, d.Solitary)
        ax.text(i * 2.6 + 0.5, ptext_y[1], f'P = {w.pvalue:.2f}', ha='center', va='bottom',
                fontsize=7.0, color=F.INK)


def fig2():
    fig = F.figure(F.W3, 13.2 * CM)
    # 공통 격자: 2행 12열. 1행 A·B·C·D(각 3열), 2행 E(삽화 2열 + forest 3열)·F(3열)·G(4열)
    G_ = F.Grid(fig, 2, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[1.0, 1.12])
    rng = np.random.default_rng(11)

    # ------------------------------------------------ A 반기별 회수율 (원자료)
    a = G_.ax(0, 0, cs=3, padl=1.32, padb=1.18, padr=0.12, padt=0.42)
    L = load('figS03a_rates_by_half.csv')
    for who, col, off in (('provider', F.PROV, -0.08), ('others', F.OTH, 0.08)):
        for c in COH:
            d = L[(L.cohort == c) & (L.who == who)].set_index('half')
            if d.rate_per_min.max() <= 0:
                continue
            a.plot([0 + off, 1 + off], [max(d.loc['early', 'rate_per_min'], 1e-3),
                                        max(d.loc['late', 'rate_per_min'], 1e-3)],
                   color=col, lw=0.7, marker='o', ms=2.0, alpha=0.8)
    a.set_yscale('log')
    a.set_xticks([0, 1]); a.set_xticklabels(['Early', 'Late'], fontsize=7.0)
    a.set_xlim(-0.45, 1.45); a.set_ylim(0.1, 40)
    F.tidy(a, 'Half of session', 'Retrieval rate\n(per min at risk)')
    a.yaxis.labelpad = 1.0; a.xaxis.labelpad = 1.5; a.tick_params(**TIGHTY)
    TT = load('fig2a_early_late_test.csv').set_index('who')
    for lab_, col, key in (('Provider', F.PROV, 'provider'), ('Others', F.OTH, 'others')):
        r = TT.loc[key]
        a.plot([], [], color=col, lw=1.0,
               label=change_label(lab_, r.rate_ratio - 1.0, r.p))
    a.set_ylim(0.1, 200)
    leg = a.legend(loc='upper left', fontsize=7.0, frameon=False, borderpad=0.1,
                   labelspacing=0.25, handlelength=0.0, handletextpad=0.0,
                   title='Early vs late', alignment='left')
    leg.get_title().set_fontsize(7.0); leg.get_title().set_color(F.INK)
    for _t, _c in zip(leg.get_texts(), (F.PROV, F.OTH)):
        _t.set_color(_c)

    # ------------------------------------------------ B Shapley 분해
    b = G_.ax(0, 3, cs=3, padl=1.08, padb=1.18, padr=0.12, padt=0.42)
    S = load('fig2a_entrenchment_shapley.csv').sort_values('delta')
    y = np.arange(len(S))
    b.barh(y, S.provider_part, 0.62, color=F.OTHL, lw=0, label='Provider’s rate')
    b.barh(y, S.others_part, 0.62, left=S.provider_part, color=F.PROV, lw=0,
           label='Others’ rate')
    b.axvline(0, color=F.INK, lw=0.5)
    b.set_yticks(y); b.set_yticklabels(S.cohort, fontsize=7.0)
    F.tidy(b, 'Change in provider’s\nfraction of retrievals', 'Cohort')
    b.yaxis.labelpad = 1.0; b.xaxis.labelpad = 1.5; b.tick_params(**TIGHTY)
    b.set_xlim(-0.09, 0.62)
    frac = S.others_part.sum() / S.delta.sum()
    F.note(b, 'Others’ rate: {:.0f}% (CI 75–106)'.format(frac * 100),
           x=0.5, y=1.03, ha='center', va='bottom', size=7.0)
    b.legend(loc='lower right', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.25, handlelength=1.0, handletextpad=0.4,
             bbox_to_anchor=(1.06, -0.02))

    # ------------------------------------------------ C 진입 잠복기 궤적
    c = G_.ax(0, 6, cs=3, padl=1.18, padb=1.18, padr=0.12, padt=0.42)
    traj_panel(c, load('fig2b_entry_latency_trials.csv'), load('fig2b_entry_latency_fit.csv'),
               'Entry latency (s)', legend_loc='upper left')

    # ------------------------------------------------ D 회수 소요 시간 궤적
    d = G_.ax(0, 9, cs=3, padl=1.18, padb=1.18, padr=0.12, padt=0.42)
    traj_panel(d, load('fig2e_retrieval_latency_trials.csv'),
               load('fig2e_retrieval_latency_fit.csv'), 'Retrieving time (s)',
               legend_loc='upper left')

    # ------------------------------------------------ F 추종: 삽화 + 지연 forest
    fi = fig.add_axes(G_.rect(1, 0, cs=2, padb=0.55, padt=0.05))
    import matplotlib.image as mpimg
    fi.imshow(mpimg.imread(os.path.join(ART, 'starter_follower.png')))
    fi.set_xticks([]); fi.set_yticks([])
    for s_ in fi.spines.values():
        s_.set_visible(False)
    g = G_.ax(1, 2, cs=3, padl=0.45, padb=1.55, padr=0.62, padt=0.42)
    L2 = load('fig2d_follower_lmm.csv').iloc[::-1].reset_index(drop=True)
    for i, r in L2.iterrows():
        col = F.PROV if r.p < 0.05 else F.OTH
        g.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.0)
        g.scatter([r.slope_ln], [i], s=13, color=col, lw=0, zorder=3)
    g.axvline(0, color=F.INK, lw=0.5, ls='--')
    g.set_yticks([]); g.spines['left'].set_visible(False)
    g.set_ylim(-0.55, len(L2) - 0.15)
    F.tidy(g, 'Change in follow delay\n(ln s per session)', None)
    g.xaxis.labelpad = 1.5; g.tick_params(**TIGHTY)
    g.set_xlim(-0.40, 1.95)
    for i, r in L2.iterrows():
        g.text(-0.36, i + 0.13, s_wrap(r.stratum), fontsize=7.0, va='bottom',
               ha='left', color=F.INK, linespacing=1.15)
        g.text(r.ci_hi + 0.07, i, f'P = {r.p:.3f}' if r.p >= 1e-3 else 'P < 0.001',
               fontsize=7.0, va='center', color=F.INK)

    # ------------------------------------------------ H 시행 간 전이
    h = G_.ax(1, 5, cs=3, padl=1.58, padb=1.55, padr=1.05, padt=0.42)
    TR = load('figS03de_transitions.csv')
    states = ['retrieved', 'entered', 'stayed out']
    cmap = {'retrieved': F.PROV, 'entered': F.OTHL, 'stayed out': F.INK2}
    ylab, ypos = [], []
    k = 0
    for s_ in states:
        for role, tag in (('provider', 'P'), ('other', 'O')):
            d = TR[(TR.role == role) & (TR.from_state == s_)]
            left = 0.0
            for st in states:
                q = d[d.to_state == st]
                w = float(q.prob.iloc[0]) if len(q) else 0.0
                h.barh(k, w, 0.74, left=left, color=cmap[st], lw=0,
                       label=st.capitalize() if k == 0 else None)
                left += w
            h.text(1.03, k, 'Provider' if tag == 'P' else 'Others', ha='left',
                   va='center', fontsize=6.5, color=F.INK)
            ypos.append(k)
            k += 1
        ylab.append(s_.capitalize())
        k += 0.6
    h.set_yticks([0.5, 3.1, 5.7]); h.set_yticklabels(ylab, fontsize=7.0)
    h.set_xlim(0, 1); h.set_xticks([0, 0.5, 1])
    h.set_xticklabels(['0', '.5', '1'], fontsize=7.0)
    h.invert_yaxis()
    F.tidy(h, 'Probability of next state', None)
    h.xaxis.labelpad = 1.5; h.tick_params(**TIGHTY)
    h.tick_params(axis='y', pad=1.5)
    F.tidy(h, 'Probability of state on trial $t$ + 1', 'State on trial $t$')
    h.yaxis.labelpad = 1.0
    h.legend(loc='upper center', fontsize=7.0, frameon=False, ncol=3, borderpad=0.1,
             handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
             bbox_to_anchor=(0.5, -0.17))

    # ------------------------------------------------ H 혼자 대 집단: 얼마나 자주 대 얼마나 잘
    i_ = G_.ax(1, 8, cs=4, padl=1.38, padb=1.55, padr=0.15, padt=0.42)
    RT = ratio_table()
    SG = load('figS02a_solitary_vs_group_per_mouse.csv')
    SG = SG[SG.cohort == 'A'].copy()      # Fig. 2G 의 'Retrieved' 는 코호트 A 전용 유지
    SG['lnratio'] = np.log(SG.rate_group / SG.rate_solitary)
    meas = ['Retrieved', 'Entry', 'Retrieving']
    for mi, m in enumerate(meas):
        split = True
        for ri, role in enumerate(('other', 'provider')):
            if m == 'Retrieved':
                d = SG[SG.role == role]
            else:
                d = RT[(RT.measure == m) & (RT.role == role)]
            if not len(d):
                continue
            x0 = mi + ((ri - 0.5) * 0.44 if split else 0.0)
            jit = 0.13 if m == 'Retrieved' else 0.09
            i_.scatter(np.full(len(d), x0) + rng.uniform(-jit, jit, len(d)), d.lnratio,
                       s=7, color=CROLE[role], lw=0, alpha=0.85, zorder=3)
            if split and len(d) >= 3:
                i_.plot([x0 - 0.15, x0 + 0.15], [d.lnratio.median()] * 2, color=F.INK,
                        lw=1.1, zorder=5)
                if m != 'Retrieved':        # 'Equal contribution' 표기와 겹치지 않도록
                    i_.plot([x0, x0], [d.lnratio.quantile(.25), d.lnratio.quantile(.75)],
                            color=F.INK, lw=0.8, zorder=4)
        if m == 'Retrieved':
            pv = stats.wilcoxon(SG.rate_group, SG.rate_solitary).pvalue
            i_.text(mi, 4.1, f'{pv:.2f}', ha='center', va='bottom', fontsize=7.0, color=F.INK)
            i_.plot([mi - 0.45, mi + 0.45], [np.log(1 / 6)] * 2, color=F.INK, ls=':', lw=0.7)
            i_.text(mi - 0.45, -3.45, 'Equal contribution', ha='left', va='center',
                    fontsize=7.0, color=F.INK)
        else:
            for ri, role in enumerate(('other', 'provider')):
                d = RT[(RT.measure == m) & (RT.role == role)]
                pv = stats.wilcoxon(d.Group, d.Solitary).pvalue
                i_.text(mi + (ri - 0.5) * 0.50, 4.1, f'{pv:.2f}',
                        ha='center', va='bottom', fontsize=7.0, color=F.INK)
    i_.axhline(0, color=F.INK, ls='--', lw=0.6)
    i_.axvline(0.58, color=F.GRID, lw=0.6)
    for _x0, _x1, _lab in ((-0.62, 0.50, 'How often'), (0.66, 2.55, 'How well')):
        i_.plot([_x0, _x1], [5.35] * 2, color=F.OTH, lw=0.6, zorder=2)
        i_.text((_x0 + _x1) / 2, 5.55, _lab, ha='center', va='bottom',
                fontsize=7.0, color=F.INK)
    i_.set_xticks(range(len(meas)))
    i_.set_xticklabels(['Retrieved\n(cohort A)', 'Entry', 'Retrieving'], fontsize=7.0)
    i_.set_xlim(-0.65, 2.6); i_.set_ylim(-5.9, 7.0)
    F.tidy(i_, None, 'Group vs alone, per mouse\n(ln ratio)')
    i_.yaxis.labelpad = 1.0; i_.tick_params(**TIGHTY)
    i_.text(-0.62, 4.1, 'P', ha='left', va='bottom', fontsize=7.0, color=F.INK)
    for _c, _l in ((CROLE['provider'], 'Provider'), (CROLE['other'], 'Others')):
        i_.scatter([], [], s=7, color=_c, lw=0, label=_l)
    i_.legend(loc='lower center', fontsize=7.0, frameon=False, borderpad=0.1,
              labelspacing=0.2, handletextpad=0.3, ncol=2, columnspacing=0.9,
              bbox_to_anchor=(0.5, 1.02))

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 3), ('C', 0, 6), ('D', 0, 9),
                         ('E', 1, 0), ('F', 1, 5), ('G', 1, 8)):
        G_.label(r_, c_, TAGS['fig2'] + lab_)
    return F.save(fig, 'Fig3', PNG, PDF)


# ======================================================================= fig. S3
def figS03():
    """굳어짐의 세부: 분해의 불확실성과 미진입 시행의 이탈."""
    fig = F.figure(F.W3, 6.4 * CM)
    G_ = F.Grid(fig, 1, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95)
    rng = np.random.default_rng(4)

    # A bootstrap 분포
    a = G_.ax(0, 0, cs=6, padl=1.35, padb=1.20, padr=0.12, padt=0.45)
    B = load('figS03b_shapley_bootstrap.csv')
    v = B.others_share_of_delta * 100
    a.hist(v, bins=40, color=F.OTHL, lw=0)
    lo, hi = np.percentile(v, [2.5, 97.5])
    a.axvspan(lo, hi, color=F.PROV, alpha=0.15, lw=0)
    a.axvline(v.median(), color=F.PROV, lw=1.2)
    F.tidy(a, 'Others’ contribution to the change (%)', 'Bootstrap samples')
    a.yaxis.labelpad = 1.0; a.xaxis.labelpad = 1.5; a.tick_params(**TIGHTY)
    F.note(a, f'{v.median():.0f}%\n95% CI {lo:.0f}–{hi:.0f}', x=0.04, y=0.97,
           ha='left', va='top', size=7.0)

    # B 미진입 시행의 운동 에너지
    b = G_.ax(0, 6, cs=6, padl=1.62, padb=1.20, padr=0.12, padt=0.45)
    KE = load('figS03f_stayout_engagement.csv')
    for i, h in enumerate(('early', 'late')):
        v2 = KE[KE.half == h].ke_foraging.dropna().values
        b.scatter(np.full(v2.size, i) + rng.uniform(-0.22, 0.22, v2.size), v2, s=1.8,
                  color=F.OTHL if h == 'early' else F.INK2, lw=0, alpha=0.8)
        b.plot([i - 0.3, i + 0.3], [np.median(v2)] * 2, color=F.INK, lw=1.1)
    b.set_yscale('log')
    b.set_xticks([0, 1]); b.set_xticklabels(['First', 'Second'], fontsize=7.0)
    b.set_xlim(-0.55, 1.55)
    F.tidy(b, 'Half of session', 'Kinetic energy when\nstaying out (a.u.)')
    b.yaxis.labelpad = 1.0; b.xaxis.labelpad = 1.5; b.tick_params(**TIGHTY)
    u = stats.mannwhitneyu(KE[KE.half == 'early'].ke_foraging.dropna(),
                           KE[KE.half == 'late'].ke_foraging.dropna())
    F.note(b, 'P < 0.001' if u.pvalue < 1e-3 else f'P = {u.pvalue:.3f}',
           x=0.5, y=0.99, ha='center', va='top', size=7.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 6)):
        G_.label(r_, c_, TAGS['figS03'] + lab_)
    return F.save(fig, 'figS03', PNG, PDF)


# ======================================================================= fig. S4
def figS04():
    """경주 모형이 그림만으로 이해되도록: 모형 → 상태변수 → 적합 → 모수 회복 →
    AIC 계산 → cohort별 모형 비교."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig = F.figure(F.W3, 14.4 * CM)
    # 1행 A·B·C(각 4열) / 2행 D(4열)·E 수식+표(8열) / 3행 F(5열)·G(4열)
    G_ = F.Grid(fig, 3, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[1.0, 1.25, 1.0])
    TT = load('figS04a_trial_table.csv')
    EX = TT[TT.cohort == 'A']

    # ---------------------------------------------------- A 모형
    b = G_.ax(0, 0, cs=4, padl=1.35, padb=1.20, padr=0.12, padt=0.45)
    b.set_xlim(0, 100); b.set_ylim(-10, 64); b.set_xticks([]); b.set_yticks([])
    for s_ in b.spines.values():
        s_.set_visible(False)
    box = dict(boxstyle='round,pad=0.30,rounding_size=2.2', lw=0.6)
    b.add_patch(FancyBboxPatch((4, 46), 34, 12, fc='#fbeadf', ec=F.PROV, **box))
    b.text(21, 52, 'Provider $\\lambda_P$', ha='center', va='center', fontsize=7.0, color=F.INK)
    b.add_patch(FancyBboxPatch((62, 46), 34, 12, fc='#efeeec', ec=F.OTH, **box))
    b.text(79, 52, 'Others $\\lambda_N$', ha='center', va='center', fontsize=7.0, color=F.INK)
    b.add_patch(FancyBboxPatch((27, 22), 46, 11, fc='#ffffff', ec=F.INK, **box))
    b.text(50, 27.5, 'Retrieval on a trial', ha='center', va='center', fontsize=7.0, color=F.INK)
    b.annotate('', xy=(21, 45), xytext=(35, 34),
               arrowprops=dict(arrowstyle='->', lw=0.8, color=F.PROV))
    b.annotate('', xy=(79, 45), xytext=(65, 34),
               arrowprops=dict(arrowstyle='->', lw=0.8, color=F.OTH))
    b.text(22, 38, 'r', ha='right', va='center', fontsize=7.0, color=F.INK)
    b.text(78, 38, 'r', ha='left', va='center', fontsize=7.0, color=F.INK)
    b.annotate('', xy=(61, 52), xytext=(39, 52),
               arrowprops=dict(arrowstyle='-[, widthB=0.30, lengthB=0.12', lw=0.8, color=F.PROV))
    b.text(50, 57, '−w', ha='center', va='center', fontsize=7.0, color=F.INK)
    b.text(50, 14, r'$\log \lambda_P = a_P + r\, c_P$', ha='center', va='center',
           fontsize=7.0, color=F.INK)
    b.text(50, 1, r'$\log \lambda_N = a_N + r\, c_N - w\, c_P$', ha='center',
           va='center', fontsize=7.0, color=F.INK)

    # ---------------------------------------------------- B 상태변수
    c = G_.ax(0, 4, cs=4, padl=1.45, padb=1.20, padr=0.12, padt=0.45)
    c.plot(EX.trial, EX.cum_provider, color=F.PROV, lw=1.2, label='$c_P$, by the provider')
    c.plot(EX.trial, EX.cum_others, color=F.OTH, lw=1.2, label='$c_N$, by the others')
    F.tidy(c, 'Trial', 'Fraction of trials\nretrieved so far')
    c.yaxis.labelpad = 1.0; c.xaxis.labelpad = 1.5; c.tick_params(**TIGHTY)
    c.set_ylim(0, 0.62)
    c.legend(loc='upper left', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.25, handlelength=1.0, handletextpad=0.4)

    # ---------------------------------------------------- C 적합된 위험률
    d = G_.ax(0, 8, cs=4, padl=1.60, padb=1.20, padr=0.12, padt=0.45)
    d.plot(EX.trial, EX.lambda_provider, color=F.PROV, lw=1.2)
    d.plot(EX.trial, EX.lambda_others, color=F.OTH, lw=1.2)
    d.set_yscale('log'); d.set_ylim(0.1, 40)
    for w, col, y0 in (('provider', F.PROV, 22), ('others', F.OTH, 15)):
        t = EX[EX.winner == w].trial
        d.plot(t, np.full(len(t), y0), marker='|', ms=2.6, lw=0, color=col)
    F.tidy(d, 'Trial', 'Fitted rate (per min at risk)')
    d.yaxis.labelpad = 1.0; d.xaxis.labelpad = 1.5; d.tick_params(**TIGHTY)

    # ---------------------------------------------------- D 모수 회복
    e = G_.ax(1, 0, cs=6, padl=1.55, padb=1.20, padr=0.12, padt=0.45)
    RECV = load('figS04b_parameter_recovery.csv')
    e.plot([-1.6, 3.2], [-1.6, 3.2], color=F.GRID, lw=0.6)
    e.scatter(RECV.alpha_true, RECV.alpha_fit, s=7, color=F.PROV, lw=0, label='Reinforcement (r)')
    e.scatter(RECV.beta_true, RECV.beta_fit, s=7, color=F.OTH, lw=0, label='Withdrawal (w)')
    F.tidy(e, 'Simulated value', 'Fitted value')
    e.yaxis.labelpad = 1.0; e.xaxis.labelpad = 1.5; e.tick_params(**TIGHTY)
    e.legend(loc='upper left', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.25, handletextpad=0.3)
    F.note(e, 'Recovery ρ = {:.2f} and {:.2f}'.format(
        RECV.alpha_true.corr(RECV.alpha_fit), RECV.beta_true.corr(RECV.beta_fit)),
        x=0.97, y=0.03, ha='right', va='bottom', size=7.0)

    # ---------------------------------------------------- E AIC 를 어떻게 얻는가
    g = G_.ax(1, 6, cs=6, padl=0.20, padb=1.20, padr=0.20, padt=0.45)
    g.set_xlim(0, 100); g.set_ylim(0, 100); g.set_xticks([]); g.set_yticks([])
    for s_ in g.spines.values():
        s_.set_visible(False)
    g.text(50, 97, r'$\ell_c = \sum_t\, [\, \log \lambda_{\mathrm{winner}}(t)'
                   r' - (\lambda_P(t) + \lambda_N(t))\, E_t \,]$',
           ha='center', va='top', fontsize=7.0, color=F.INK)
    g.text(50, 78, r'$\mathrm{AIC} = \sum_c\, (2k - 2\ell_c)$,'
                   r'   $\Delta\mathrm{AIC} = \mathrm{AIC} - \mathrm{AIC}_{\min}$',
           ha='center', va='top', fontsize=7.0, color=F.INK)
    MC = load('figS04d_model_comparison.csv').groupby('model').agg(
        k=('k', 'first'), ll=('logLik', 'sum'), aic=('aic', 'sum'))
    MC['d'] = MC.aic - MC.aic.min()
    rows = [('Constant', 'Constant'), ('Reinforcement', 'r only'),
            ('Withdrawal', 'w only'), ('Both', 'r + w')]
    xs = (3, 34, 58, 79, 99)
    hdr = ('Model', '$k$', '$\\Sigma\\,\\ell_c$', 'AIC', '$\\Delta$AIC')
    ytop = 54
    for x, h in zip(xs, hdr):
        g.text(x, ytop, h, ha='left' if x == 3 else 'right', va='center',
               fontsize=7.0, color=F.INK)
    g.plot([1, 100], [ytop - 6, ytop - 6], color=F.INK, lw=0.5)
    for i, (key, lab_) in enumerate(rows):
        y = ytop - 15 - i * 11
        r = MC.loc[key]
        best = r.d < 0.01
        if best:
            g.add_patch(Rectangle((1, y - 5), 99, 10, fc='#f7ece4', ec='none', zorder=0))
        vals = (lab_, f'{int(r.k)}', f'{r.ll:.0f}'.replace('-', '\u2212'), f'{r.aic:.0f}',
                f'{r.d:.1f}')
        for x, v, ha in zip(xs, vals, ('left', 'right', 'right', 'right', 'right')):
            g.text(x, y, v, ha=ha, va='center', fontsize=7.0, color=F.INK,
                   fontweight='bold' if best else 'normal')
    g.plot([1, 100], [ytop - 15 - 3 * 11 - 6, ytop - 15 - 3 * 11 - 6], color=F.INK, lw=0.5)

    # ---------------------------------------------------- F cohort별 모형 비교
    f = G_.ax(2, 0, cs=5, padl=1.45, padb=1.20, padr=0.12, padt=0.45)
    MC2 = load('figS04d_model_comparison.csv').pivot_table(index='cohort', columns='model',
                                                           values='aic')
    D = MC2.sub(MC2.min(axis=1), axis=0)
    order_m = ['Withdrawal', 'Both', 'Reinforcement', 'Constant']
    mk = {'Withdrawal': ('o', F.PROV), 'Both': ('s', F.OTH),
          'Reinforcement': ('^', F.OTHL), 'Constant': ('D', F.INK2)}
    CAP = 20.0
    off = dict(zip(order_m, (-0.21, -0.07, 0.07, 0.21)))
    for i, coh in enumerate(D.index[::-1]):
        for m in order_m:
            v = float(D.loc[coh, m]); s_ = mk[m]
            f.plot([min(v, CAP)], [i + off[m]], marker=s_[0], ms=2.8, color=s_[1], lw=0,
                   clip_on=False, markeredgecolor='none' if m != 'Constant' else F.INK,
                   markerfacecolor=s_[1] if m != 'Constant' else 'none',
                   markeredgewidth=0.5)
            if v > CAP:
                f.annotate('', xy=(CAP + 1.6, i + off[m]), xytext=(CAP, i + off[m]),
                           arrowprops=dict(arrowstyle='->', lw=0.5, color=s_[1]))
    f.axvline(0, color=F.GRID, lw=0.6)
    f.set_yticks(range(len(D))); f.set_yticklabels(D.index[::-1], fontsize=7.0)
    f.set_xlim(-1.2, CAP + 3); f.set_ylim(-0.7, len(D) - 0.3)
    F.tidy(f, 'ΔAIC within cohort', 'Cohort')
    f.yaxis.labelpad = 1.0; f.xaxis.labelpad = 1.5; f.tick_params(**TIGHTY)
    for m in order_m:
        s_ = mk[m]
        f.plot([], [], marker=s_[0], ms=2.8, lw=0, color=s_[1], label=m,
               markerfacecolor=s_[1] if m != 'Constant' else 'none',
               markeredgecolor='none' if m != 'Constant' else F.INK, markeredgewidth=0.5)
    f.legend(loc='lower right', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.2, handletextpad=0.25, handlelength=1.0)

    # ---------------------------------------------------- G cohort별 적합 계수
    gg = G_.ax(2, 5, cs=4, padl=1.55, padb=1.20, padr=0.12, padt=0.45)
    FP = load('figS04c_fitted_parameters.csv')
    rng4 = np.random.default_rng(11)
    for i_, (col_, colr) in enumerate((('alpha', F.PROV), ('beta', F.OTH))):
        v = FP[col_].values
        gg.scatter(np.full(v.size, i_) + rng4.uniform(-0.18, 0.18, v.size), v,
                   s=11, color=colr, lw=0)
        gg.plot([i_ - 0.3, i_ + 0.3], [np.median(v)] * 2, color=F.INK, lw=1.1)
    gg.axhline(0, color=F.INK, ls='--', lw=0.6)
    gg.set_xticks([0, 1]); gg.set_xticklabels(['r', 'w'], fontsize=7.0)
    gg.set_xlim(-0.55, 1.55); gg.set_ylim(-1.6, 6.6)
    F.tidy(gg, None, 'Fitted parameter (per session)')
    gg.yaxis.labelpad = 1.0; gg.tick_params(**TIGHTY)
    F.note(gg, 'w > 0 in {}/{} cohorts'.format(int((FP.beta > 0).sum()), len(FP)),
           x=0.0, y=1.02, ha='left', va='bottom', size=7.0)
    MC2 = load('figS04d_model_comparison.csv').groupby('model').aic.sum()
    MC2 = MC2 - MC2.min()
    F.note(gg, 'Without r, AIC {:+.0f}\nWithout w, AIC {:+.0f}'.format(
        -MC2['Both'], MC2['Reinforcement'] - MC2['Both']).replace('-', '−'),
        x=0.02, y=0.97, ha='left', va='top', size=7.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 4), ('C', 0, 8),
                         ('D', 1, 0), ('E', 1, 6), ('F', 2, 0), ('G', 2, 5)):
        G_.label(r_, c_, TAGS['figS04'] + lab_)
    return F.save(fig, 'figS04', PNG, PDF)


# ======================================================================= fig. S5
def figS05():
    """보류의 근거: 같은 개체를 혼자와 집단에서 잰 절대값(A–C)과 두 가지 대조(D, E)."""
    fig = F.figure(F.W3, 9.6 * CM)
    G_ = F.Grid(fig, 2, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95)
    rng = np.random.default_rng(8)
    EN = load('fig2f_entry_solitary_vs_group.csv')

    # A 개체별 진입 잠복기
    a = G_.ax(0, 0, cs=4, padl=1.60, padb=1.05, padr=0.12, padt=0.85)
    per = (EN.groupby(['mouse', 'role', 'condition']).entry_latency_s.median()
           .unstack('condition').dropna().reset_index())
    per_mouse_pairs(a, per, 'Entry latency, per mouse (s)', (0.2, 900), (620, 180))

    # B 개체별 회수 소요 시간
    b = G_.ax(0, 4, cs=4, padl=1.60, padb=1.05, padr=0.12, padt=0.85)
    RT = load('figS05b_retrieval_solitary_vs_group.csv')
    pr = (RT.groupby(['mouse', 'role', 'condition']).retrieval_latency_s.median()
          .unstack('condition').dropna().reset_index())
    per_mouse_pairs(b, pr, 'Retrieving time, per mouse (s)', (2.2, 900), (600, 250))

    # C 개체별 이동량 (가속도계를 단 개체)
    c = G_.ax(0, 8, cs=4, padl=1.60, padb=1.05, padr=0.12, padt=0.85)
    KE = load('fig2g_movement_solitary_vs_group.csv')
    role_of = EN.drop_duplicates('mouse').set_index('mouse').role.to_dict()
    KE = KE.assign(role=KE.mouse.map(role_of)).dropna(subset=['role'])
    pk = (KE.groupby(['mouse', 'role', 'condition']).kinetic_energy.median()
          .unstack('condition').dropna().reset_index())
    per_mouse_pairs(c, pk, 'Kinetic energy, per mouse (a.u.)', (2.5e-4, 6e-3),
                    (3.3e-3, 2.2e-3), pooled=True, median_bars=False)
    # 개체마다 혼자 시행 대 집단 시행을 직접 비교하고 BH 보정 후 유의한 쌍만 표시
    pv, mice = [], []
    for m, gm in KE.groupby('mouse'):
        a_ = gm[gm.condition == 'Solitary'].kinetic_energy
        b_ = gm[gm.condition == 'Group'].kinetic_energy
        pv.append(stats.mannwhitneyu(a_, b_).pvalue); mice.append(m)
    pv = np.asarray(pv); o = np.argsort(pv); qv = np.empty_like(pv)
    qv[o] = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    qmap = dict(zip(mice, qv))
    xoff_c = {'other': 0.0, 'provider': 2.6}
    for _, r_ in pk.iterrows():
        if qmap.get(r_.mouse, 1.0) < 0.05:
            c.text(xoff_c[r_.role] + 1.18, np.sqrt(r_.Solitary * r_.Group), '*',
                   ha='left', va='center', fontsize=8.0, color=F.INK)
    F.note(c, '*, q < 0.05', x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    # D 시행 합산 진입 잠복기 분포 (합산의 함정)
    d = G_.ax(1, 0, cs=4, padl=1.60, padb=1.05, padr=0.12, padt=0.85)
    groups, vals, cols = [], [], []
    for role in ('other', 'provider'):
        for cond in ('Solitary', 'Group'):
            groups.append(cond)
            vals.append(EN[(EN.role == role) & (EN.condition == cond)].entry_latency_s.values)
            cols.append(CROLE[role] if cond == 'Group' else F.OTHL)
    strip(d, groups, vals, cols, width=0.22, s=1.4)
    d.set_yscale('log'); d.set_ylim(0.15, 9000)
    d.set_xticks(range(4)); d.set_xticklabels(['Alone', 'Group', 'Alone', 'Group'], fontsize=7.0)
    d.set_xlim(-0.6, 3.6)
    F.tidy(d, None, 'Entry latency, per trial (s)')
    d.yaxis.labelpad = 1.0; d.tick_params(**TIGHTY)
    for i, (role, lab_) in enumerate((('other', 'Others'), ('provider', 'Provider'))):
        s_ = EN[(EN.role == role) & (EN.condition == 'Solitary')].entry_latency_s
        g_ = EN[(EN.role == role) & (EN.condition == 'Group')].entry_latency_s
        u = stats.mannwhitneyu(s_, g_)
        d.text(i * 2 + 0.5, 4200, lab_, ha='center', va='bottom', fontsize=7.0, color=F.INK)
        d.text(i * 2 + 0.5, 1400, 'P < 0.001' if u.pvalue < 1e-3 else f'P = {u.pvalue:.2f}',
               ha='center', va='bottom', fontsize=7.0, color=F.INK)

    # E 혼자 시행 동안의 추세 (피로·포만 대조)
    e = G_.ax(1, 4, cs=4, padl=1.60, padb=1.05, padr=0.12, padt=0.85)
    SL = load('fig1b_solitary_latency.csv')
    TRD = load('figS05c_solitary_trend.csv').set_index('measure')
    for col_, colr, lab_ in (('entry_latency_s', F.OTH, 'Entry'),
                             ('total_latency_s', F.PROV, 'Retrieval')):
        e.scatter(SL.trial + rng.uniform(-0.18, 0.18, len(SL)), SL[col_], s=2.2,
                  color=colr, lw=0, alpha=0.6)
        m = TRD.loc[lab_]
        xs = np.linspace(1, SL.trial.max(), 20)
        inter = np.log(SL[col_].replace(0, np.nan).dropna()).mean() - m.slope_ln_per_trial * SL.trial.mean()
        e.plot(xs, np.exp(inter + m.slope_ln_per_trial * xs), color=colr, lw=1.2, label=lab_)
    e.set_yscale('log'); e.set_ylim(0.05, 20000)
    F.tidy(e, 'Solitary trial', 'Latency (s)')
    e.yaxis.labelpad = 1.0; e.xaxis.labelpad = 1.5; e.tick_params(**TIGHTY)
    e.legend(loc='upper right', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.25, handlelength=1.0, handletextpad=0.4)
    F.note(e, 'Entry P = {:.2f}\nRetrieval P = {:.3f}'.format(TRD.loc['Entry', 'p'],
                                                               TRD.loc['Retrieval', 'p']),
           x=0.98, y=0.03, ha='right', va='bottom', size=7.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 4), ('C', 0, 8),
                         ('D', 1, 0), ('E', 1, 4)):
        G_.label(r_, c_, TAGS['figS05'] + lab_)
    return F.save(fig, 'figS05', PNG, PDF)


if __name__ == '__main__':
    for fn in (fig2, figS03, figS04, figS05):
        print('saved', fn())
