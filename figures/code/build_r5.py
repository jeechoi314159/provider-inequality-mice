# -*- coding: utf-8 -*-
"""Fig. 4 (구 Fig. 3) 와 fig. S6 생성. data/ 의 파일만 읽는다."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
SUB  = os.path.dirname(HERE)
D    = os.path.join(SUB, 'data')
PNG  = os.path.join(SUB, 'png')
PDF  = os.path.join(SUB, 'pdf')
CM   = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
TIGHTY = dict(axis='y', labelsize=7.0, length=2.0, pad=1.0)
TAGS = {'fig3': '', 'figS06': ''}

SHARE = LinearSegmentedColormap.from_list('share', ['#ffffff', '#f7d9c4', F.PROV])
PORDER = ['Group', 'Provider out', '+ successor out', 'Split',
          'Rejoined, provider out', 'Successor back', 'All back']
PSHORT = {'Group': 'Group', 'Provider out': 'Provider\nout',
          '+ successor out': '+ successor\nout', 'Split': 'Split',
          'Rejoined, provider out': 'Rejoined,\nprovider\nout',
          'Successor back': 'Successor\nback', 'All back': 'All\nback'}


# ----------------------------------------------------------------- 박스 배열
SPLITN = {'A': 3, 'B': 3, 'C': 2}   # 분리 단계에서 제공자가 속한 하위집단의 크기


def box_array(ax, coh, G, row_fs=7.0, day_step=2, show_daylab=True):
    """행 = 개체(회수 순위), 열 = 실험일, 칸 음영 = 그날 회수 중 그 개체의 비율."""
    g = G[G.cohort == coh].copy()
    g['subgroup'] = g.subgroup.fillna('')
    days = sorted(g.day.unique())
    di = {d: i for i, d in enumerate(days)}
    order = (g[['mouse', 'rank']].drop_duplicates().sort_values('rank'))
    mi = {m: i for i, m in enumerate(order.mouse)}
    nR = len(mi)
    prov = g.loc[g.is_provider == 1, 'mouse'].iloc[0]

    for _, r in g.iterrows():
        x, y = di[r.day], nR - 1 - mi[r.mouse]
        if r.present == 0:
            ax.add_patch(Rectangle((x + .08, y + .08), .84, .84, fc='#eae8e4',
                                   ec=F.OTH, lw=0.35, zorder=1))
            ax.plot([x + .14, x + .86], [y + .14, y + .86], color='#5f5d58', lw=0.6, zorder=2)
            ax.plot([x + .14, x + .86], [y + .86, y + .14], color='#5f5d58', lw=0.6, zorder=2)
        else:
            v = 0.0 if not np.isfinite(r.share) else float(r.share)
            ax.add_patch(Rectangle((x + .08, y + .08), .84, .84, fc=SHARE(v),
                                   ec='#b9b6b0', lw=0.3, zorder=1))
    ph_by_day = g.drop_duplicates('day').set_index('day').phase.to_dict()
    bounds, cur, start = [], ph_by_day[days[0]], 0
    for i, d in enumerate(days):
        if ph_by_day[d] != cur:
            bounds.append((cur, start, i - 1)); cur, start = ph_by_day[d], i
    bounds.append((cur, start, len(days) - 1))
    # 단계 구분선과 구간 막대
    anchors, texts = [], []
    for _k, (ph, i0, i1) in enumerate(bounds):
        if _k % 2 == 1:                       # 단계 교대 배경(칸 사이 여백에만 드러남)
            ax.add_patch(Rectangle((i0, -0.1), i1 - i0 + 1, nR + 0.2,
                                   fc='#f4f2ef', ec='none', zorder=0))
        if i0 > 0:
            ax.plot([i0, i0], [-0.15, nR + 0.15], color=F.INK, lw=0.6, zorder=5)
        ax.plot([i0 + .12, i1 + .88], [nR + 0.45] * 2, color=F.INK, lw=0.6,
                clip_on=False, zorder=5)
        cx = (i0 + i1 + 1) / 2
        anchors.append(cx)
        texts.append(ax.text(cx, nR + 1.05, PSHORT[ph], ha='center', va='bottom',
                             fontsize=7.0, color=F.INK, linespacing=1.15,
                             clip_on=False, zorder=6))
    sp = g[(g.phase == 'Split') & (g.subgroup.isin(['with provider', 'without provider']))]
    groups, i0, i1 = None, None, None
    if len(sp):
        groups = {sub: list(sp[sp.subgroup == sub].mouse.unique())
                  for sub in ('with provider', 'without provider')}
        i0 = min(di[d] for d in sp.day.unique()); i1 = max(di[d] for d in sp.day.unique())
    else:
        # 하위집단이 자료에 따로 표시되지 않은 코호트(C)는 기준기 기여 순위로 나눔
        spl = g[g.phase == 'Split']
        n_with = SPLITN.get(coh)
        if len(spl) and n_with:
            rk = spl[['mouse', 'rank']].drop_duplicates().sort_values('rank')
            groups = {'with provider': list(rk.mouse[:n_with]),
                      'without provider': list(rk.mouse[n_with:])}
            i0 = min(di[d] for d in spl.day.unique()); i1 = max(di[d] for d in spl.day.unique())
    if groups:
        for sub, mice_ in groups.items():
            rs = sorted(nR - 1 - mi[m] for m in mice_ if m in mi)
            if not rs:
                continue
            ax.add_patch(Rectangle((i0 + .02, min(rs) - .02), i1 - i0 + .96, len(rs) + .04,
                                   fc='none', ec=F.INK, lw=0.7, zorder=6))
    ax.set_xlim(-0.05, len(days) + 0.05); ax.set_ylim(-0.1, nR + 0.1)
    ax.set_aspect('equal', adjustable='box', anchor='NW')     # 칸을 정사각형·좌측 정렬
    ax.set_yticks([nR - 0.5 - i for i in range(nR)])
    ax.set_yticklabels([(m + ' (provider)' if m == prov else m) for m in order.mouse],
                       fontsize=row_fs)
    xt = [i for i in range(len(days)) if (i + 1) % day_step == 0 or i == 0]
    ax.set_xticks([i + 0.5 for i in xt])
    ax.set_xticklabels([str(int(days[i])) for i in xt], fontsize=7.0)
    ax.tick_params(axis='x', length=1.6, pad=1.0)
    ax.tick_params(axis='y', length=0, pad=1.5)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    # 라벨은 모두 수평. 폭을 재어 한 높이에 늘어놓되, 폭이 모자라면 두 단으로 나눈다.
    fig_ = ax.figure
    fig_.canvas.draw()
    rend = fig_.canvas.get_renderer()
    inv = ax.transData.inverted()
    half, thi = [], []
    for t in texts:
        bb = t.get_window_extent(rend)
        x0, y0 = inv.transform((bb.x0, bb.y0))
        x1, y1 = inv.transform((bb.x1, bb.y1))
        half.append((x1 - x0) / 2.0)
        thi.append(abs(y1 - y0))
    GAP, LO, HI = 0.45, -0.3, len(days) + 0.3

    def _pack(idx):
        pos = [anchors[i] for i in idx]
        for k in range(len(idx)):
            lo = (LO + half[idx[k]] if k == 0 else
                  pos[k - 1] + half[idx[k - 1]] + GAP + half[idx[k]])
            pos[k] = max(pos[k], lo)
        for k in range(len(idx) - 1, -1, -1):
            hi = (HI - half[idx[k]] if k == len(idx) - 1 else
                  pos[k + 1] - half[idx[k + 1]] - GAP - half[idx[k]])
            pos[k] = min(pos[k], hi)
        return pos

    def _fits(idx, pos):
        for k in range(len(idx) - 1):
            if pos[k + 1] - half[idx[k + 1]] < pos[k] + half[idx[k]] + 0.40:
                return False
        return True

    all_i = list(range(len(texts)))
    pos_all = _pack(all_i)
    if _fits(all_i, pos_all):
        levels = [(all_i, pos_all, 0.0)]
    else:                                   # 한 단에 안 들어가면 두 단(둘 다 수평)
        dy = max(thi) * 1.12
        ev, od = all_i[0::2], all_i[1::2]
        levels = [(ev, _pack(ev), 0.0), (od, _pack(od), dy)]
    for idx, pos, off in levels:
        for k, i in enumerate(idx):
            texts[i].set_x(pos[k])
            texts[i].set_y(nR + 1.05 + off)
            ax.plot([anchors[i], pos[k]], [nR + 0.52, nR + 1.0 + off],
                    color=F.GRID, lw=0.35, clip_on=False, zorder=4)
    if show_daylab:
        ax.set_xlabel('Day', fontsize=8.0, labelpad=1.5)
    return nR


def share_bar(fig, rect, label='Fraction of retrievals'):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=SHARE), cax=cax,
                      orientation='horizontal', ticks=[0, 0.5, 1])
    cb.outline.set_linewidth(0.4)
    cax.set_xticklabels(['0', '.5', '1'], fontsize=7.0)
    cax.tick_params(length=1.6, pad=1.0)
    cax.set_xlabel(label, fontsize=7.0, labelpad=1.5)
    return cax


def absent_key(fig, x, y, size=0.013, text='Absent from the group', stack=False):
    ax = fig.add_axes([x, y, size, size * 1.55])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((0, 0), 1, 1, fc='#eae8e4', ec=F.OTH, lw=0.35))
    ax.plot([.08, .92], [.08, .92], color='#5f5d58', lw=0.6)
    ax.plot([.08, .92], [.92, .08], color='#5f5d58', lw=0.6)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    if stack:
        fig.text(x, y - size * 0.7, text, fontsize=7.0, va='top', ha='left',
                 color=F.INK, linespacing=1.2)
    else:
        fig.text(x + size * 1.6, y + size * 0.75, text, fontsize=7.0,
                 va='center', ha='left', color=F.INK)
    return ax


# ======================================================================= Fig. 3
def fig3():
    fig = F.figure(F.W3, 11.1 * CM)
    # 공통 격자 2행 12열. 1행 A 격자(10열) + 범례(2열), 2행 B·C·D 각 4열
    G_ = F.Grid(fig, 2, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[3.88, 5.62])
    G = load('fig3a_daily_grid.csv')
    MK = {'A': 'o', 'B': 's', 'C': '^'}

    # ---------------------------------------------------- A cohort A 일별 격자
    a = G_.ax(0, 0, cs=10, padl=1.58, padb=0.95, padr=0.10, padt=1.00)
    box_array(a, 'A', G, day_step=3)
    _lx = G_.cell_cm(0, 10, cs=2)[0]
    share_bar(fig, [G_.fx(_lx + 0.05), G_.cell(0, 10)[1] + G_.fy(2.55),
                    G_.fx(1.35), G_.fy(0.16)], label='Fraction of\nretrievals')
    absent_key(fig, G_.fx(_lx + 0.05), G_.cell(0, 10)[1] + G_.fy(1.05),
               size=0.013, text='Absent from\nthe group', stack=True)

    # ------------------------------ B 이전 순위가 다음 제공자를 예측함
    b = G_.ax(1, 0, cs=4, padl=1.45, padb=1.48, padr=0.12, padt=1.35)
    RK = load('fig3b_rank_vs_contribution.csv')
    RK['subgroup'] = RK.subgroup.fillna('')
    PROV_OF = {c: G[(G.cohort == c) & (G.is_provider == 1)].mouse.iloc[0] for c in 'ABC'}
    for (coh, ph, sub), g in RK.groupby(['cohort', 'phase', 'subgroup']):
        g = g.sort_values('avail_rank')
        b.plot(g.avail_rank, g.delivery_share, color=F.GRID, lw=0.5, zorder=1)
        for _, r in g.iterrows():
            orig = r.mouse == PROV_OF[r.cohort]
            col = F.PROV if orig else F.OTH
            if r.is_top:
                b.plot([r.avail_rank], [r.delivery_share], marker='o', ms=3.2, lw=0,
                       markerfacecolor=col, markeredgecolor=F.INK, markeredgewidth=0.5, zorder=4)
            else:
                b.plot([r.avail_rank], [r.delivery_share], marker='o', ms=2.2, lw=0,
                       markerfacecolor='white', markeredgecolor=col, markeredgewidth=0.6, zorder=3)
    b.set_xticks(range(1, 7)); b.set_xlim(0.6, 6.4); b.set_ylim(-0.03, 1.06)
    F.tidy(b, 'Rank among the mice available\n(earlier work rate)',
           'Fraction of the episode’s\nretrievals')
    b.yaxis.labelpad = 1.0; b.xaxis.labelpad = 1.5; b.tick_params(**TIGHTY)
    hs = [mlines.Line2D([], [], marker='o', ms=3.2, lw=0, markerfacecolor=F.OTH,
                        markeredgecolor=F.INK, markeredgewidth=0.5,
                        label='Main retriever'),
          mlines.Line2D([], [], marker='o', ms=2.2, lw=0, markerfacecolor='white',
                        markeredgecolor=F.OTH, markeredgewidth=0.6, label='Others available'),
          mlines.Line2D([], [], marker='o', ms=3.2, lw=0, markerfacecolor=F.PROV,
                        markeredgecolor=F.INK, markeredgewidth=0.5,
                        label='Original provider')]
    b.legend(handles=hs, loc='lower left', fontsize=7.0, frameon=False, borderpad=0.1,
             labelspacing=0.22, handletextpad=0.3, ncol=2, columnspacing=0.8,
             bbox_to_anchor=(-0.02, 1.01))

    bi = fig.add_axes(G_.rect(1, 0, cs=4, padl=3.70, padb=2.45, padr=0.12, padt=1.50))
    IN = load('fig3b_top_rank_prediction.csv')
    vals = [IN.observed.mean(), IN['Baseline rates, renormalised'].mean(),
            IN['Equal among available'].mean()]
    bi.bar(range(3), vals, color=[F.INK2, F.OTH, F.OTHL], width=0.66, lw=0)
    for k, v in enumerate(vals):
        bi.text(k, v + 0.05, '{:.2f}'.format(v), ha='center', va='bottom', fontsize=7.0,
                color=F.INK)
    bi.set_xticks(range(3))
    bi.set_xticklabels(['Obs.', 'Base', 'Equal'], fontsize=7.0)
    bi.set_ylim(0, 1.35); bi.set_yticks([])
    bi.tick_params(axis='x', length=1.2, pad=0.8)
    for sp_ in ('top', 'right', 'left'):
        bi.spines[sp_].set_visible(False)
    F.note(bi, 'Rank 1 took most', x=0.02, y=0.99, ha='left', va='top', size=7.0)

    # ------------------------------ C 같은 개체가 자리를 맡았다가 내놓음 (cohort별)
    c_ = G_.ax(1, 4, cs=4, padl=1.45, padb=1.48, padr=0.12, padt=1.35)
    CD = load('fig3c_three_conditions.csv')
    JIT = {'A': -0.09, 'B': 0.0, 'C': 0.09}
    for _, r in CD.iterrows():
        ys = [r.present, r.absent, r.returned]
        xs = [k_ + JIT[r.cohort] for k_ in (0, 1, 2)]
        edge = F.INK if r.highest_available else F.OTH
        c_.plot(xs, ys, color=F.GRID, lw=0.8, zorder=2)
        for x_, y_ in zip(xs, ys):
            c_.plot([x_], [y_], marker=MK[r.cohort], ms=4.2, lw=0, markerfacecolor=SHARE(y_),
                    markeredgecolor=edge, markeredgewidth=0.55, zorder=3)
    c_.set_xticks([0, 1, 2])
    c_.set_xticklabels(['Present', 'Absent', 'Back'], fontsize=7.0)
    c_.set_xlim(-0.45, 2.45); c_.set_ylim(-0.04, 1.10)
    F.tidy(c_, 'Higher-ranked mouse', 'Retrievals per trial,\nreplacement mouse')
    c_.yaxis.labelpad = 1.0; c_.xaxis.labelpad = 1.5; c_.tick_params(**TIGHTY)
    hs3 = [mlines.Line2D([], [], marker=MK[c_h], ms=4.2, lw=0, markerfacecolor='white',
                         markeredgecolor=F.INK, markeredgewidth=0.55, label='Cohort ' + c_h)
           for c_h in 'ABC']
    c_.legend(handles=hs3, loc='upper left', fontsize=7.0, frameon=False, borderpad=0.1,
              labelspacing=0.2, handletextpad=0.3, bbox_to_anchor=(-0.02, 1.02))


    # ------------------------------ D 구성 변화 전후의 시행 정렬 궤적
    EA = load('fig3d_event_aligned.csv')
    WIN = 15
    XR = 45
    TINT = {('prov', 'A'): '#c4551a', ('prov', 'B'): F.PROV, ('prov', 'C'): '#f2a271',
            ('oth', 'A'): '#5f5d58', ('oth', 'B'): F.OTH, ('oth', 'C'): '#c2c0ba'}
    _dc = G_.cell_cm(1, 8, cs=4)
    _dxc = _dc[0] + _dc[2] / 2.0
    for di, (ev, lab) in enumerate((('removal', 'Removal'),
                                    ('reunion', 'Reunion'))):
        ax = fig.add_axes(G_.rect(1, 8, cs=4, padl=1.42 + di * 2.42,
                                  padb=1.48, padr=0.12 + (1 - di) * 2.42, padt=1.35), label='<sub>')
        for (coh, role), g in EA[EA.event == ev].groupby(['cohort', 'role']):
            g = g.sort_values('rel_trial')
            y = g.retrieved.rolling(WIN, center=True, min_periods=1).mean().values
            x = g.rel_trial.values
            if role.startswith('Returning'):
                # 2026-10-02 저자 결정: 원 제공자는 부재 구간(제거 후 · 재결합 전)에서 회수할 수 없으므로 그 구간 값만 0.
                # 나머지(대체 개체, 창, 이동 평균)는 원래대로 둠.
                y = np.where((x >= 0) if ev == 'removal' else (x < 0), 0.0, y)
            col = TINT[('prov' if role.startswith('Returning') else 'oth', coh)]
            ax.plot(x, y, color=col, lw=1.0, zorder=2)
            # 시행이 더 없는 구간은 마지막 값으로 수평 연장
            if x[-1] < XR:
                ax.plot([x[-1], XR], [y[-1]] * 2, color=col, lw=1.0, zorder=2)
            if x[0] > -XR:
                ax.plot([-XR, x[0]], [y[0]] * 2, color=col, lw=1.0, zorder=2)
        ax.axvline(0, color=F.INK, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(-XR, XR); ax.set_ylim(-0.04, 1.06)
        F.tidy(ax, None, 'Probability of retrieving' if di == 0 else None)
        ax.yaxis.labelpad = 1.0; ax.xaxis.labelpad = 1.5; ax.tick_params(**TIGHTY)
        if di == 1:
            ax.set_yticklabels([])
        # 조건 레일: 원래 제공자가 집단에 있는 구간과 빠진 구간
        _st = (('in', 'out') if ev == 'removal' else ('out', 'in'))
        for _sx, _ex, _s in ((-XR, 0, _st[0]), (0, XR, _st[1])):
            ax.add_patch(Rectangle((_sx, 1.12), _ex - _sx, 0.06,
                                   fc=('#fbe6d8' if _s == 'in' else '#eae8e4'),
                                   ec=F.OTH, lw=0.4, clip_on=False, zorder=6))
        ax.text(0.0, 1.26, 'Provider', ha='right', va='bottom', fontsize=7.0,
                color=F.PROV, clip_on=False)
        ax.text(0.0, 1.26, ': {} \u2192 {}'.format(*_st), ha='left', va='bottom',
                fontsize=7.0, color=F.INK, clip_on=False)
        F.note(ax, lab, x=0.5, y=1.34, ha='center', va='bottom', size=7.0)
        ax.set_xticks([-40, -20, 0, 20, 40])
    fig.text(G_.fx(_dxc), G_.cell(1, 8)[1] + G_.fy(0.52), 'Trial relative to the change in membership',
             ha='center', va='bottom', fontsize=7.0, color=F.INK)
    hs2 = [mlines.Line2D([], [], color=TINT[('oth', c)], lw=1.2, label='Cohort ' + c)
           for c in 'ABC']
    fig.legend(handles=hs2, loc='lower center', bbox_to_anchor=(G_.fx(_dxc), G_.cell(1, 8)[1] + G_.fy(0.05)), ncol=3,
               fontsize=7.0, frameon=False, handlelength=1.0, handletextpad=0.35,
               columnspacing=0.9)

    for lab, r_, c_i in (('A', 0, 0), ('B', 1, 0), ('C', 1, 4), ('D', 1, 8)):
        G_.label(r_, c_i, TAGS['fig3'] + lab)
    return F.save(fig, 'Fig4', PNG, PDF)


# ===================================================================== fig. S6
def figS06():
    fig = F.figure(F.W3, 19.4 * CM)
    # 공통 격자 4행 12열
    #  1–2행 A·B 격자(각 10열) + 범례(2열, 두 행에 걸침)
    #  3행 C·D·E·F(각 3열) / 4행 G(4열) + H(8열)
    G_ = F.Grid(fig, 4, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.95, hratios=[4.47, 4.06, 3.40, 4.00])
    G = load('fig3a_daily_grid.csv')
    ST = {'A': ('o', F.PROV), 'B': ('s', F.OTH), 'C': ('^', F.INK)}

    # ------------------------------------------------ A·B cohort B·C 일별 격자
    a = G_.ax(0, 0, cs=10, padl=1.48, padb=0.95, padr=0.10, padt=1.05)
    box_array(a, 'B', G, day_step=2)
    b = G_.ax(1, 0, cs=10, padl=1.48, padb=0.95, padr=0.10, padt=1.05)
    box_array(b, 'C', G, day_step=2)
    _lc = G_.cell_cm(0, 10, rs=2, cs=2)
    share_bar(fig, [G_.fx(_lc[0] + 0.05), G_.fy(_lc[1] + _lc[3] - 2.10),
                    G_.fx(1.35), G_.fy(0.16)], label='Fraction of\nretrievals')
    absent_key(fig, G_.fx(_lc[0] + 0.05), G_.fy(_lc[1] + _lc[3] - 3.55),
               size=0.013, text='Absent from\nthe group', stack=True)

    # ---------------------------------------------------- C 기준기 순위의 신뢰도
    c = G_.ax(2, 0, cs=4, padl=1.45, padb=1.25, padr=0.12, padt=0.45)
    BR = load('figS06d_baseline_rank.csv')
    for coh, g in BR.groupby('cohort'):
        g = g.sort_values('rate', ascending=False).reset_index(drop=True)
        xs = np.arange(1, len(g) + 1)
        m, col = ST[coh]
        c.errorbar(xs, g.rate, yerr=[np.clip(g.rate - g.lo, 0, None), np.clip(g.hi - g.rate, 0, None)], fmt=m, ms=2.6,
                   color=col, lw=0, elinewidth=0.6, capsize=1.2, markeredgecolor='none',
                   ecolor=col, label='Cohort ' + coh)
    c.set_xticks(range(1, 7)); c.set_xlim(0.5, 6.5); c.set_ylim(-0.04, 1.08)
    F.tidy(c, 'Rank before any change', 'Retrievals per trial')
    c.yaxis.labelpad = 1.0; c.xaxis.labelpad = 1.5; c.tick_params(**TIGHTY)
    c.legend(loc='upper right', fontsize=7.0, frameon=False, borderpad=0.1, labelspacing=0.2,
             handletextpad=0.3)

    # ---------------------------------------------------- D 해제 속도
    d = G_.ax(2, 4, cs=4, padl=1.45, padb=1.25, padr=0.12, padt=0.45)
    R = load('figS06c_release_by_day.csv')
    for coh, g in R.groupby('cohort'):
        m, col = ST[coh]
        d.plot(g.day, g.frac_retrieved, marker=m, ms=3.0, lw=0.9, color=col, clip_on=False,
               label='{} ({:.0f}%)'.format(coh, 100 * g.provider_fraction_before.iloc[0]))
    F.tidy(d, 'Day without the provider', 'Trials with a retrieval')
    d.yaxis.labelpad = 1.0; d.xaxis.labelpad = 1.5; d.tick_params(**TIGHTY)
    d.set_ylim(-0.03, 1.05); d.set_xlim(0.8, 6.2)
    d.legend(loc='lower right', fontsize=7.0, frameon=False, borderpad=0.1, labelspacing=0.15,
             handletextpad=0.3, handlelength=1.1,
             title='Cohort (earlier fraction)', title_fontsize=7.0,
             bbox_to_anchor=(1.05, -0.04))

    # ---------------------------------------------------- E 선발 표현형
    e = G_.ax(2, 8, cs=4, padl=1.45, padb=1.25, padr=0.12, padt=0.45)
    SH = load('figS06e_starter_halves.csv')
    isP = SH.Role.eq('P') if 'Role' in SH else SH.role.eq('provider')
    e.plot([0, 0.9], [0, 0.9], color=F.GRID, lw=0.6)
    e.scatter(SH.first_half[~isP], SH.second_half[~isP], s=7, color=F.OTH, lw=0, label='Others')
    e.scatter(SH.first_half[isP], SH.second_half[isP], s=7, color=F.PROV, lw=0, label='Provider')
    F.tidy(e, 'Starter rate, first half', 'Starter rate,\nsecond half')
    e.yaxis.labelpad = 1.0; e.xaxis.labelpad = 1.5; e.tick_params(**TIGHTY)
    e.legend(loc='lower right', fontsize=7.0, frameon=False, borderpad=0.1, labelspacing=0.2,
             handletextpad=0.3, bbox_to_anchor=(1.02, -0.02))
    F.note(e, 'Phenotype effect\nP = 3 $\\times$ 10$^{-8}$', x=0.03, y=0.97,
           ha='left', va='top', size=7.0)

    # ---------------------------------------------------- F 제공자의 선발 과다
    f = G_.ax(3, 0, cs=3, padl=1.45, padb=1.35, padr=0.12, padt=0.45)
    EX = load('figS06f_starter_excess.csv').sort_values('cohort')
    yy = np.arange(len(EX))[::-1]
    f.barh(yy, EX.observed, color=F.PROV, height=0.62, lw=0)
    for y_, r in zip(yy, EX.itertuples()):
        f.plot([r.expected] * 2, [y_ - 0.34, y_ + 0.34], color=F.INK, lw=1.0, zorder=4)
    f.set_yticks(yy); f.set_yticklabels(EX.cohort, fontsize=7.0)
    f.set_xlim(0, 1.0); f.set_ylim(-0.7, len(EX) + 0.15)
    F.tidy(f, 'Trials started by the provider', 'Cohort')
    f.yaxis.labelpad = 1.0; f.xaxis.labelpad = 1.5; f.tick_params(**TIGHTY)
    F.note(f, 'q < 0.05 in 8 of 8', x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    # ---------------------------------------------------- G 선발자에 따른 회수율
    g_ = G_.ax(3, 3, cs=3, padl=1.58, padb=1.35, padr=0.12, padt=0.45)
    RB = load('figS06g_rate_by_starter.csv')
    for _, r in RB.iterrows():
        g_.plot([0, 1], [r.rate_after_provider, r.rate_after_other], color=F.GRID, lw=0.5, zorder=1)
    g_.scatter(np.zeros(len(RB)), RB.rate_after_provider, s=6, color=F.OTH, lw=0, zorder=3)
    g_.scatter(np.ones(len(RB)), RB.rate_after_other, s=6, color=F.OTH, lw=0, zorder=3)
    for k, col in ((0, 'rate_after_provider'), (1, 'rate_after_other')):
        g_.plot([k - 0.24, k + 0.24], [RB[col].median()] * 2, color=F.INK, lw=1.1, zorder=4)
    w = stats.wilcoxon(RB.rate_after_other, RB.rate_after_provider)
    g_.set_xticks([0, 1])
    g_.set_xticklabels(['Provider\nstarted', 'Another\nmouse started'], fontsize=7.0)
    g_.set_xlim(-0.45, 1.45)
    F.tidy(g_, None, 'Retrievals per trial\nby one non-provider')
    g_.yaxis.labelpad = 1.0; g_.tick_params(**TIGHTY)
    F.note(g_, '{} mice, P = {:.3f}'.format(len(RB), w.pvalue),
           x=0.5, y=1.02, ha='center', va='bottom', size=7.0)

    # ---------------------------------------------------- H 효과 크기
    h = G_.ax(3, 6, cs=6, padl=3.95, padb=1.35, padr=0.12, padt=0.45)
    E = load('figS06h_effect_sizes.csv')
    labs = list(dict.fromkeys(E.effect))
    YL = ['Provider present\nvs removed\n(3 cohorts)',
          'Provider vs another\nmouse started the trial\n(8 cohorts)']
    for i, lab in enumerate(labs):
        gg = E[E.effect == lab]
        yv = len(labs) - 1 - i
        col = F.PROV if i == 0 else F.OTH
        h.scatter(gg.h, np.full(len(gg), yv) + np.linspace(-0.13, 0.13, len(gg)), s=8,
                  color=col, lw=0, zorder=3)
        h.plot([gg.h.median()] * 2, [yv - 0.26, yv + 0.26], color=F.INK, lw=1.1, zorder=4)
        h.text(gg.h.median(), yv + 0.32, '{:.2f}'.format(gg.h.median()), ha='center',
               va='bottom', fontsize=7.0, color=F.INK)
    h.axvline(0, color=F.INK, lw=0.5, ls=(0, (3, 2)))
    h.set_yticks(range(len(labs)))
    h.set_yticklabels(YL[::-1], fontsize=7.0)
    h.set_ylim(-0.55, len(labs) - 0.25); h.set_xlim(-0.3, 2.3)
    F.tidy(h, 'Effect on the others\n(Cohen’s h)', None)
    h.xaxis.labelpad = 1.5; h.tick_params(**TIGHTY)

    for lab, r_, c_ in (('A', 0, 0), ('B', 1, 0), ('C', 2, 0), ('D', 2, 4),
                        ('E', 2, 8), ('F', 3, 0), ('G', 3, 3), ('H', 3, 6)):
        G_.label(r_, c_, TAGS['figS06'] + lab)
    return F.save(fig, 'figS06', PNG, PDF)


if __name__ == '__main__':
    fig3(); figS06()
