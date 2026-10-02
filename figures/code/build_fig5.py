# -*- coding: utf-8 -*-
"""Fig. 5 초안 (신경 한 절). data/ 의 파일만 읽는다.
A–C 개체 간: 프로파일은 기여를 기술하나 제공자를 지목하지 못함.
D–F 개체 내: 동료 행동 직후 PFC high beta 가 그 시행의 합류와 이어짐(운동량 통제 포함).
G–H 개입: PFC 억제가 같은 대역을 낮추고 제공 행동이 같은 방향으로 움직임."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
CM = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
BANDS = ['low_theta', 'high_theta', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']
BL = [r'$\theta$ 4–8', r'$\theta$ 8–12', r'$\beta$ 18–24', r'$\beta$ 24–32',
      r'$\gamma$ 35–50', r'$\gamma$ 70–90']
DB = ['theta1', 'theta2', 'beta1', 'beta2', 'gamma1', 'gamma2']


def ladder(vals, sep, iters=200):
    """겹치는 라벨을 최소 간격까지 서로 밀어낸다. 무리의 무게중심은 보존(원자료 불변)."""
    y = np.asarray(vals, float).copy(); o = np.argsort(y)
    for _ in range(iters):
        moved = False
        for k in range(len(o) - 1):
            a, b = o[k], o[k + 1]
            d = sep - (y[b] - y[a])
            if d > 1e-9:
                y[a] -= d / 2.0; y[b] += d / 2.0; moved = True
        if not moved:
            break
    return y


HIB = '#2a9d8f'   # PFC high β (Fig. 5 D·F); 2026-10-02, 보라·주황과 색각 구분 검증(ΔE ≥ 14)


def fig5():
    fig = F.figure(F.W3, 15.8 * CM)
    G_ = F.Grid(fig, 3, 12, left=0.06, right=0.10, top=0.50, bottom=0.10,
                wgap=0.22, hgap=0.80, hratios=[1.13, 1.02, 1.22])

    # --------------------------------------------------- A 기여 프로파일 가중치
    W = load('fig4c_enet_weights.csv').set_index(['region', 'band'])
    a = G_.ax(0, 0, cs=5, padl=1.35, padb=1.52, padr=0.10, padt=0.62)
    xs, labs = [], []
    for gi, g in enumerate(('PFC', 'NAc', 'BLA')):
        for bi, b in enumerate(BANDS):
            x = gi * 6.8 + bi
            w = float(W.loc[(g, b), 'weight'])
            a.bar(x, w, 0.74, color=(F.PROV if w > 0 else F.BET), lw=0)
            xs.append(x); labs.append(BL[bi])
        a.annotate(g, xy=(gi * 6.8 + 2.5, 0), xycoords=('data', 'axes fraction'),
                   xytext=(0, -34), textcoords='offset points',
                   ha='center', va='top', fontsize=7.5, color=F.INK)
    a.axhline(0, color=F.INK, lw=0.5)
    a.set_xticks(xs); a.set_xticklabels(labs, rotation=90)
    a.tick_params(axis='x', pad=1.5, labelsize=6.8)
    a.set_xlim(-0.9, 2 * 6.8 + 5.9); a.set_ylim(-6.2, 8.3)
    a.set_ylabel('Elastic-net weight\non work rate')
    F.note(a, 'benefit', x=0.015, y=0.97, ha='left', va='top', size=7.0, color=F.PROV)
    F.note(a, 'cost', x=0.015, y=0.85, ha='left', va='top', size=7.0, color=F.BET)

    # ------------------------------------------------------- B benefit–cost 평면
    BC = load('fig4d_benefit_cost.csv')
    b_ = G_.ax(0, 5, cs=4, padl=1.30, padb=1.52, padr=0.12, padt=0.62)
    lim = max(abs(BC.benefit).max(), abs(BC.cost).max()) * 1.18
    b_.plot([-lim, lim], [-lim, lim], color=F.GRID, lw=0.6, zorder=0)
    for role, col, lab in (('NP', F.OTH, 'Others'), ('P', F.PROV, 'Providers')):
        s = BC[BC.Role == role]
        b_.scatter(s.benefit, s.cost, s=13, color=col, lw=0, label=lab, zorder=3)
    b_.set_xlim(-lim, lim); b_.set_ylim(-lim, lim)
    b_.set_xlabel('Benefit (a.u.)'); b_.set_ylabel('Cost (a.u.)')
    # 2026-10-02: 범례 글자가 마커와 겹치지 않게 — 마커 칸·간격을 두고 축 안쪽으로 (사용자 red note)
    lg = b_.legend(loc='upper left', bbox_to_anchor=(0.01, 1.01), handlelength=0.7,
                   handletextpad=0.35, labelspacing=0.22, borderpad=0.1, borderaxespad=0.15)
    for t, col in zip(lg.get_texts(), (F.OTH, F.PROV)):
        t.set_color(col)
    F.note(b_, 'equal benefit and cost', x=0.97, y=0.03, ha='right', va='bottom',
           size=6.5, color=F.OTH)

    # ------------------------------------------------------------- C 표본 밖 AUC
    AU = load('fig5c_heldout_auc.csv')          # 2026-10-01 재계산: 부트스트랩 95% CI, 전 절차 순열 P
    c_ = G_.ax(0, 9, cs=3, padl=1.32, padb=1.52, padr=0.12, padt=0.62)
    xs = np.arange(len(AU))
    c_.scatter(xs, AU.auc, s=16, color=F.INK, lw=0, zorder=3)
    c_.errorbar(xs, AU.auc, yerr=[AU.auc - AU.ci_lo, AU.ci_hi - AU.auc], fmt='none', ecolor=F.INK,
                elinewidth=0.7, capsize=2.0, zorder=2)
    c_.axhline(0.5, color=F.INK, lw=0.6, ls=(0, (2.5, 1.8)))
    for x, (_i, r) in zip(xs, AU.iterrows()):
        c_.text(x + 0.10, r.auc, '%.2f\nP = %.2f' % (r.auc, r.perm_P), ha='left', va='center',
                fontsize=6.5, color=F.INK)
    c_.set_xticks(xs); c_.set_xticklabels(['One mouse\nheld out', 'One cohort\nheld out'])
    c_.set_xlim(-0.5, len(AU) - 0.3); c_.set_ylim(0.3, 1.02); c_.set_yticks([0.5, 0.75, 1.0])
    c_.set_yticklabels(['0.50', '0.75', '1.00'])
    c_.set_ylabel('AUC for identifying\nthe provider (95% CI)')
    F.note(c_, 'chance', x=0.02, y=0.245, ha='left', va='top', size=6.5, color=F.OTH)

    # ------------------------------------------- D 개체 내 시간 곡선 (θ tilt, β 비교)
    TC = load('fig5d_tilt_timecourse.csv'); CL = load('fig5d_tilt_clusters.csv')
    tc = TC[TC.key.str.startswith('theta')].sort_values('t')
    tb = TC[TC.key.str.startswith('PFC')].sort_values('t')
    cl = CL[CL.key.str.startswith('theta')].sort_values('P_family').iloc[0]
    d_ = G_.ax(1, 0, cs=4, padl=1.60, padb=1.05, padr=0.12, padt=0.62)
    d_.add_patch(Rectangle((cl.t_start, -8), cl.t_end - cl.t_start, 18, fc=F.BET,
                           alpha=0.13, ec='none', zorder=0))
    d_.plot(tb.t, tb.tstat, color=HIB, lw=1.0, zorder=2)      # 2026-10-02: 회색 → 청록 (잘 보이게)
    d_.plot(tc.t, tc.tstat, color=F.BET, lw=1.0, zorder=3)
    d_.axhline(0, color=F.INK, lw=0.5)
    d_.axvline(0, color=F.OTH, lw=0.6, ls=(0, (2.5, 1.8)))
    d_.set_xlim(-8, 3); d_.set_ylim(-3.4, 10.6)
    d_.set_yticks([-2, 0, 2, 4, 6]); d_.set_xticks([-8, -6, -4, -2, 0, 2])
    d_.set_xlabel('Time from a cagemate’s entry (s)')
    d_.set_ylabel('Within-mouse $t$\n(predicting joining)')
    F.note(d_, '$\\theta$ tilt (purple) vs PFC high $\\beta$ (teal)\n'
           '%.1f–%.1f s, family-wise P < 0.001\n'
           '5 non-providers; evoked mean nil'
           % (cl.t_start, cl.t_end), x=0.02, y=0.97, ha='left', va='top', size=6.2)

    # ------------------------------------------------------------- E 개체별 효과
    pm = load('fig5e_tilt_per_mouse.csv')
    pm = pm[pm['sample'] == 'n616'].sort_values('mouse')
    e_ = G_.ax(1, 4, cs=3, padl=2.02, padb=1.05, padr=0.12, padt=0.62)
    xs = np.arange(len(pm))
    e_.axhline(0, color=F.INK, lw=0.5)
    for x, v in zip(xs, pm.beta_tilt):
        e_.plot([x, x], [0, v], color=F.BET, lw=0.8, zorder=2)
    e_.scatter(xs, pm.beta_tilt, s=15, color=F.BET, lw=0, zorder=3)
    e_.set_xticks(xs); e_.set_xticklabels(pm.mouse)
    e_.set_xlim(-0.6, len(pm) - 0.4); e_.set_ylim(-0.02, 0.215)
    e_.set_yticks([0, 0.1, 0.2])
    e_.set_ylabel('Change in joining\nper SD of $\\theta$ tilt')
    F.note(e_, 'same sign in 5/5', x=0.97, y=0.97, ha='right', va='top', size=6.5)

    # ---------------------------------------------------------------- F 강건성
    RB = load('fig5f_tilt_robustness.csv')
    LAB = {'Both windows available': 'Both windows',
           'Post-entry window available': 'Post-entry window',
           'Movement controlled': 'Movement controlled',
           'Independent period, Days 21–45': 'Days 21–45',
           'PFC high β, previous feature': 'PFC high $\\beta$ (previous)'}
    KCOL = {'tilt': F.BET, 'tilt_mc': F.BET, 'beta': HIB}     # 2026-10-02: 회색 → 청록
    f_ = G_.ax(1, 7, cs=5, padl=2.12, padb=1.05, padr=2.02, padt=0.62)
    ys = np.arange(len(RB))[::-1]
    f_.axvline(0, color=F.INK, lw=0.5)
    for y, (_i, r) in zip(ys, RB.iterrows()):
        col = KCOL[r.kind]
        f_.plot([0, r.beta], [y, y], color=col, lw=0.8, zorder=2)
        f_.scatter([r.beta], [y], s=15, color=col if r.kind != 'tilt_mc' else '#ffffff',
                   edgecolor=col, lw=0.8 if r.kind == 'tilt_mc' else 0, zorder=3)
        ptxt = 'P < 0.001' if r.P < 0.001 else 'P = %.2g' % r.P
        f_.text(0.106, y, '%s, n = %d' % (ptxt, int(r.n)), ha='left', va='center',
                fontsize=6.2, color=F.INK)
    f_.set_yticks(ys); f_.set_yticklabels([LAB.get(c, c) for c in RB.check])
    f_.tick_params(axis='y', length=0, pad=1.5, labelsize=6.4)
    f_.set_ylim(-0.7, len(RB) - 0.3); f_.set_xlim(-0.006, 0.104)
    f_.set_xticks([0, 0.05, 0.10])
    f_.set_xlabel('Change in joining per SD')
    f_.spines['left'].set_visible(False)

    # ------------------------------------------ G–J 개입: 조작 → 상태 → 행동 (후보 A)
    import build_dreadd_candidates as DC
    DM = DC.data()
    PB2 = 1.30
    g_ = G_.ax(2, 0, cs=3, padl=1.45, padb=PB2, padr=0.12, padt=0.87); DC.spectrum_panel(g_, DM)
    _lo, _hi = g_.get_ylim(); g_.set_ylim(_lo, _hi + 0.06)
    F.note(g_, 'baseline segment\nthin, E1 F4 H4; thick, mean', x=0.03, y=0.97, ha='left', va='top', size=6.2)
    h_ = G_.ax(2, 3, cs=3, padl=1.45, padb=PB2, padr=0.12, padt=0.87); DC.tilt_pairs_panel(h_, DM)
    h_.set_ylim(-0.9, 0.30)   # 상단 주석 공간
    F.note(h_, 'grey saline, orange CNO\nnumbers: CNO − saline', x=0.03, y=0.97, ha='left', va='top', size=6.2)
    i_ = G_.ax(2, 6, cs=3, padl=1.45, padb=PB2, padr=0.12, padt=0.87); DC.state_behaviour_panel(i_, DM)
    for _t in list(i_.texts): _t.remove()          # 후보 A 주석 제거 후 재배치(데이터 충돌 회피)
    F.note(i_, 'not retrieved', x=0.98, y=0.90, ha='right', va='center', size=6.0, color=F.OTH)
    F.note(i_, '−0.20 log$_{10}$ s per SD\nP = 0.004\nany retrieval: P = 0.018',
           x=0.98, y=0.84, ha='right', va='top', size=6.0)
    j_ = G_.ax(2, 9, cs=3, padl=1.40, padb=PB2, padr=0.12, padt=0.87)
    # 전체 표본(4마리 47시행)으로 모아 본 회수: 시행 시작부터의 누적 회수 확률 (Kaplan–Meier), saline 대 CNO
    from build_dreadd_pooled import km as _km
    PT = pd.read_csv(os.path.join(D, 'dreadd_pfc_trials.csv'))
    for c, col, lab in ((0, F.OTH, 'Saline'), (1, F.PROV, 'CNO')):
        g = PT[PT.cno == c]; ts, cum = _km(g.time.values.astype(float), g.event.values.astype(int))
        keep = ts <= 180; ts, cum = ts[keep], cum[keep]
        j_.step(np.append(ts, 180), np.append(cum, cum[-1]), where='post', color=col, lw=1.1,
                label='%s (%d/%d)' % (lab, int(g.retrieved.sum()), len(g)))
    j_.set_xlim(0, 180); j_.set_xticks([0, 60, 120, 180]); j_.set_ylim(0, 1.72); j_.set_yticks([0, 0.5, 1.0])
    j_.set_xlabel('Time from trial start (s)'); j_.set_ylabel('Trials retrieved\n(cumulative fraction)')
    j_.legend(loc='lower right', bbox_to_anchor=(1.02, 0.02), fontsize=6.2, borderpad=0.1, labelspacing=0.2, handlelength=1.1)
    F.note(j_, 'four providers, 47 trials\nwithin-animal P = 0.021\nlog-rank P = 0.011',
           x=0.03, y=0.97, ha='left', va='top', size=6.0)

    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 5), ('C', 0, 9),
                         ('D', 1, 0), ('E', 1, 4), ('F', 1, 7),
                         ('G', 2, 0), ('H', 2, 3), ('I', 2, 6), ('J', 2, 9)):
        G_.label(r_, c_, lab_)
    return F.save(fig, 'Fig5', PNG, PDF)


if __name__ == '__main__':
    print('saved', fig5())
