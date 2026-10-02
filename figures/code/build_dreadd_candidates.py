# -*- coding: utf-8 -*-
"""DREADD 뇌파+행동 종합 그림 후보 세 장.
A 조작→상태→행동 사슬 / B 개체별 3중 요약(균일한 조작, 갈리는 행동) / C 시행 시계열 대시보드."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
CM = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))
BANDS = ['theta1', 'theta2', 'beta1', 'beta2', 'gamma1', 'gamma2']
BL = [r'$\theta$ 4–8', r'$\theta$ 8–12', r'$\beta$ 18–24', r'$\beta$ 24–32', r'$\gamma$ 35–50', r'$\gamma$ 70–90']
MK = {'E': 'o', 'F': 's', 'H': '^', 'G': 'D'}
NAME = {'E': 'E1', 'F': 'F4', 'G': 'G2', 'H': 'H4'}
CCOL = {0: F.OTH, 1: F.PROV}
CENS_Y = 320.0                                            # 미회수 시행의 표시 위치(중도절단)


def data():
    M = load('dreadd_trial_neural.csv')
    M['tilt'] = M['tilt_PFC__Baseline']
    M['tshow'] = np.where(M.event == 1, M.time, CENS_Y)
    return M


def spectrum_panel(ax, M, seg='Baseline'):
    """CNO−Saline Δlog10 PFC 출력, 개체별 선 + 평균."""
    xs = np.arange(6); allv = []
    for coh in ('E', 'F', 'H'):
        g = M[M.cohort == coh]
        v = [g[g.cno == 1]['PFC_%s__%s' % (b, seg)].mean() - g[g.cno == 0]['PFC_%s__%s' % (b, seg)].mean() for b in BANDS]
        allv.append(v)
        ax.plot(xs, v, color=F.OTHL, lw=0.8, marker=MK[coh], ms=3.2, mfc='#ffffff', mec=F.OTH, mew=0.7, zorder=2)
    ax.plot(xs, np.mean(allv, 0), color=F.INK, lw=1.3, marker='o', ms=3.4, zorder=3)
    ax.axhline(0, color=F.INK, lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(BL, rotation=90); ax.tick_params(axis='x', pad=1.5, labelsize=6.6)
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.30, 0.36); ax.set_yticks([-0.2, 0, 0.2])
    ax.set_ylabel('CNO − saline, PFC\n(log$_{10}$ power)')


def tilt_pairs_panel(ax, M):
    """개체별 시행 수준 기저 θ tilt, saline 대 CNO."""
    for k, coh in enumerate(('E', 'F', 'H')):
        g = M[(M.cohort == coh) & M.tilt.notna()]
        for c in (0, 1):
            v = g[g.cno == c].tilt.values; x = k + (c - 0.5) * 0.36
            jit = np.linspace(-0.09, 0.09, len(v))
            ax.scatter(x + jit, v, s=9, color=CCOL[c], lw=0, zorder=3)
            ax.plot([x - 0.14, x + 0.14], [v.mean()] * 2, color=F.INK, lw=0.9, zorder=4)
        d = g[g.cno == 1].tilt.mean() - g[g.cno == 0].tilt.mean()
        ax.text(k, -0.86, '%+.2f' % d, ha='center', va='bottom', fontsize=6.5, color=F.INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(['E1', 'F4', 'H4']); ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.9, 0.12); ax.set_yticks([-0.8, -0.4, 0])
    ax.set_ylabel('Pre-trial $\\theta$ ratio, PFC\n(log$_{10}$ $\\theta$ 8–12 / $\\theta$ 4–8)')


def state_behaviour_panel(ax, M):
    """기저 tilt 대 회수 시각(로그), 실패는 위쪽 중도절단 표시."""
    g = M[M.tilt.notna()]
    for c in (0, 1):
        for coh in ('E', 'F', 'H'):
            s = g[(g.cno == c) & (g.cohort == coh)]
            ok = s.event == 1
            ax.scatter(s[ok].tilt, s[ok].tshow, s=14, marker=MK[coh], color=CCOL[c], lw=0, zorder=3)
            ax.scatter(s[~ok].tilt, s[~ok].tshow, s=16, marker=MK[coh], facecolor='#ffffff',
                       edgecolor=CCOL[c], lw=0.8, zorder=3)
    ax.axhline(CENS_Y / 1.25, color=F.GRID, lw=0.5, ls=(0, (2, 2)))
    ax.set_yscale('log'); ax.set_ylim(3, 520); ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(['5', '10', '20', '50', '100', '200'])
    ax.set_xlim(-0.85, 0.12); ax.set_xticks([-0.8, -0.4, 0])
    ax.set_xlabel('Pre-trial $\\theta$ ratio, PFC (log$_{10}$)'); ax.set_ylabel('Time from door opening\nto retrieval (s)')
    F.note(ax, 'open: not retrieved', x=0.98, y=0.80, ha='right', va='top', size=6.2, color=F.OTH)
    F.note(ax, '$\\beta$ = −0.53/SD, P = 0.004\nany retrieval: P = 0.017',
           x=0.98, y=0.70, ha='right', va='top', size=6.0)


def behaviour_panel(ax, M):
    for k, coh in enumerate(('E', 'F', 'G', 'H')):
        g = M[M.cohort == coh]; s, c = g[g.cno == 0].retrieved.mean(), g[g.cno == 1].retrieved.mean()
        ax.plot([k - 0.18, k + 0.18], [s, c], color=F.PROV, lw=0.8, zorder=2)
        ax.scatter([k - 0.18], [s], s=13, color=F.OTH, lw=0, zorder=3); ax.scatter([k + 0.18], [c], s=13, color=F.PROV, lw=0, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(['E1', 'F4', 'G2', 'H4']); ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.05, 1.12); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel('Trials retrieved\n(fraction)')
    F.note(ax, 'saline → CNO\nwithin-animal P = 0.021', x=0.03, y=0.06, ha='left', va='bottom', size=6.2)


# ============================================================ 후보 A: 사슬
def cand_A():
    M = data()
    fig = F.figure(F.W3, 6.6 * CM)
    G_ = F.Grid(fig, 1, 12, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.22, hgap=0.8)
    a = G_.ax(0, 0, cs=3, padl=1.45, padb=1.30, padr=0.12, padt=0.62); spectrum_panel(a, M)
    F.note(a, 'baseline segment\nthin, E1 F4 H4; thick, mean', x=0.03, y=0.97, ha='left', va='top', size=6.2)
    b = G_.ax(0, 3, cs=3, padl=1.45, padb=1.30, padr=0.12, padt=0.62); tilt_pairs_panel(b, M)
    F.note(b, 'grey saline, orange CNO\nnumbers: CNO − saline', x=0.03, y=0.97, ha='left', va='top', size=6.2)
    c = G_.ax(0, 6, cs=3, padl=1.45, padb=1.30, padr=0.12, padt=0.62); state_behaviour_panel(c, M)
    d = G_.ax(0, 9, cs=3, padl=1.40, padb=1.30, padr=0.12, padt=0.62); behaviour_panel(d, M)
    for lab_, c0 in (('A', 0), ('B', 3), ('C', 6), ('D', 9)): G_.label(0, c0, lab_)
    return F.save(fig, 'cand_dreadd_A_chain', PNG, PDF)


# ============================================================ 후보 B: 개체별 3중
def cand_B():
    M = data(); S = load('dreadd_animal_summary.csv').set_index('cohort')
    fig = F.figure(F.W3, 6.6 * CM)
    G_ = F.Grid(fig, 1, 12, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.22, hgap=0.8)
    a = G_.ax(0, 0, cs=3, padl=1.45, padb=1.30, padr=0.12, padt=0.62); spectrum_panel(a, M)
    F.note(a, 'baseline segment', x=0.03, y=0.97, ha='left', va='top', size=6.2)
    # B: Δ tilt 개체별 (기저·채집)
    b = G_.ax(0, 3, cs=3, padl=1.45, padb=1.30, padr=0.12, padt=0.62)
    xs = np.arange(3)
    for j, (seg, col) in enumerate((('Baseline', F.BET), ('Foraging', F.OTHL))):
        v = [S.loc[c, 'd_tilt_PFC_%s' % seg] for c in ('E', 'F', 'H')]
        b.bar(xs + (j - 0.5) * 0.36, v, 0.32, color=col, lw=0, label=seg)
    b.axhline(0, color=F.INK, lw=0.5)
    b.set_xticks(xs); b.set_xticklabels(['E1', 'F4', 'H4']); b.set_xlim(-0.6, 2.6)
    b.set_ylim(-0.46, 0.06); b.set_yticks([-0.4, -0.2, 0])
    b.set_ylabel('CNO − saline\n$\\theta$ ratio, PFC (log$_{10}$)')
    b.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.02), handlelength=0.9, borderpad=0.1)
    # C: Δ 행동 개체별 (회수율 변화, 회수 시각 변화)
    c = G_.ax(0, 6, cs=3, padl=1.45, padb=1.30, padr=0.12, padt=0.62)
    xs = np.arange(4); cohs = ['E', 'F', 'G', 'H']
    c.bar(xs - 0.18, [S.loc[k, 'd_ret'] for k in cohs], 0.32, color=F.PROV, lw=0, label='retrieval fraction')
    c.bar(xs + 0.18, [S.loc[k, 'd_logTauGet'] for k in cohs], 0.32, color=F.OTH, lw=0, label='log$_{10}$ entry→retrieval')
    c.axhline(0, color=F.INK, lw=0.5)
    c.set_xticks(xs); c.set_xticklabels(['E1', 'F4', 'G2', 'H4']); c.set_xlim(-0.6, 3.6)
    c.set_ylim(-0.9, 0.9); c.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    c.set_ylabel('CNO − saline\n(behaviour)')
    c.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.04), handlelength=0.9, borderpad=0.1, labelspacing=0.2)
    # D: Δ tilt 대 Δ 행동, 개체 점
    d = G_.ax(0, 9, cs=3, padl=1.40, padb=1.30, padr=0.12, padt=0.62)
    for k in ('E', 'F', 'H'):
        d.scatter(S.loc[k, 'd_tilt_PFC_Baseline'], S.loc[k, 'd_logTauGet'], s=22, marker=MK[k], color=F.BET, lw=0, zorder=3)
        d.annotate(NAME[k], (S.loc[k, 'd_tilt_PFC_Baseline'], S.loc[k, 'd_logTauGet']), xytext=(4, 3),
                   textcoords='offset points', fontsize=6.5)
    d.axhline(0, color=F.GRID, lw=0.5); d.set_xlim(-0.46, -0.16); d.set_ylim(-0.25, 0.9)
    d.set_xticks([-0.4, -0.3, -0.2]); d.set_yticks([0, 0.4, 0.8])
    d.set_xlabel('CNO − saline $\\theta$ tilt (log$_{10}$)'); d.set_ylabel('CNO − saline log$_{10}$\nentry→retrieval')
    F.note(d, 'three animals — descriptive', x=0.97, y=0.05, ha='right', va='bottom', size=6.2, color=F.OTH)
    for lab_, c0 in (('A', 0), ('B', 3), ('C', 6), ('D', 9)): G_.label(0, c0, lab_)
    return F.save(fig, 'cand_dreadd_B_animals', PNG, PDF)


# ============================================================ 후보 C: 시행 시계열
def cand_C():
    M = data()
    fig = F.figure(F.W3, 7.6 * CM)
    G_ = F.Grid(fig, 2, 3, left=0.06, right=0.10, top=0.55, bottom=0.10, wgap=0.35, hgap=0.30, hratios=[1, 1])
    for k, coh in enumerate(('E', 'F', 'H')):
        g = M[M.cohort == coh].copy()
        g = pd.concat([g[g.cno == 0].sort_values('trial'), g[g.cno == 1].sort_values('trial')])
        x = np.arange(len(g)); cut = (g.cno == 0).sum() - 0.5
        a = G_.ax(0, k, padl=1.45 if k == 0 else 0.55, padb=0.30, padr=0.12, padt=0.55)
        a.plot(x[g.cno == 0], g[g.cno == 0].tilt, color=F.OTH, lw=0.8, marker='o', ms=3, zorder=3)
        a.plot(x[g.cno == 1], g[g.cno == 1].tilt, color=F.PROV, lw=0.8, marker='o', ms=3, zorder=3)
        a.axvline(cut, color=F.GRID, lw=0.6); a.set_xlim(-0.6, len(g) - 0.4); a.set_ylim(-0.9, 0.12)
        a.set_yticks([-0.8, -0.4, 0]); a.set_xticks([])
        a.set_title(NAME[coh], pad=2, fontsize=8)
        if k == 0: a.set_ylabel('Pre-trial $\\theta$ ratio\nPFC (log$_{10}$)')
        else: a.set_yticklabels([])
        b = G_.ax(1, k, padl=1.45 if k == 0 else 0.55, padb=1.05, padr=0.12, padt=0.15)
        for c in (0, 1):
            s = g[g.cno == c]; xx = x[g.cno == c]
            ok = (s.event == 1).values
            b.scatter(xx[ok], s[ok].time, s=14, color=CCOL[c], lw=0, zorder=3)
            b.scatter(xx[~ok], [CENS_Y] * (~ok).sum(), s=16, facecolor='#ffffff', edgecolor=CCOL[c], lw=0.8, zorder=3)
            b.vlines(xx, 3, np.where(ok, s.time, CENS_Y), color=CCOL[c], lw=0.6, zorder=2)
        b.axvline(cut, color=F.GRID, lw=0.6); b.set_yscale('log'); b.set_ylim(3, 520)
        b.set_yticks([5, 20, 100]); b.set_yticklabels(['5', '20', '100'] if k == 0 else [])
        b.set_xlim(-0.6, len(g) - 0.4); b.set_xticks(x); b.set_xticklabels(g.trial.astype(int))
        b.tick_params(axis='x', labelsize=6.4)
        b.set_xlabel('Trial (saline | CNO)')
        if k == 0: b.set_ylabel('Door opening →\nretrieval (s)')
        if k == 2: F.note(b, 'open: not retrieved', x=0.5, y=0.985, ha='center', va='top', size=6.2, color=F.OTH)
    G_.label(0, 0, 'A'); G_.label(1, 0, 'B')
    return F.save(fig, 'cand_dreadd_C_timeline', PNG, PDF)


if __name__ == '__main__':
    for f in (cand_A, cand_B, cand_C):
        print(f())
