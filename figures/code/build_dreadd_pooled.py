# -*- coding: utf-8 -*-
"""DREADD 행동, 시행을 모아 본 결과. A 개체별 회수율(PFC 4 + NAc 8), B PFC 시행 시작→회수 생존곡선,
C PFC 회수 소요(진입→회수) 시행 분포."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
D, PNG, PDF = (os.path.join(SUB, x) for x in ('data', 'png', 'pdf'))
CM = F.CM
load = lambda n: pd.read_csv(os.path.join(D, n))


def km(t, e):
    """Kaplan–Meier: 회수 확률(1−S)."""
    o = np.argsort(t); t, e = t[o], e[o]
    ts, surv = [0.0], [1.0]; s = 1.0; n = len(t); i = 0
    while i < n:
        tt = t[i]; d = ((t == tt) & (e == 1)).sum(); r = (t >= tt).sum()
        if d:
            s *= 1 - d / r; ts.append(tt); surv.append(s)
        i += ((t == tt)).sum()
    return np.array(ts), 1 - np.array(surv)


def fig():
    P = load('dreadd_pfc_trials.csv'); N = load('dreadd_nac_trials.csv')
    N = N[N.condition.isin(['Saline', 'CNO']) & (N.worker != '-')].copy()
    N['cno'] = (N.condition == 'CNO').astype(int); N['retrieved'] = N.worker_retrieved
    R = load('dreadd_pooled_retrieval.csv') if os.path.exists(os.path.join(D, 'dreadd_pooled_retrieval.csv')) else None
    fig = F.figure(F.W3, 6.6 * CM)
    G_ = F.Grid(fig, 1, 12, left=0.06, right=0.10, top=0.50, bottom=0.10, wgap=0.22, hgap=0.8)

    # ---- A 개체별 회수율
    a = G_.ax(0, 0, cs=6, padl=1.45, padb=1.30, padr=0.12, padt=0.62)
    per_p = P.groupby(['cohort', 'cno']).retrieved.mean().unstack('cno')
    per_n = N.groupby(['group', 'cno']).retrieved.mean().unstack('cno')
    labs = ['E1', 'F4', 'G2', 'H4'] + ['#%s' % g.replace('#', '').split(',')[0].strip() for g in per_n.index]
    labs = ['E1', 'F4', 'G2', 'H4'] + [w.replace('#', '') for w in N.groupby('group').worker.first().loc[per_n.index]]
    vals = list(per_p.values) + list(per_n.values)
    xs = np.arange(len(vals), dtype=float); xs[4:] += 0.8
    for x, (s, c), k in zip(xs, vals, range(len(vals))):
        col = F.PROV if k < 4 else F.GAM
        a.plot([x - 0.18, x + 0.18], [s, c], color=col, lw=0.8, zorder=2)
        a.scatter([x - 0.18], [s], s=13, color=F.OTH, lw=0, zorder=3)
        a.scatter([x + 0.18], [c], s=13, color=col, lw=0, zorder=3)
    a.set_xticks(xs); a.set_xticklabels(labs, rotation=90)
    a.tick_params(axis='x', pad=1.5, labelsize=6.6)
    a.set_ylim(-0.05, 1.12); a.set_yticks([0, 0.5, 1.0])
    a.set_ylabel('Trials retrieved by the\ntreated animal (fraction)')
    a.axvline(3.9, color=F.GRID, lw=0.6)
    for xx, lab, col in ((1.5, 'PFC inhibition', F.PROV), (xs[4:].mean(), 'NAc inhibition', F.GAM)):
        a.annotate(lab, xy=(xx, 0), xycoords=('data', 'axes fraction'), xytext=(0, -24),
                   textcoords='offset points', ha='center', va='top', fontsize=7.5, color=col)
    F.note(a, 'PFC 23/24 → 18/23, within-animal P = 0.021\nNAc 36/36 → 32/44, P = 0.001; 3 of 12 animals change',
           x=0.01, y=0.02, ha='left', va='bottom', size=6.5)

    # ---- B 생존곡선
    b_ = G_.ax(0, 6, cs=3, padl=1.35, padb=1.30, padr=0.12, padt=0.62)
    for c, col, lab in ((0, F.OTH, 'Saline'), (1, F.PROV, 'CNO')):
        g = P[P.cno == c]; ts, cum = km(g.time.values.astype(float), g.event.values.astype(int))
        keep = ts <= 180; ts, cum = ts[keep], cum[keep]          # 표시 범위 안에서만
        b_.step(np.append(ts, 180), np.append(cum, cum[-1]), where='post', color=col, lw=1.0, label=lab)
        cens = g[(g.event == 0) & (g.time <= 180)].time.values
        b_.scatter(cens, [cum[-1]] * len(cens), marker='|', s=18, color=col, lw=0.8, zorder=4)
    b_.set_xlim(0, 180); b_.set_ylim(0, 1.04); b_.set_yticks([0, 0.5, 1.0]); b_.set_xticks([0, 60, 120, 180])
    b_.set_xlabel('Time from trial start (s)'); b_.set_ylabel('Snack retrieved by\nthe provider (cumulative)')
    b_.legend(loc='lower right', bbox_to_anchor=(1.02, 0.02), handlelength=1.0, borderpad=0.1)
    F.note(b_, 'PFC, 47 trials\nlog-rank P = 0.011', x=0.97, y=0.855, ha='right', va='center', size=6.2)

    # ---- C 회수 소요
    c_ = G_.ax(0, 9, cs=3, padl=1.35, padb=1.30, padr=0.12, padt=0.62)
    rngj = np.random.default_rng(3)
    for c, col in ((0, F.OTH), (1, F.PROV)):
        g = P[(P.cno == c) & P.Tau_get.notna() & (P.Tau_get > 0)]
        x = c + rngj.uniform(-0.16, 0.16, len(g))
        c_.scatter(x, g.Tau_get, s=11, color=col, lw=0, alpha=0.85, zorder=3)
        med = g.Tau_get.median()
        c_.plot([c - 0.28, c + 0.28], [med, med], color=F.INK, lw=0.9, zorder=4)
    c_.set_yscale('log'); c_.set_ylim(1.5, 2500); c_.set_yticks([2, 5, 10, 20, 50, 100, 500])
    c_.set_yticklabels(['2', '5', '10', '20', '50', '100', '500'])
    c_.set_xticks([0, 1]); c_.set_xticklabels(['Saline', 'CNO']); c_.set_xlim(-0.55, 1.55)
    c_.set_ylabel('Entry to retrieval (s)')
    F.note(c_, '×1.74, P = 0.038\nn = 41 retrieved trials', x=0.03, y=0.97, ha='left', va='top', size=6.5)
    for lab_, c0 in (('A', 0), ('B', 6), ('C', 9)):
        G_.label(0, c0, lab_)
    return F.save(fig, 'figS_dreadd_pooled', PNG, PDF)


if __name__ == '__main__':
    print(fig())
