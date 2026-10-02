# -*- coding: utf-8 -*-
"""fig. S7 — 표본 밖 제공자 식별 (2026-10-01 검증 재계산). data/figS07_heldout_scores.csv, data/figS07_null_auc.csv,
data/fig5c_heldout_auc.csv 만 읽는다. 폭 12.1 cm, 2행 2열: A·B 개체별 held-out 점수(역할별), C·D 순열 귀무분포."""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import build_r12 as R
warnings.filterwarnings('ignore')
CM, load, PNG, PDF, TIGHTY = F.CM, R.load, R.PNG, R.PDF, R.TIGHTY


def figS07():
    SC = load('figS07_heldout_scores.csv'); NU = load('figS07_null_auc.csv'); AU = load('fig5c_heldout_auc.csv').set_index('left_out')
    fig = F.figure(F.W2, 9.4 * CM)
    G_ = F.Grid(fig, 2, 2, left=0.06, right=0.30, top=0.50, bottom=0.10, wgap=0.35, hgap=0.95)
    rng = np.random.default_rng(4)
    for k, (lo, lab) in enumerate((('mouse', 'One mouse held out'), ('cohort', 'One cohort held out'))):
        # --- A·B 개체별 점수
        ax = G_.ax(0, k, padl=1.42, padb=1.20, padr=0.12, padt=0.62)
        s = SC[SC.left_out == lo]
        for j, (role, col) in enumerate((('NP', F.OTH), ('P', F.PROV))):
            v = s[s.Role == role].oos_score.values
            ax.scatter(np.full(v.size, j) + rng.uniform(-0.18, 0.18, v.size), v, s=12, color=col, lw=0, zorder=3)
            ax.plot([j - 0.28, j + 0.28], [np.median(v)] * 2, color=F.INK, lw=1.0, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Others\n(11)', 'Providers\n(7)'], fontsize=7.0)
        for _t, _c in zip(ax.get_xticklabels(), (F.OTH, F.PROV)): _t.set_color(_c)
        ax.set_xlim(-0.6, 1.6)
        F.tidy(ax, None, 'Held-out score\n(benefit − cost)' if k == 0 else None); ax.yaxis.labelpad = 1.0; ax.tick_params(**TIGHTY)
        r = AU.loc[lo]
        F.note(ax, '%s\nAUC = %.2f (%.2f–%.2f)' % (lab, r.auc, r.ci_lo, r.ci_hi), x=0.5, y=1.02, ha='center', va='bottom', size=7.0)
        # --- C·D 순열 귀무분포
        bx = G_.ax(1, k, padl=1.42, padb=1.20, padr=0.12, padt=0.62)
        nv = NU[NU.left_out == lo].null_auc.values
        bx.hist(nv, bins=np.arange(0.1, 1.001, 0.03), color=F.OTHL, lw=0)
        bx.axvline(r.auc, color=F.PROV, lw=1.2)
        bx.set_xlim(0.1, 1.02); bx.set_xticks([0.25, 0.5, 0.75, 1.0]); bx.set_xticklabels(['0.25', '0.50', '0.75', '1.00'])
        F.tidy(bx, 'AUC under permuted work rates', 'Permutations' if k == 0 else None)
        bx.xaxis.labelpad = 1.5; bx.yaxis.labelpad = 1.0; bx.tick_params(**TIGHTY)
        F.note(bx, 'observed %.2f\nP = %.2f (%d draws)' % (r.auc, r.perm_P, int(r.n_perm)), x=0.03, y=0.97, ha='left', va='top', size=7.0)
    for lab_, r_, c_ in (('A', 0, 0), ('B', 0, 1), ('C', 1, 0), ('D', 1, 1)):
        G_.label(r_, c_, lab_)
    return F.save(fig, 'figS07', PNG, PDF)


if __name__ == '__main__':
    print('saved', figS07())
