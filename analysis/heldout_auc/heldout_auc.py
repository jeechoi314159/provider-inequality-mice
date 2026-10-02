# Recompute held-out identification of the provider from pre-entry spectral features.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *(['..'] * 2))))
from pvsnp_paths import *  # data locations (see pvsnp_paths.py)
# Nested: elastic net (l1_ratio 0.5, 5-fold CV for alpha) refitted inside every training fold;
# score = benefit − cost on the held-out mouse (LOMO) or cohort (LOCO); AUC over 18 mice (7 P, 11 NP).
# Uncertainty: stratified bootstrap of mice (2000) on the out-of-sample scores; permutation null (1000):
# work rates AND role labels permuted together across mice, full pipeline refit.
import numpy as np, pandas as pd, warnings, sys, time
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
IR = pd.read_csv((INTER + 'data_for_IRC.csv'))
F = [c for c in IR.columns if c.startswith('Pre_')]
mice = sorted(IR.Mouse_ID.unique())
Mf = IR.groupby('Mouse_ID')[F].mean().loc[mice]
WR = IR.groupby('Mouse_ID').WR.first().loc[mice].values
grp = IR.groupby('Mouse_ID').Group.first().loc[mice].values
role = (IR.groupby('Mouse_ID').Role.first().loc[mice].values == 'P').astype(int)
X = Mf.values
def enet_fit(Xt, yt):
    mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1
    en = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0, max_iter=20000, n_alphas=60).fit((Xt - mu) / sd, yt)
    return en.coef_, mu, sd
def oos_scores(X, y_wr, folds):
    s = np.full(len(y_wr), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y_wr)), te)
        b, mu, sd = enet_fit(X[tr], y_wr[tr])
        Z = (X[te] - mu) / sd
        s[te] = (Z * np.clip(b, 0, None)).sum(1) - (Z * np.clip(-b, 0, None)).sum(1)
    return s
folds_m = [np.array([i]) for i in range(len(mice))]
folds_c = [np.where(grp == g)[0] for g in sorted(set(grp))]
rng = np.random.default_rng(1)
out = []
for name, folds in (('mouse', folds_m), ('cohort', folds_c)):
    t0 = time.time()
    s = oos_scores(X, WR, folds); auc = roc_auc_score(role, s)
    # bootstrap CI (stratified by role) on the OOS scores
    P, N = np.where(role == 1)[0], np.where(role == 0)[0]; bs = []
    for _ in range(4000):
        i = np.concatenate([rng.choice(P, len(P)), rng.choice(N, len(N))]); bs.append(roc_auc_score(role[i], s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # permutation null, full pipeline
    NP_ = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; null = []
    for k in range(NP_):
        p = rng.permutation(len(mice)); sp = oos_scores(X, WR[p], folds); null.append(roc_auc_score(role[p], sp))
    null = np.array(null); pP = (np.sum(null >= auc) + 1) / (len(null) + 1)
    out.append(dict(left_out=name, auc=round(auc, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), perm_P=round(pP, 4),
                    n_perm=NP_, null_median=round(float(np.median(null)), 3), n_mice=len(mice), n_P=int(role.sum())))
    print(out[-1], 'elapsed %.0f s' % (time.time() - t0), flush=True)
    pd.DataFrame(dict(Mouse_ID=mice, Group=grp, Role=np.where(role == 1, 'P', 'NP'), oos_score=s)).to_csv('heldout_scores_%s.csv' % name, index=False)
pd.DataFrame(out).to_csv('heldout_auc_verified.csv', index=False)
