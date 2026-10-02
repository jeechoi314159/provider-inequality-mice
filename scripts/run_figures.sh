#!/usr/bin/env bash
# Rebuild every main and supplementary figure from figures/data (Dryad 01_figure_source_data).
# Order matters: later scripts import helpers from earlier ones.
set -euo pipefail
PY="${PYTHON:-python3}"
cd "$(dirname "$0")/../figures"
test -d data || { echo "figures/data is missing: download 01_figure_source_data from Dryad (see README)"; exit 1; }
mkdir -p png pdf audit
cd code
for s in build_r12 build_s02 build_fig12 build_r34 build_r5 build_fig5 build_figS07 build_lfp_summary \
         build_follower_spectro build_auc build_figS_extra build_r67 build_dreadd_pooled build_dreadd_candidates finalize_supp; do
  echo "== $s"; "$PY" "$s.py"
done
echo "Figures written to figures/png and figures/pdf"
