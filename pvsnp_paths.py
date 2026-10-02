"""Data locations for the extraction and analysis scripts.

Every location can be set with an environment variable; the defaults point inside the repository.
Download the Dryad dataset (see README) and either unpack it under data/ or point the variables at it.
"""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))


def _d(var, default):
    p = os.environ.get(var, default)
    return p if (not p or p.endswith(os.sep)) else p + os.sep


FIGDATA = _d('PVSNP_FIGDATA', os.path.join(ROOT, 'figures', 'data'))           # figure source data (Dryad 01_figure_source_data)
RAWROOT = _d('PVSNP_RAWROOT', os.path.join(ROOT, 'data', 'behaviour_records'))  # behavioural records (Dryad 02_behaviour); contains Data/
BEHAV = RAWROOT + 'Data' + os.sep
INTER = _d('PVSNP_INTERMEDIATE', os.path.join(ROOT, 'data', 'intermediate'))   # trial-level tables (Dryad 02_behaviour/intermediate)
LFP = _d('PVSNP_LFP', os.path.join(ROOT, 'data', 'lfp_processed'))              # processed LFP arrays (Dryad 04_lfp_processed)
CHEMO = _d('PVSNP_CHEMO', os.path.join(ROOT, 'data', 'chemogenetics'))         # DREADD records (Dryad 05_chemogenetics)
RAW_LFP = _d('PVSNP_RAW_LFP', '')                                                # raw CBRAIN recordings: not public, available on request
WORK = _d('PVSNP_WORK', os.path.join(ROOT, 'work'))                              # scratch outputs
FIGCODE = os.path.join(ROOT, 'figures', 'code')
LFPCODE = os.path.join(ROOT, 'analysis', 'lfp_events')
