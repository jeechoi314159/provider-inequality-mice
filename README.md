# Provider inequality emerges through withdrawal of capable group members in mice — code

Code for the article *"Provider inequality emerges through withdrawal of capable group members in mice"*
(J. Lee, G.-H. Lee, S. Y. Kim, D. Jung, J. Kim, Y. Lee, S.-Y. Choi, R. C. Froemke, J. H. Choi; submitted).

Eight cohorts of mice lived with a robot that carried a snack; one mouse retrieved it at a cost and all ate.
This repository reproduces every main and supplementary figure from the deposited figure source data and documents
how those data were derived from the behavioural records and processed local field potentials (LFPs).

| What | Where |
|---|---|
| Code (this repository) | GitHub `jeechoi314159/provider-inequality-mice`, archived at Zenodo (DOI on release) |
| Data | Dryad (DOI on publication; reviewers receive a private link) |
| Tabulated values behind every figure panel | Data S1 of the article and `01_figure_source_data/` on Dryad |
| Raw LFP recordings and raw videos | not deposited because of their size; available from the corresponding author on request |

## Repository layout

```
figures/code/      one script per figure (build_*.py), shared style (figstyle.py), layout check (audit.py)
figures/art/       illustrations used in the figures
extract/           behavioural records -> figure source data (extract_*.py)
analysis/          LFP and statistical analyses whose outputs are figure source data
  lfp_events/        event-aligned band power, cluster tests, theta tilt, decoders, entry response (Fig. 5D–F, figs. S9–S12)
  heldout_auc/       out-of-sample identification of the provider (Fig. 5C, fig. S7)
  member_change/     Days 21–45 of cohort A (fig. S14)
  chemogenetics/     DREADD experiments (Fig. 5G–J, figs. S15–S17)
  movement/          kinetic-energy covariate
  raw_lfp/           band-power extraction from raw CBRAIN recordings (raw data on request)
pvsnp_paths.py     all data locations, settable by environment variables
scripts/           run_figures.sh (rebuild all figures), compare_png.py (pixel comparison)
docs/figure_panel_map.csv   panel -> source-data file -> Data S1 sheet -> script
```

## Reproduce the figures

1. Create the environment (Python 3.10):
   `conda env create -f environment.yml && conda activate provider-inequality-mice`
   (or `pip install -r requirements.txt`).
2. Download `01_figure_source_data/` from Dryad and place its contents in `figures/data/`.
3. Run `bash scripts/run_figures.sh`. PNG (600 dpi) and PDF files are written to `figures/png/` and `figures/pdf/`;
   about one minute on a laptop. The order of the scripts in `run_figures.sh` matters.
4. Optional: `python scripts/compare_png.py <folder with the published PNGs>` compares the output pixel by pixel.

Fonts: the figures were set in Arial. Arial is not redistributed here; place the Arial `.ttf` files in
`figures/fonts/` to obtain pixel-identical output, otherwise `figstyle.py` falls back to Liberation Sans or DejaVu Sans
(same layout, slightly different glyphs). With Arial and the package versions in `environment.yml`, the regenerated
figures were verified identical to the submitted ones (all 30 PNG files, maximum pixel difference 0).

## From records to figure data

The extraction and analysis scripts read their inputs through `pvsnp_paths.py`:

| Variable | Default | Content (Dryad folder) |
|---|---|---|
| `PVSNP_FIGDATA` | `figures/data/` | figure source data, written by `extract/` and `analysis/` (01) |
| `PVSNP_RAWROOT` | `data/behaviour_records/` | behavioural records with their original folder names under `Data/` (02) |
| `PVSNP_INTERMEDIATE` | `data/intermediate/` | trial-level tables shared by several scripts (02) |
| `PVSNP_LFP` | `data/lfp_processed/` | processed LFP arrays: `powerspect_extract/`, `member_change/` (04) |
| `PVSNP_CHEMO` | `data/chemogenetics/` | DREADD records and band power (05) |
| `PVSNP_RAW_LFP` | (empty) | raw recordings, not public |
| `PVSNP_SOURCE_DATA` | (empty) | 2026 source-data workbook used by three extraction steps, available on request |

`extract/extract_r5.py` was re-run through this configuration and reproduced its figure source data
(identical files; one table equal to within 1e-16). The scripts in `analysis/` are the versions that produced the
deposited outputs; steps that start from raw recordings (`analysis/raw_lfp/`, parts of `analysis/member_change/`) need
the raw data, which are available on request. Comments in some scripts are in Korean.

## Software

Python 3.10; numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9, pillow 12.3.0, openpyxl 3.1.5;
statsmodels and scikit-learn for the mixed models and decoders; xlrd for `.xls` records.

## License and citation

Code: MIT (see `LICENSE`). Data on Dryad: CC0. Please cite the article (see `CITATION.cff`).

Contact: Jee Hyun Choi, Korea Institute of Science and Technology (jeechoi@kist.re.kr).
