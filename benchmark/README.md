# benchmark

Physical-organic criteria for triplet energy transfer, scored against the Stern–Volmer
screen: can they tell which photocatalyst–quencher pairs quench, and at what
computational cost?

```
uv run reproduce.py benchmark               classification and cost
uv run reproduce.py further_analysis        every script in further_analysis/
uv run benchmark/classification.py          quenches or not
uv run benchmark/cost.py                    CPU time per quencher
uv run --extra ml python benchmark/literature/predict.py      literature predictions
uv run --extra ml python benchmark/literature/<script>.py     one literature comparison
```

Inputs: `../data/exp_dataset_ent.csv`, which also carries the molecular radii of
photocatalyst and quencher for the contact distance (`pc_radius_dft`, `q_radius_dft`,
computed in the data stage from the S0 DFT geometries); `cost.py` also reads
`../data/computational_data_processed/quencher_timings.csv` and `quencher_timings_ml.csv`,
and the quencher SMILES from `../data/compound_identification.xlsx`.

Tests: `uv run --no-project --with pytest --with numpy --with pandas --with scipy pytest benchmark/tests`

## Layout

```
benchmark/
├── pairs.py            loads the dataset and names its columns; which pairs are scored
├── theory.py           barrier and rate expressions
├── scoring.py          metrics, AUC, choice of the best setting
├── classification.py   the classification criteria
├── cost.py             CPU time per quencher for each criterion
├── tests/              pytest tests of theory, scoring, classification, cost and the
│                       further analyses
├── literature/         the literature benchmark: predict.py, common.py, one script per study
├── results/
│   ├── classification/
│   │   ├── settings.csv    every DFT setting of every criterion
│   │   └── best.csv        best DFT setting per criterion, and the same setting with ML
│   ├── cost/
│   └── literature/     predictions and comparison tables of the literature benchmark
└── further_analysis/   variants and diagnostics beyond the benchmark table
    ├── labels/         the analyses of the Stern–Volmer labels
    └── results/
```

## The pairs

`exp_dataset_ent` is built in `../data/` by `build_exp_dataset_ent.py`.
- **Non-quenching pairs:** all kept.
- **Quenching pairs:** kept only if every single-electron-transfer channel with a
  measured quencher potential is more than 0.25 V uphill.

`pairs.py` leaves out PC22 on loading, because its emission is too weak for Stern–Volmer
analysis, and the quenchers Q31 (quenching that is not EnT) and Q38 (1,3-butadiene, no
pair quenches), and scores a pair only if every input of every criterion is present, so all
criteria are scored on the same pairs.

## Classification

The ground truth is `quenches`. Photocatalyst properties are DFT throughout.

| criterion | predicts quenching when | variants | chosen on the data |
|---|---|---|---|
| `adiabatic` | ΔG_EnT ≤ 0 | — | nothing |
| `dvte` | E_T(PC) ≥ DvTE(p) | p = 0.005, 0.008, 0.01, 0.015; normal or KDE quantile | nothing |
| `two_gate` | ΔG_EnT ≤ 0 and asymmetric ΔG‡ ≤ t_PC(τ₀, r) | r from DFT sizes, r = 7 A | nothing |
| `marcus_barrier` | symmetric ΔG‡ < threshold (the plain-λ reference) | — | threshold |

- **Two-gate.** t_PC is the barrier at which k·τ₀ = K_SV,crit in the exergonic limit of the
  full scheme (see Physics); it depends on the photocatalyst through τ₀ and on the pair
  through the contact distance r. The reported variant takes r = r_PC + r_Q from the DFT
  molecular sizes; the variant `r = 7 A` takes the same r for every pair.
- **DvTE.** The reported variant is p = 0.008 with the normal quantile; the others are
  scored for comparison.
- **Grids.** Barrier threshold = 0.1–20 kcal/mol in steps of 0.1. K_SV,crit = 10 M⁻¹.
- **Choice.** The setting with the highest MCC on the DFT parameters; among tied settings
  the middle one in grid order.
- **ML.** `adiabatic` and `two_gate` are scored with each source directly. `marcus_barrier`
  is scored again with the ML quencher parameters, at the threshold chosen on DFT. `dvte`
  has no ML counterpart.
- **Metrics.** Confusion counts, accuracy, precision, recall (sensitivity), specificity,
  F1, MCC, and ROC AUC on each criterion's continuous score (−ΔG_EnT, E_T(PC) − DvTE,
  t_PC − ΔG‡ with every endergonic pair ranked below every exergonic one, −ΔG‡). ROC AUC within photocatalyst over the photocatalysts with both classes:
  `AUC_within_PC_median`, the median of the per-photocatalyst AUCs, and
  `AUC_within_PC_pooled`, every quenching pair compared only with the non-quenching pairs
  of its own photocatalyst. Written with three decimals.
- **Columns.** `parameters` is the source of the quencher parameters, `dft` or `ml`.
  `n_free` counts the parameters chosen on the data (the barrier threshold), and
  `n_settings` the grid points that choice was made from; it is 0 on `ml` rows, which take the setting of their `dft` row.

## Physics

Energies in kcal/mol, T = 298.15 K.

- **Pair reorganization energies.** Each partner contributes the surface it ends on, at the
  geometry it starts from:
  λ_f = λ_S0_T1(PC) + λ_T1_S0(Q) and λ_r = λ_T1_S0(PC) + λ_S0_T1(Q), as computed or
  predicted; no scale factor.
- **Marcus barrier, symmetric.** (ΔG + λ)² / 4λ with λ = √(λ_f·λ_r).
- **Marcus barrier, asymmetric.** Crossing of λ_r·x² and ΔG + λ_f·(x − 1)²; the symmetric
  form where they do not cross.
- **Rate, full scheme.** k = k_d / (1 + exp(ΔG/RT) + (k_−d/ν)·exp(ΔG‡/RT)), with
  k_d = 2 × 10¹⁰ M⁻¹ s⁻¹ (acetonitrile) and k_−d/ν = (k_d/K_A)/(k_BT/h) at the contact
  distance r of each pair, as in the lifetime barrier below.
- **Rate, irreversible.** k = k_d / (1 + (k_d/Z)·exp(ΔG‡/RT)), k_d/Z = 0.2.
- **Sandros.** k = k_d / (1 + exp(ΔG/RT)).
- **Lifetime barrier (two-gate).** t_PC = RT·ln((k_d·τ₀/K_SV,crit − 1) / (k_−d/ν)): the full
  scheme without exp(ΔG/RT), solved for k·τ₀ = K_SV,crit. Here k_−d/ν = (k_d/K_A)/(k_BT/h),
  with the Eigen encounter constant K_A = 4π·N_A·r³/3 at the contact distance r.
- **Contact distance.** r = r_PC + r_Q, each the radius of the sphere with the molecule's
  van der Waals volume: the union of atomic spheres with Bondi radii (Ir and Ru 2.00 Å)
  around the S0 DFT geometry, counted on a 0.1 Å grid. The radii are a data product
  (`../data/molecular_size.py`, written by the DFT extraction scripts and carried into
  the dataset); `pairs.py` adds them. The variant `r = 7 A` uses 7 Å for every pair.

`two_gate` uses the asymmetric Marcus barrier and the lifetime barrier; `marcus_barrier`
the symmetric Marcus barrier. The rate expressions appear only in
`further_analysis/rate_predictions.py`. Reorganization energies are named
`lambda_<surface>_<geometry>`, as in `../data/`.

## Cost

`cost.py` sums the CPU hours of the calculations each criterion needs per quencher;
photocatalyst calculations are not counted. Outputs in `results/cost/`:
- `per_quencher.csv`: CPU hours per quencher and group, per molecule and per atom
  (hydrogens included, counted from the SMILES).
- `statistics.csv`: min, max, mean, standard deviation and median over the quenchers.
- `dataset.csv`: totals over all quenchers, and total CPU hours per atom.

| group | calculations | criteria |
|---|---|---|
| adiabatic DFT | S0 and T1 opt+freq, S0 and T1 single points | `adiabatic` (dft) |
| four-point DFT | the above and both vertical spin-change single points | `two_gate`, `marcus_barrier` (dft) |
| DvTE | Gaussian opt+freq, MD trajectories, vertical-gap single points; steps that failed or hit the time limit are not counted | `dvte` |
| ML, per call / batch | model inference of one quencher per call, or its share of one call for all quenchers | every `ml` row |

DFT and DvTE: wall time × cores from `quencher_timings.csv` (MD trajectories from the Slurm
reports). ML: process CPU time from `quencher_timings_ml.csv`, measured on one machine with
the deployed model. Written with four significant figures.

## Further analysis

Each script writes to `further_analysis/results/<script>/`; the scripts in `labels/`
write to `further_analysis/results/labels/noise/`, `robust/` and `linear/`. Besides the
dataset, some read `../data/computational_data_processed/quencher_geometry_S0_T1.csv`,
`molecular_extent.csv` or `reorganization_modes.csv`, the Stern–Volmer tables in
`../data/experimental_data_processed/`, or the quencher classes written by
`quencher_classes.py`, which therefore runs first; each docstring names its inputs.

| script | contents |
|---|---|
| `influence.py` | metrics with each quencher or photocatalyst removed, at the benchmark settings |
| `held_out.py` | leave-one-quencher-out and leave-one-photocatalyst-out scores of the tuned criteria |
| `within_groups.py` | metrics and AUC within each photocatalyst and within each quencher |
| `adiabatic_variants.py` | margin E_T(PC) ≥ E_T(Q) + m, m = 0–3 kcal/mol; cutoff on the ML uncertainty of E_T(Q) |
| `barrier_variants.py` | barrier criteria for every barrier model, with and without ΔG_EnT ≤ 0, and with a threshold of 5 kcal/mol |
| `rate_predictions.py` | log₁₀ k for every pair and source: full scheme at r_PC + r_Q and at 7 Å, irreversible, Sandros; parity with the measured k_q, and of a constant predictor; nothing fitted |
| `quencher_classes.py` | the 48 quenchers as rigid, flexible or dissociative, from structure and the S0→T1 geometry change (`quencher_geometry_S0_T1.csv`); λ_T1_S0 = 20 kcal/mol as a cross-check; ambiguous assignments flagged |
| `metrics_extended.py` | per-pair table for the figures; full metrics of every criterion (PR AUC, within-PC, LOQO/LOPCO, PC-only baseline, plain-λ), confusion counts, metrics per quencher class including flexible and dissociative |
| `bootstrap.py` | two-way cluster bootstrap over photocatalysts and quenchers (2000 replicates, fixed seed): intervals of MCC, ROC AUC and within-PC AUC; paired differences with P(Δ > 0); ML − DFT |
| `lifetimes.py` | quenching fraction by barrier bin for τ₀ < 1 µs and ≥ 1 µs; short-lived photocatalysts; the criterion at 7 Å against one constant threshold; DvTE false positives rejected, by gate and quencher class |
| `figure_tables.py` | per-quencher and per-photocatalyst tables of the scored pairs; cost per quencher of each method over the 46 scored quenchers with the metrics it enables; t_PC against τ₀ (median, min and max contact distance, 7 Å, k_−d/ν = 0.25) |
| `contact_distance.py` | the criterion with r = r_PC + r_Q from van der Waals volumes or molecular extents, or fixed r = 4–12 Å; distribution of r; shift of t_PC per Å |
| `gate_offset.py` | first gate moved to ΔG ≤ c (DvTE − E_T(PC) ≤ c), c = −5…+2 kcal/mol: optimum, held-out c (nested LOPCO/LOQO), the window −1.5 < ΔG ≤ 0, changed pairs by quencher class, one photocatalyst left out |
| `error_propagation.py` | Monte Carlo errors in the computed parameters (per molecule, 1000 draws) through the classification and the predicted log₁₀ k_q; local sensitivity of log₁₀ k_q to E_T(Q) and λ |
| `mlj.py` | Marcus–Levich–Jortner rates from the mode-resolved λ (`reorganization_modes.csv`; split at 3 k_BT, one quantum mode): rate parity, MLJ criteria against two-gate (bootstrap ΔMCC), k_−d/ν = 0.25, one variant with tuned classical λ (LOQO/LOPCO) |
| `kinetics_series.py` | Spearman ρ of predicted against measured log₁₀ k_q within photocatalyst and quencher series (k_q > 0, ≥ 5 pairs); Q41 and Q2 across all photocatalysts |
| `exclusion.py` | Q38 and Q31: outcomes, ΔG, K_SV, I₀/I and gate rejections; K_SV against τ₀ and E_T(PC); metrics with and without each quencher |
| `anharmonicity.py` | skewness and tail of the DvTE gap distributions (trajectory bootstrap), skewness against the signed error, ΔG shifted by the tail, crossing position, torsional instead of harmonic curves |
| `dvte_reorganization.py` | DvTE − E_T against λ, four-point Gaussian estimate of DvTE, symmetric against asymmetric barrier, position of the Youden and Sandros boundaries, literature E_T(PC) (`../data/literature_benchmark/photocatalyst_triplet_energies.csv`) |
| `ml_vs_dft.py` | the criterion with ML against DFT quencher parameters: overlap with the training set, per-quencher deviations and change in correct predictions, parameter swaps, DFT fallback by uncertainty; needs RDKit |
| `kdet_threshold.py` | the criterion with each photocatalyst's K_det (floor 200 M⁻¹ for PC25–27) instead of K_SV,crit = 10 M⁻¹, under three label sets; t_PC per photocatalyst |
| `labels/s1_noise_model.py` … `s5_profile_g.py` | s1 noise model of the plate reads; s2 K_SV posteriors, empirical null and K_det per photocatalyst; s3 power of the concentration ladder; s4 label confidence, metrics under label uncertainty, noise in the rate parity, censored rate likelihood; s5 profile of its slope g |
| `labels/robust_fits.py`, `robust_labels.py` | K_SV from 13 estimators and labels from each; blank-well noise test, with CV₀ the MAD-SD of the blank wells, each divided by its plate's median blank |
| `labels/linear_shapes.py`, `linear_variants.py`, `linear_downstream.py` | shapes of the Stern–Volmer curves; linear-range and artifact variants, background, drift and failed wells; their metrics |

The label analyses never change the labels of the dataset; their variants enter the metrics
only.

## Literature

`literature/` compares the model's predictions with published studies. The published
tables are in `../data/literature_benchmark/`, one CSV per study. `predict.py` predicts E_T
and both reorganization energies for every structure the model can encode and writes
`results/literature/predictions_<case>.csv` and `predictions_members_<case>.csv`; the
comparison scripts read those files and never call the model. The literature benchmark is
not a stage of `reproduce.py`, and its scripts run with the `ml` extra:

```
uv run --extra ml python benchmark/literature/predict.py
uv run --extra ml python benchmark/literature/ma_tolerance.py
```

| script | study |
|---|---|
| `ma_tolerance.py` | Ma 2023: functional-group tolerance of the azaarene reduction |
| `sole_daura_ez.py` | Solé-Daura 2024: E/Z isomerization |
| `gilmour_ez.py` | Gilmour 2020: β-boryl acrylates |
| `leonori_ranking.py` | Leonori 2025: nitroarene photocatalysts |
| `bach_allene.py` | Bach 2026: allene amides |
| `booker_milburn_sensitisers.py` | Booker-Milburn 2020: thioxanthone sensitizers |
| `paton_dvte.py` | Paton 2024: DvTE against the predicted reorganization energies |

Each script's docstring names the tables it writes.
