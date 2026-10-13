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
├── tests/              pytest tests of theory, scoring, classification and cost
├── literature/         the literature benchmark: predict.py, common.py, one script per study
├── results/
│   ├── classification/
│   │   ├── settings.csv    every DFT setting of every criterion
│   │   └── best.csv        best DFT setting per criterion, and the same setting with ML
│   ├── cost/
│   └── literature/     predictions and comparison tables of the literature benchmark
└── further_analysis/   variants and diagnostics beyond the benchmark table
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
| `marcus_barrier` | ΔG‡ < threshold | — | threshold |

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
- **Rehm–Weller barrier.** ΔG/2 + √((ΔG/2)² + (λ/4)²) with λ = √(λ_f·λ_r).
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

Each script writes to `further_analysis/results/<script>/`.

| script | contents |
|---|---|
| `influence.py` | metrics with each quencher or photocatalyst removed, at the benchmark settings |
| `held_out.py` | leave-one-quencher-out and leave-one-photocatalyst-out scores of the tuned criteria |
| `within_groups.py` | metrics and AUC within each photocatalyst and within each quencher |
| `adiabatic_variants.py` | margin E_T(PC) ≥ E_T(Q) + m, m = 0–3 kcal/mol; cutoff on the ML uncertainty of E_T(Q) |
| `barrier_variants.py` | barrier criteria for every barrier model, with and without ΔG_EnT ≤ 0, and with a threshold of 5 kcal/mol |
| `rate_predictions.py` | log₁₀ k for every pair, source, rate law and barrier model; nothing fitted |

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
