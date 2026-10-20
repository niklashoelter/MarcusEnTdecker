# benchmark

Physical-organic criteria for triplet energy transfer, scored against the Stern–Volmer
screen, and their computational cost.

```
uv run reproduce.py benchmark               classification and cost
uv run reproduce.py further_analysis        every script in further_analysis/
uv run benchmark/classification.py          quenches or not
uv run benchmark/cost.py                    CPU time per quencher
uv run --extra ml python benchmark/literature/predict.py      literature predictions
uv run --extra ml python benchmark/literature/<script>.py     one literature comparison
```

Inputs: `../data/exp_dataset_ent.csv` (including the molecular radii `pc_radius_dft`,
`q_radius_dft` from the S0 DFT geometries); `cost.py` also reads
`../data/computational_data_processed/quencher_timings.csv`, `quencher_timings_ml.csv`
and the quencher SMILES from `../data/compound_identification.xlsx`.

## Layout

```
benchmark/
├── pairs.py            loads the dataset and names its columns; which pairs are scored
├── theory.py           barrier and rate expressions
├── scoring.py          metrics, AUC, choice of the best setting
├── classification.py   the classification criteria
├── cost.py             CPU time per quencher for each criterion
├── literature/         the literature benchmark: predict.py, common.py, one script per study
├── results/
│   ├── classification/
│   │   ├── settings.csv    every DFT setting of every criterion
│   │   └── best.csv        best DFT setting per criterion, and the same setting with ML
│   ├── cost/
│   ├── entdecker_1/    E_T, timings and adiabatic scores of EnTdecker 1.0, results only
│   └── literature/     predictions and comparison tables of the literature benchmark
└── further_analysis/   variants and diagnostics beyond the benchmark table
    └── results/
```

## The pairs

`exp_dataset_ent` is built in `../data/` by `build_exp_dataset_ent.py`.
- **Non-quenching pairs:** all kept.
- **Quenching pairs:** kept only if every single-electron-transfer channel with a
  measured quencher potential is more than 0.25 V uphill.

`pairs.py` leaves out PC22 and the quenchers Q31 and Q38 on loading (excluded, see SI)
and scores a pair only if every input of every criterion is present.

## Classification

Ground truth: `quenches`. Photocatalyst properties are DFT throughout.

| criterion | predicts quenching when | variants | chosen on the data |
|---|---|---|---|
| `adiabatic` | ΔG_EnT ≤ 0 | — | nothing |
| `dvte` | E_T(PC) ≥ DvTE(p) | p = 0.005, 0.008, 0.01, 0.015; normal or KDE quantile | nothing |
| `two_gate` | ΔG_EnT ≤ 0 and asymmetric ΔG‡ ≤ t_PC(τ₀, r) | r = r_PC + r_Q from DFT sizes, r = 7 A | nothing |

- **Two-gate.** t_PC is the barrier at which k·τ₀ = K_SV,crit in the exergonic limit of the
  full scheme (see Physics).
- **Detection threshold.** K_SV,crit = 10 M⁻¹.
- **Settings.** The criteria use fixed physical settings; none is fitted to the labels.
- **ML.** `adiabatic` and `two_gate` are scored with each source directly; `dvte` has no ML row.
- **Metrics.** Confusion counts, accuracy, precision, recall (sensitivity), specificity,
  F1, MCC, and ROC AUC on the continuous score (−ΔG_EnT, E_T(PC) − DvTE, t_PC − ΔG‡ with
  endergonic pairs ranked last). Within photocatalysts with both classes:
  `AUC_within_PC_median` (median of per-photocatalyst AUCs) and `AUC_within_PC_pooled`
  (each quenching pair compared only with non-quenching pairs of its photocatalyst).
  Three decimals.
- **Columns.** `parameters`: source of the quencher parameters (`dft`, `ml`).
  `n_free` is 0: none of the settings is fitted to the labels. `n_settings` is 1 for
  each fixed variant, with each source evaluated independently.

## Physics

Energies in kcal/mol, T = 298.15 K.

- **Pair reorganization energies.** λ_f = λ_T1_S0(PC) + λ_S0_T1(Q),
  λ_r = λ_S0_T1(PC) + λ_T1_S0(Q); unscaled.
- **Marcus barrier, symmetric.** (ΔG + λ)² / 4λ with λ = √(λ_f·λ_r).
- **Marcus barrier, asymmetric.** Crossing of λ_f·x² and ΔG + λ_r·(x − 1)²; the symmetric
  form where they do not cross.
- **Rate, full scheme.** k = k_d / (1 + exp(ΔG/RT) + (k_−d/ν)·exp(ΔG‡/RT)), with
  k_d = 2 × 10¹⁰ M⁻¹ s⁻¹ (acetonitrile) and k_−d/ν = (k_d/K_A)/(k_BT/h) at the contact
  distance r.
- **Lifetime barrier (two-gate).** t_PC = RT·ln((k_d·τ₀/K_SV,crit − 1) / (k_−d/ν)): the full
  scheme without exp(ΔG/RT), solved for k·τ₀ = K_SV,crit, with the Eigen encounter constant
  K_A = 4π·N_A·r³/3.
- **Contact distance.** r = r_PC + r_Q, each the radius of the sphere with the molecule's
  van der Waals volume (union of Bondi-radius spheres, Ir and Ru 2.00 Å, around the S0 DFT
  geometry, 0.1 Å grid; `../data/molecular_size.py`). Variant `r = 7 A`: 7 Å for every pair.

`two_gate` uses the asymmetric Marcus barrier and the lifetime barrier.
`further_analysis/rate_predictions.py` compares the full-scheme rates with measured rates.
Reorganization energies are named
`lambda_<surface>_<geometry>`, as in `../data/`.

## Cost

`cost.py` sums the CPU hours of the calculations each criterion needs per quencher;
photocatalyst calculations are not counted. Outputs in `results/cost/`:
- `per_quencher.csv`: CPU hours per quencher and group, per molecule and per atom
  (hydrogens included, from the SMILES).
- `statistics.csv`: min, max, mean, standard deviation and median over the quenchers.
- `dataset.csv`: totals over all quenchers, and total CPU hours per atom.

| group | calculations | criteria |
|---|---|---|
| adiabatic DFT | S0 and T1 opt+freq, S0 and T1 single points | `adiabatic` (dft) |
| four-point DFT | the above and both vertical spin-change single points | `two_gate` (dft) |
| DvTE | Gaussian opt+freq, MD trajectories, vertical-gap single points; failed or timed-out steps not counted | `dvte` |
| ML, per call / batch | inference of one quencher per call, or its share of one call for all quenchers | every `ml` row |

DFT and DvTE: wall time × cores from `quencher_timings.csv` (MD trajectories from the Slurm
reports). ML: process CPU time from `quencher_timings_ml.csv` (deployed model). Four
significant figures.

## EnTdecker 1.0

`results/entdecker_1/` holds the results of the two models of EnTdecker 1.0 (Schlosser et
al., JACS 2024) on the benchmark quenchers; the model run is not part of the repository:
- `quencher_E_T.csv`: E_T of every quencher from the triplet-energy model, and per model
  whether it processed the quencher, with the reason where not (`triplet_note`,
  `spin_note`). A spin result counts as processed only if the decoded SMILES is the input.
- `quencher_timings.csv`: inference timings in the columns of
  `../data/computational_data_processed/quencher_timings_ml.csv`, with a leading `model`
  column: `triplet`, `spin`, and `sum` (row-wise sum). The header lines give the machine,
  package versions, checkpoints and code commits.
- `adiabatic.csv`: the `adiabatic` criterion with these quencher E_T, in the columns of
  `classification/best.csv` (`parameters` = `entdecker`), written by `entdecker_1.py`.

**Models.** The published checkpoints from Zenodo (10.5281/zenodo.10391170), with the code of
github.com/le-schlo/EnTdecker: `triplet_energy.pt`, the chemprop D-MPNN, through the
prediction path of `Models/triplet_energy/chemprop/eval.py`; `spin_population.pt`, the
transformer for the spin populations (EasyChemML), through the model and tokenizer calls of
`Models/spin_population/eval.py` with greedy decoding, on the CPU.

**Timing.** As for `quencher_timings_ml.csv`: import, checkpoint load and first call once,
then each quencher in its own call and all quenchers in one call, with the model loaded;
process CPU time over all threads, including chemprop's predictions-file write. Each
model in its own process; the header gives the protocol of each.

**Scoring.** `entdecker_1.py` (a `reproduce.py` stage after `classification.py`) forms
ΔG_EnT = E_T(Q) − E_T(PC, DFT), rounded to 2 decimals, and scores it with
`classification.py` on the pairs `pairs.py` scores.

## Further analysis

Each script writes to `further_analysis/results/<script>/`. Besides the dataset, some read
`../data/computational_data_processed/quencher_geometry_S0_T1.csv`, the Stern–Volmer tables in
`../data/experimental_data_processed/`, or the quencher classes written by
`quencher_classes.py` (runs first); each docstring names its inputs.

| script | contents |
|---|---|
| `rate_predictions.py` | log₁₀ k for every pair and source: full scheme at r_PC + r_Q and at 7 Å; parity with the measured k_q, and of a constant predictor |
| `quencher_classes.py` | quenchers as rigid, flexible or dissociative, from structure and the S0→T1 geometry change (`quencher_geometry_S0_T1.csv`); λ_T1_S0 = 20 kcal/mol cross-check; ambiguous assignments flagged |
| `metrics_extended.py` | per-pair table for the figures; full metrics of every criterion (PR AUC, within-PC), metrics per quencher class |
| `lifetimes.py` | outcomes and false positives for τ₀ < 1 µs and ≥ 1 µs; pairs rejected by the barrier gate |
| `figure_tables.py` | per-quencher and per-photocatalyst tables of the scored pairs; cost per quencher of each method over the scored quenchers with its metrics |
| `contact_distance.py` | the criterion with r = r_PC + r_Q from van der Waals volumes, or fixed r = 4–12 Å; distribution of r; shift of t_PC per Å |
| `gate_offset.py` | first gate moved to ΔG ≤ c (DvTE − E_T(PC) ≤ c), c = −5…+2 kcal/mol: MCC sweeps, optimum and quenching fractions in driving-force windows |
| `error_propagation.py` | Monte Carlo errors in the computed parameters (per molecule, 1000 draws) through the classification and the predicted log₁₀ k_q; local sensitivity of log₁₀ k_q to E_T(Q) and λ |
| `exclusion.py` | Q38 and Q31: outcomes, ΔG, K_SV, I₀/I and gate rejections |
| `anharmonicity.py` | skewness and tail of the DvTE gap distributions (trajectory bootstrap), skewness against the signed error, ΔG shifted by the tail, crossing position, torsional instead of harmonic curves |
| `dvte_reorganization.py` | DvTE − E_T against λ, symmetric against asymmetric barrier, driving-force bins and removal of the energy gate |
| `ml_vs_dft.py` | the criterion with ML against DFT quencher parameters: per-quencher deviations and change in correct predictions |

## Literature

`literature/` compares the model's predictions with published studies. Published tables:
`../data/literature_benchmark/`, one CSV per study. `predict.py` predicts E_T and both
reorganization energies for every structure the model can encode and writes
`results/literature/predictions_<case>.csv`; the
comparison scripts read those files and never call the model. Not a stage of
`reproduce.py`; run with the `ml` extra:

```
uv run --extra ml python benchmark/literature/predict.py
uv run --extra ml python benchmark/literature/ma_tolerance.py
```

| script | study |
|---|---|
| `ma_tolerance.py` | Ma 2023: functional-group tolerance of the azaarene reduction |
| `sole_daura_ez.py` | Solé-Daura 2024: E/Z isomerization |
| `sole_daura_tx.py` | Solé-Daura 2024: the same with thioxanthone from one source at a time (paper DFT, our OPC library DFT, our prediction) |
| `leonori_ranking.py` | Leonori 2025: nitroarene photocatalysts |
| `bach_allene.py` | Bach 2026: allene amides |
| `paton_dvte.py` | Paton 2024: DvTE against the predicted reorganization energies |

Each script's docstring names the tables it writes.
