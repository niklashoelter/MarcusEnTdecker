# benchmark

Physical-organic criteria for triplet energy transfer, scored against the Stern–Volmer
screen: can they tell which photocatalyst–quencher pairs quench, and at what
computational cost?

```
uv run reproduce.py benchmark               classification and cost
uv run reproduce.py further_analysis        every script in further_analysis/
uv run benchmark/classification.py          quenches or not
uv run benchmark/cost.py                    CPU time per quencher
```

Inputs: `../data/exp_dataset_ent.csv`; `cost.py` also reads
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
├── results/
│   ├── classification/
│   │   ├── settings.csv    every DFT setting of every criterion
│   │   └── best.csv        best DFT setting per criterion, and the same setting with ML
│   └── cost/
└── further_analysis/   variants and diagnostics beyond the benchmark table
    └── results/
```

## The pairs

`exp_dataset_ent` is built in `../data/` by `build_exp_dataset_ent.py`.
- **Non-quenching pairs:** all kept.
- **Quenching pairs:** kept only if every single-electron-transfer channel with a
  measured quencher potential is more than 0.25 V uphill.

`pairs.py` leaves out PC22 on loading, because its emission is too weak for Stern–Volmer
analysis, and scores a pair only if every input of every criterion is present, so all
criteria are scored on the same pairs.

## Classification

The ground truth is `quenches`. Photocatalyst properties are DFT throughout.

| criterion | predicts quenching when | variants | chosen on the data |
|---|---|---|---|
| `adiabatic` | ΔG_EnT ≤ 0 | — | nothing |
| `dvte` | E_T(PC) ≥ DvTE(p) | p = 0.005, 0.008, 0.01, 0.015; normal or KDE quantile | nothing |
| `marcus_barrier` | ΔG‡ < threshold | `lambda raw`, `lambda scaled` | threshold; α (scaled) |
| `marcus_rate` | k·τ₀ ≥ K_SV,crit | `lambda raw`, `lambda scaled` | α (scaled) |
| `rehm_weller_rate` | k·τ₀ ≥ K_SV,crit | `lambda raw`, `lambda scaled` | α (scaled) |

- **Grids.** α = 0.05–1.00 in steps of 0.05; barrier threshold = 0.1–20 kcal/mol in steps
  of 0.1. K_SV,crit = 10 M⁻¹. `lambda raw` is α = 1.
- **Choice.** The setting with the highest MCC on the DFT parameters; among tied settings
  the middle one in grid order.
- **ML.** The barrier and rate criteria are scored again with the ML quencher
  parameters, at the setting chosen on DFT. `adiabatic` is scored with each source
  directly; `dvte` has no ML counterpart.
- **Metrics.** Confusion counts, accuracy, precision, recall (sensitivity), specificity,
  F1, MCC, and ROC AUC on each criterion's continuous score (−ΔG_EnT, E_T(PC) − DvTE,
  −ΔG‡, log₁₀ k·τ₀). Written with three decimals.
- **Columns.** `parameters` is the source of the quencher parameters, `dft` or `ml`.
  `n_free` counts the parameters chosen on the data (α, the barrier threshold), and
  `n_settings` the grid points that choice was made from; it is 0 on `ml` rows, which take the setting of their `dft` row.

## Physics

Energies in kcal/mol, T = 298.15 K.

- **Pair reorganization energies.** Each partner contributes the surface it ends on, at the
  geometry it starts from:
  λ_f = λ_S0_T1(PC) + λ_T1_S0(Q) and λ_r = λ_T1_S0(PC) + λ_S0_T1(Q). α multiplies both.
- **Marcus barrier, symmetric.** (ΔG + λ)² / 4λ with λ = √(λ_f·λ_r).
- **Marcus barrier, asymmetric.** Crossing of λ_r·x² and ΔG + λ_f·(x − 1)²; the symmetric
  form where they do not cross.
- **Rehm–Weller barrier.** ΔG/2 + √((ΔG/2)² + (λ/4)²) with λ = √(λ_f·λ_r).
- **Rate, full scheme.** k = k_d / (1 + exp(ΔG/RT) + (k_−d/ν)·exp(ΔG‡/RT)), with
  k_d = 2 × 10¹⁰ M⁻¹ s⁻¹ (acetonitrile) and k_−d/ν = 0.25.
- **Rate, irreversible.** k = k_d / (1 + (k_d/Z)·exp(ΔG‡/RT)), k_d/Z = 0.2.
- **Sandros.** k = k_d / (1 + exp(ΔG/RT)).

The benchmark criteria use the symmetric Marcus or the Rehm–Weller barrier with the full
scheme; the asymmetric barrier, the irreversible rate and Sandros appear only in
`further_analysis/`. Reorganization energies are named
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
| four-point DFT | the above and both vertical spin-change single points | barrier and rate criteria (dft) |
| DvTE | Gaussian opt+freq, MD trajectories, vertical-gap single points including failed and time-limited steps | `dvte` |
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
| `rate_variants.py` | rate criteria for every barrier model and both rate laws; MCC over α × K_SV,crit (1–1000 M⁻¹) |
| `rate_predictions.py` | log₁₀ k for every pair, source, rate law, barrier model and α; nothing fitted |
