# data

Measurements and calculations for the photocatalysts and quenchers. Raw instrument files
and ORCA jobs are turned into one property workbook per compound class. The experimental
workbooks become the pair-level dataset `exp_dataset.csv`. Its energy-transfer subset,
with the calculated properties joined on, is `exp_dataset_ent.csv`, the input to
`../benchmark/`.

```
data/
├── experimental_data_raw/           instrument files, plus the tables derived from them
│   ├── pc_emission/                 emission scans
│   ├── pc_tcspc/                    PC<n>.txt/.FL, one trace each -> lifetimes_tcspc.csv
│   ├── quencher_absorption/         absorption scans -> absorption_extinction.csv
│   ├── quencher_cyclic_voltammetry/ measured data/ (voltammograms), redox potentials in three tables
│   └── stern_volmer/                plate reads, pipetting maps -> plate_reader_filters.csv
├── experimental_data_processed/
│   ├── photocatalyst_properties_exp.xlsx
│   ├── quencher_properties_exp.xlsx
│   ├── stern_volmer_intensities.csv
│   └── stern_volmer_fits.csv
├── computational_data_raw/          ORCA/Milo jobs (Zenodo, not in git), plus derived tables
│   ├── photocatalysts/              PC_<n>/{S0,T1,ox,red,conf}/ -> photocatalyst_dft.csv
│   ├── quencher_adiabatic_DFT/      Q<n>/{S0,T1,ox,red}/ -> quencher_dft.csv
│   └── quencher_adiabatic_DvTE/     MD trajectories -> quencher_dvte.csv, _samples.csv.gz,
│                                    quencher_timings_vertical_sp.csv.gz
├── computational_data_processed/
│   ├── photocatalyst_properties_calc.xlsx
│   ├── quencher_properties_calc.xlsx
│   ├── quencher_properties_ml.xlsx  the same three quantities, predicted from the SMILES
│   ├── quencher_timings.csv         wall time, cores and CPU hours of every quencher job
│   └── quencher_timings_ml.csv      the model's inference time per quencher
├── experimental_data_scripts/       one script per raw folder, plus properties_exp.py
├── computational_data_scripts/      one script per raw folder, plus properties_calc.py,
│                                    quencher_timings.py, quencher_ml.py,
│                                    quencher_ml_timing.py and photocatalyst_curate.py
├── compound_identification.py/.xlsx compound ids, names, SMILES, old -> new PC ids
├── properties_workbook.py           layout shared by the workbooks
├── build_exp_dataset.py             experimental processed data -> one row per pair
├── exp_dataset.csv, .xlsx           the pair-level experimental dataset
├── build_exp_dataset_ent.py         exp_dataset -> energy-transfer subset + properties
└── exp_dataset_ent.csv, .xlsx       the input to ../benchmark/
```

## Pipeline

1. Each extraction script reads one raw folder and writes its table into the same folder.
   The two Stern–Volmer scripts write to `experimental_data_processed/`.
2. `properties_exp.py` and `properties_calc.py` copy the per-compound values into the
   workbooks.
3. `build_exp_dataset.py` builds `exp_dataset` from the experimental processed folder.
4. `build_exp_dataset_ent.py` builds `exp_dataset_ent` from `exp_dataset` and the
   property workbooks.

`uv run reproduce.py data` runs all of this in order, including `quencher_timings.py`,
which writes the computational cost of the quencher jobs to
`computational_data_processed/`. Three scripts are not part of it:

- `quencher_ml.py` needs the model environment and is run by hand:
  `uv sync --extra ml`, then
  `uv run --extra ml python data/computational_data_scripts/quencher_ml.py`.
  Its workbook is in git.
- `quencher_ml_timing.py` measures the model's inference time on the machine it runs on,
  in the same environment:
  `uv run --extra ml python data/computational_data_scripts/quencher_ml_timing.py`.
  Its table is in git.
- `photocatalyst_curate.py` builds the deposited photocatalyst folders from a job tree.
  It is not a pipeline stage.

## Photocatalyst ids

Raw files and extraction scripts use the old id from the screen (`PC10_QBatch1.csv`,
`pc_tcspc/PC4.txt`). New ids, PC1–PC27, enter at the processed stage: the photocatalyst
workbooks carry `old_id` and `new_id`, `exp_dataset` carries `pc_old_id` and `pc_new_id`,
and everything downstream uses the new id. Old PC7, PC26, PC30 and PC31 are not in the
data.

## Compound identity

`compound_identification.xlsx` has one sheet per compound class with ids, photocatalyst
old/new ids, names, quencher SMILES and molar masses. All scripts read it through
`compound_identification.py`. Quencher names are derived from the SMILES. The pipetting
maps use their own shorthand labels, which are not read.

## Workbooks

One row per compound, sorted by new id (PC1–27) or quencher id (Q1–48). Columns are
grouped in blocks (identity first, then one block per property), separated by an empty
column. Row 1 names the blocks; row 2 holds the headers as `quantity / unit`. The
`Column key` sheet gives each column's description and source table.

## exp_dataset

One row per photocatalyst–quencher pair, 27 × 48 = 1296, sorted by `pc_new_id` then
`quencher_id`. Ten columns, all measured: the ids, K_SV, τ0, k_q = K_SV/τ0, `quenches`
and `censored`. The xlsx holds the same table in the workbook
layout.

PC22 (old PC24) is included although its emission was too weak; the theory benchmark
drops it.

`quenches` is 1 if the Stern–Volmer fit meets any one of these thresholds:
K_SV > 50 at r² > 0.75, > 20 at 0.80, > 10 at 0.90, > 0 at 0.99.
`stern_volmer_fits.csv` records which one applied.
Pairs that do not quench have `censored = 1`: their true K_SV is below what was
detectable, so their k_q is an upper bound.

## exp_dataset_ent

The pairs of `exp_dataset` on which quenching can be attributed to energy transfer.

- **Non-quenching pairs:** all kept, whatever the potentials.
- **Quenching pairs:** kept only if single-electron transfer is ruled out, meaning every
  channel with a measured quencher potential is more than **0.25 V** uphill:
  - ΔE_SET,ox = E_ox(Q) − E(PC\*/PC⁻), oxidation of the quencher
  - ΔE_SET,red = E(PC⁺/PC\*) − E_red(Q), reduction of the quencher
- **Potentials:** quencher potentials are measured, photocatalyst excited-state
  potentials are calculated.
- **Limits:** a measured limit (E_ox > x, E_red < x) makes ΔE a lower bound, marked in
  `dE_SET_ox_bound` / `dE_SET_red_bound`. It rules a channel out only if the limit itself
  is past 0.25 V.

Besides the ten `exp_dataset` columns and the two ΔE_SET columns, each row carries:
- **Photocatalyst (DFT):** E_T and both λ.
- **Quencher (DFT):** E_T and both λ.
- **Quencher (DvTE):** mean and standard deviation of the vertical gaps, and their
  p-quantiles for p = 0.005, 0.008, 0.01, 0.015, from a normal distribution and from a KDE.
- **Quencher (ML):** E_T and both λ, each with the model's uncertainty. The model covers
  organic molecules only, so there are no predicted photocatalyst values.
- **Driving force:** ΔG_EnT = E_T(Q) − E_T(PC), from both quencher sources.

Blanks stay blank.

## Methods notes

**Stern–Volmer wavelengths** are the filter pairs in the plate-reader export headers,
collected with their bandwidths in `stern_volmer/plate_reader_filters.csv`. Old PC1 and
PC3 were read with saved protocols that name no filters; their wavelengths come from the
screen's photocatalyst sheet.

**Quencher redox potentials** are read by hand from the voltammograms, or taken from the
literature. Each quencher is entered in one of two files, `quencher_redox_measured.csv`
(source `CV`) or `quencher_redox_literature.csv` (a DOI, or `estimated`), and
`quencher_redox_potentials.py` joins them into `quencher_redox_potentials.csv`. Each row
has a bound column (`>` or `<` where the value is a limit) and a free-text `comment`.

**Photocatalyst lifetimes** come from one TCSPC trace per photocatalyst. τ0 and its error
are the value and standard error of a single-exponential fit, started after the pulse.
Old PC4 is fitted from 1500 ns to skip a fast early component (`FIT_START_NS` in
`lifetimes_tcspc.py`).

**Reorganization energies** are named `lambda_<surface>_<geometry>`: the energy of one
state at another state's optimized geometry, minus that state at its own geometry.
`lambda_S0_T1` is E_S0 at the T1 geometry minus E_S0. `Dplus` and `Dminus` are the
oxidized and reduced states (`ox` and `red` job folders). Electronic energies only, no
thermal corrections.

**Predicted quencher properties.** `quencher_properties_ml.xlsx` holds E_T, lambda_S0_T1
and lambda_T1_S0 predicted from the SMILES by the graph mixture density network in
`../ml/data/model_final.pt`, each with its predicted uncertainty.

**Computational cost.** `quencher_timings.csv` has one row per quencher job: DFT
(ORCA), the Gaussian optimisation and frequencies, the Milo trajectories, and the
vertical single points summed per trajectory and multiplicity (one row per single point
in `quencher_timings_vertical_sp.csv.gz`). Wall times come from the program outputs;
CPU hours are wall time × cores. The Milo trajectories kept no Gaussian logs and are
timed by their Slurm reports, as are the vertical single-point steps whose ORCA runs
failed or hit the time limit (`dvte sp failed`). Where a Slurm report exists next to a
program output it is recorded as a cross-check and disagreements are flagged.
`quencher_timings_ml.csv` holds the model's import and load times and the median time of
one prediction per quencher and of all 48 in one call, on the CPU; the machine and
versions are in its `#` header lines.

**Calculations.** The computational folders hold one calculation per state:
`Q<n>/<state>/` and `PC_<n>/<state>/` contain one geometry optimisation and its single
points, named `opt`, `sp` and `sp_vert_<kind>`. For quenchers, the lowest-energy
conformer is kept. In Q17 S0 the thermal correction, and in Q18 S0 the reduced-state
vertical single point, come from a second run of the same conformer, so there
`opt_output.out` is not the run behind `sp_output.out`.

## Git and Zenodo

From `computational_data_raw/`, only the derived tables are in git. The ORCA/Milo job
folders are on Zenodo.
