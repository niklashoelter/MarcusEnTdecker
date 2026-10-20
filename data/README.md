# data

Measurements and calculations for the photocatalysts and quenchers: raw instrument files
and ORCA jobs -> one property workbook per compound class -> the pair-level dataset
`exp_dataset.csv` -> its energy-transfer subset `exp_dataset_ent.csv`, the input to
`../benchmark/`.

```
data/
├── experimental_data_raw/           instrument files (Zenodo, not in git)
│   ├── pc_emission/                 emission scans
│   ├── pc_tcspc/                    PC<old id>.txt/.FL, one trace each
│   ├── quencher_absorption/         absorption scans, plate map
│   ├── quencher_cyclic_voltammetry/ measured data/ (voltammograms)
│   └── stern_volmer/                plate reads, pipetting maps
├── experimental_data_processed/
│   ├── lifetimes_tcspc.csv          from pc_tcspc/
│   ├── plate_reader_filters.csv     from stern_volmer/
│   ├── quencher_redox_measured.csv, quencher_redox_literature.csv
│   │                                redox potentials, entered by hand
│   ├── quencher_redox_potentials.csv  the two joined
│   ├── photocatalyst_properties_exp.xlsx
│   ├── quencher_properties_exp.xlsx
│   ├── stern_volmer_intensities.csv
│   └── stern_volmer_fits.csv
├── computational_data_raw/          ORCA/Milo jobs (Zenodo, not in git)
│   ├── photocatalysts_new_ids/      PC<n>/{S0,T1,ox,red,conf}/
│   ├── organophotocatalysts/        OPC<n>/{S0,T1}/
│   ├── quencher_adiabatic_DFT/      Q<n>/{S0,T1,ox,red}/
│   └── quencher_adiabatic_DvTE/     MD trajectories, vertical single points
├── computational_data_processed/
│   ├── photocatalyst_dft.csv        from photocatalysts_new_ids/
│   ├── organophotocatalyst_dft.csv  from organophotocatalysts/
│   ├── quencher_dft.csv             from quencher_adiabatic_DFT/
│   ├── quencher_dvte.csv, quencher_dvte_samples.csv.gz
│   │                                from quencher_adiabatic_DvTE/
│   ├── quencher_timings_vertical_sp.csv.gz  timing of every DvTE single point
│   ├── photocatalyst_properties_calc.xlsx
│   ├── quencher_properties_calc.xlsx
│   ├── quencher_properties_ml.xlsx  E_T, lambda_S0_T1 and lambda_T1_S0 predicted from the SMILES
│   ├── quencher_timings.csv         wall time, cores and CPU hours of every quencher job
│   ├── quencher_timings_ml.csv      the deployed model's inference time per quencher
│   └── quencher_geometry_S0_T1.csv  change of each quencher geometry from S0 to T1
├── experimental_data_scripts/       one script per raw folder (none for pc_emission/),
│                                    plus properties_exp.py
├── computational_data_scripts/      one script per raw folder, plus properties_calc.py,
│                                    quencher_timings.py, quencher_ml.py,
│                                    quencher_ml_timing.py, photocatalyst_curate.py,
│                                    quencher_geometry.py
├── literature_benchmark/            published tables of the literature benchmark, one CSV
│                                    per study (source PDFs are not included)
├── compound_identification.py/.xlsx compound ids, names, SMILES, old -> new PC ids
├── properties_workbook.py           layout shared by the workbooks
├── molecular_size.py                molecular radius from a DFT geometry, used by the
│                                    quencher and photocatalyst extraction scripts
├── build_exp_dataset.py             experimental processed data -> one row per pair
├── exp_dataset.csv, .xlsx           the pair-level experimental dataset
├── build_exp_dataset_ent.py         exp_dataset -> energy-transfer subset + properties
└── exp_dataset_ent.csv, .xlsx       the input to ../benchmark/
```

## Pipeline

1. Each extraction script reads one raw folder and writes its table into the matching processed folder.
2. `properties_exp.py` and `properties_calc.py` copy the per-compound values into the workbooks.
3. `build_exp_dataset.py` builds `exp_dataset` from the experimental processed folder.
4. `build_exp_dataset_ent.py` builds `exp_dataset_ent` from `exp_dataset` and the property workbooks.

`uv run reproduce.py data` runs all of this in order, plus `quencher_timings.py`,
`quencher_geometry.py`. Run separately:

- `quencher_ml.py` (model environment, workbook in git): `uv run --extra ml python data/computational_data_scripts/quencher_ml.py`.
- `quencher_ml_timing.py` (inference time on the local machine, table in git): `uv run --extra ml python data/computational_data_scripts/quencher_ml_timing.py`.
- `photocatalyst_curate.py` builds the deposited photocatalyst folders from a job tree.

## Photocatalyst ids

Every table and job folder uses the new id (`new_id`, `pc_new_id`). The raw experimental
files keep the old id in their names and pipetting maps (`stern_volmer/PC10_QBatch1.csv`,
`pc_tcspc/PC4.txt`); their scripts map it to the new id on reading.
`compound_identification.xlsx` is the only table with both ids.

## Compound identity

`compound_identification.xlsx` has one sheet per compound class with ids, photocatalyst
old/new ids, names, quencher SMILES and molar masses, read through
`compound_identification.py`. Quencher names are derived from the SMILES.

## Workbooks

One row per compound, sorted by new id or quencher id. Columns are grouped in blocks
(identity first, then one block per property), separated by an empty column. Row 1 names
the blocks; row 2 holds the headers as `quantity / unit`. The `Column key` sheet gives
each column's description and source table.

## exp_dataset

One row per photocatalyst–quencher pair, sorted by `pc_new_id` then `quencher_id`.
Measured columns only: the ids, K_SV, τ0, k_q = K_SV/τ0, `quenches` and `censored`. The
xlsx holds the same table in the workbook layout. PC22 is included; the theory benchmark
excludes it (see SI).

`quenches` is 1 if the Stern–Volmer fit meets any one of these thresholds:
K_SV > 50 at r² > 0.75, > 20 at 0.80, > 10 at 0.90, > 0 at 0.99
(`rung` in `stern_volmer_fits.csv`). Non-quenching pairs have `censored = 1`: the true
K_SV is below the detection limit, k_q is an upper bound.

## exp_dataset_ent

The pairs of `exp_dataset` on which quenching can be attributed to energy transfer.

- **Non-quenching pairs:** all kept.
- **Quenching pairs:** kept only if every single-electron-transfer channel with a measured quencher potential is more than **0.25 V** uphill:
  - ΔE_SET,ox = E_ox(Q) − E(PC\*/PC⁻), oxidation of the quencher
  - ΔE_SET,red = E(PC⁺/PC\*) − E_red(Q), reduction of the quencher
- **Potentials:** quencher potentials measured, photocatalyst excited-state potentials calculated.
- **Limits:** a measured limit (E_ox > x, E_red < x) makes ΔE a lower bound, marked in `dE_SET_ox_bound` / `dE_SET_red_bound`; it rules a channel out only if the limit itself is past 0.25 V.

Columns besides those of `exp_dataset` and the two ΔE_SET:

- **Photocatalyst (DFT):** E_T, both λ and the molecular radius.
- **Quencher (DFT):** E_T, both λ and the molecular radius.
- **Quencher (DvTE):** mean and standard deviation of the vertical gaps, and their p-quantiles for p = 0.005, 0.008, 0.01, 0.015, from a normal distribution and from a KDE.
- **Quencher (ML):** E_T and both λ, each with the model's uncertainty (organic molecules only, no photocatalyst predictions).
- **Driving force:** ΔG_EnT = E_T(Q) − E_T(PC), from both quencher sources.

Values missing from the source table are left empty.

## Methods notes

**Stern–Volmer wavelengths:** the filter pairs in the plate-reader export headers, with
their bandwidths in `plate_reader_filters.csv`. PC1 and PC3: wavelengths from
the screen's photocatalyst sheet (saved protocols name no filters).

**Quencher redox potentials:** read by hand from the voltammograms or taken from the
literature; where a voltammogram is not in `measured data/`, the row's `comment` says so.
Entered in `quencher_redox_measured.csv` (source `CV`) or `quencher_redox_literature.csv`
(a DOI, or `estimated`), joined by `quencher_redox_potentials.py` into
`quencher_redox_potentials.csv`. Bound columns: `>` or `<` where the value is a limit.

**Photocatalyst lifetimes:** one TCSPC trace per photocatalyst; τ0 and its error are the
value and standard error of a single-exponential fit started after the pulse. PC4 is
fitted from 1500 ns (`FIT_START_NS` in `lifetimes_tcspc.py`).

**Reorganization energies** are named `lambda_<surface>_<geometry>`: the energy of one
state at another state's optimized geometry, minus that state at its own geometry.
`lambda_S0_T1` is E_S0 at the T1 geometry minus E_S0. `Dplus` and `Dminus` are the
oxidized and reduced states (`ox` and `red` job folders). Electronic energies only, no
thermal corrections.

**Molecular radii:** `radius_S0` (written by `quencher_dft.py` and `photocatalyst_dft.py`)
is the radius of the sphere with the van der Waals volume of the S0 geometry
(`S0/opt_geometry.xyz`): union of atomic spheres with Bondi radii (Ir and Ru set by hand),
counted on a cubic grid (`molecular_size.py`). In `exp_dataset_ent` as `pc_radius_dft` and
`q_radius_dft`; contact distance r_PC + r_Q.

**Quencher geometry change:** `quencher_geometry_S0_T1.csv` compares the S0 and T1 minima
(`opt_geometry.xyz`): largest relative change of an S0 bond length, largest change of the
deviation from coplanarity about a non-aromatic double bond and about any heavy-atom bond,
and heavy-atom RMSD after Kabsch alignment (columns defined in `quencher_geometry.py`).

**Photocatalyst triplet:** `T1/` holds one triplet per photocatalyst, the lowest minimum
over the ligand-localized triplets; within 1 kcal/mol, the character assigned by
experiment, else the original one. Some T1 jobs and their `S0/sp_vert_T1` start from a
`guess.gbw` that is not deposited.

**Organophotocatalysts:** `organophotocatalysts/` is a separate library with no measured
pairs, not read by `exp_dataset_ent` or the benchmark. wB97X-D4 / def2-TZVP // def2-TZVPP,
CPCM(acetonitrile), no ZORA; no oxidized or reduced states, so no excited-state potentials.

**Dissociative states:** optimisation breaks a bond in the T1 states of Q10 and Q11 (S–S),
Q17 (Se–Se) and Q45 (Se–C), and the reduced states of Q17 (Se–Se) and Q25, Q26 and Q27
(C–Br). Their energies are kept; reorganization energies involving these states are not
those of an intact molecule.

**Predicted quencher properties:** `quencher_properties_ml.xlsx`, E_T, lambda_S0_T1 and
lambda_T1_S0 predicted from the SMILES by the graph mixture density network in
`../ml/data/model_final.pt`, each with its predicted uncertainty.

**Computational cost:** `quencher_timings.csv`, one row per quencher job: DFT (ORCA), the
Gaussian optimisation and frequencies, the Milo trajectories, and the vertical single
points summed per trajectory and multiplicity (per single point in
`quencher_timings_vertical_sp.csv.gz`). Wall times from the program outputs, CPU hours =
wall time × cores. Milo trajectories and failed or time-limited vertical single points
(`dvte sp failed`) are timed by their Slurm reports; a Slurm report next to a program
output is recorded as a cross-check and disagreements are flagged.
`quencher_timings_ml.csv`: import and load times and the median time of one prediction per
quencher and of all quenchers in one call, on the CPU, with `../ml/data/model_final.pt`;
machine, versions and checkpoint in its `#` header lines.

**Calculations:** one calculation per state. `Q<n>/<state>/`: geometry optimisation
(`opt`), single point at that geometry (`sp`) and vertical single points of the other
states (`sp_vert_<kind>`), lowest-energy conformer. `PC<n>/<state>/`: optimisation and
vertical single points (`sp_vert_<state>`); `PC<n>/conf/`: conformer search.
`OPC<n>/{S0,T1}/`: same layout. Q17 S0: `opt_output.out` (thermal correction) and
`sp_output.out` from two runs of the same conformer; Q18 S0: the reduced-state vertical
single point from a second run.

## Raw-data availability

The Zenodo archive includes `experimental_data_raw/` and `computational_data_raw/`
alongside the processed data and code. The GitHub checkout excludes the raw folders.
Without them, `uv run reproduce.py data` skips the extraction scripts.