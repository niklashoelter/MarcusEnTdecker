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
│   ├── organophotocatalysts/        OPC<n>/{S0,T1}/ -> organophotocatalyst_dft.csv
│   ├── quencher_adiabatic_DFT/      Q<n>/{S0,T1,ox,red}/ -> quencher_dft.csv
│   └── quencher_adiabatic_DvTE/     MD trajectories -> quencher_dvte.csv, _samples.csv.gz,
│                                    quencher_timings_vertical_sp.csv.gz
├── computational_data_processed/
│   ├── photocatalyst_properties_calc.xlsx
│   ├── quencher_properties_calc.xlsx
│   ├── quencher_properties_ml.xlsx  E_T, lambda_S0_T1 and lambda_T1_S0 predicted from the SMILES
│   ├── quencher_timings.csv         wall time, cores and CPU hours of every quencher job
│   ├── quencher_timings_ml.csv      the deployed model's inference time per quencher
│   ├── quencher_geometry_S0_T1.csv  change of each quencher geometry from S0 to T1
│   ├── molecular_extent.csv         extent radius of every photocatalyst and quencher
│   └── reorganization_modes.csv     reorganization energies resolved into normal modes
├── experimental_data_scripts/       one script per raw folder (pc_emission/ has none, its
│                                    scans are deposited only), plus properties_exp.py
├── computational_data_scripts/      one script per raw folder, plus properties_calc.py,
│                                    quencher_timings.py, quencher_ml.py,
│                                    quencher_ml_timing.py, photocatalyst_curate.py,
│                                    quencher_geometry.py, molecular_extent.py and
│                                    reorganization_modes.py
├── literature_benchmark/            published tables of the literature benchmark, one CSV
│                                    per study (the PDFs beside them are not in git), and
│                                    photocatalyst_triplet_energies.csv
├── tests/                           pytest tests of molecular_size.py and molecular_extent.py
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

1. Each extraction script reads one raw folder and writes its table into the same folder.
   The two Stern–Volmer scripts write to `experimental_data_processed/`.
2. `properties_exp.py` and `properties_calc.py` copy the per-compound values into the
   workbooks.
3. `build_exp_dataset.py` builds `exp_dataset` from the experimental processed folder.
4. `build_exp_dataset_ent.py` builds `exp_dataset_ent` from `exp_dataset` and the
   property workbooks.

`uv run reproduce.py data` runs all of this in order, including `quencher_timings.py`,
which writes the computational cost of the quencher jobs to
`computational_data_processed/`, and `quencher_geometry.py`, `molecular_extent.py` and
`reorganization_modes.py`, which write the tables of the further analyses there (see
Methods notes). Three scripts are not part of it:

- `quencher_ml.py` needs the model environment and is run by hand:
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

One row per photocatalyst–quencher pair, every combination, sorted by `pc_new_id` then
`quencher_id`. All columns are measured: the ids, K_SV, τ0, k_q = K_SV/τ0, `quenches` and
`censored`. The xlsx holds the same table in the workbook
layout.

PC22 (old PC24) is included although its emission is too weak for Stern–Volmer
analysis; the theory benchmark drops it.

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

Besides the `exp_dataset` columns and the two ΔE_SET columns, each row carries:
- **Photocatalyst (DFT):** E_T, both λ and the molecular radius.
- **Quencher (DFT):** E_T, both λ and the molecular radius.
- **Quencher (DvTE):** mean and standard deviation of the vertical gaps, and their
  p-quantiles for p = 0.005, 0.008, 0.01, 0.015, from a normal distribution and from a KDE.
- **Quencher (ML):** E_T and both λ, each with the model's uncertainty. The model covers
  organic molecules only, so there are no predicted photocatalyst values.
- **Driving force:** ΔG_EnT = E_T(Q) − E_T(PC), from both quencher sources.

A value missing from its source table is left empty here; nothing is filled in.

## Methods notes

**Stern–Volmer wavelengths** are the filter pairs in the plate-reader export headers,
collected with their bandwidths in `stern_volmer/plate_reader_filters.csv`. Old PC1 and
PC3 were read with saved protocols that name no filters; their wavelengths come from the
screen's photocatalyst sheet.

**Quencher redox potentials** are read by hand from the voltammograms, or taken from the
literature; where a voltammogram is not in `measured data/`, the row's `comment` says so. Each quencher is entered in one of two files, `quencher_redox_measured.csv`
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

**Molecular radii.** `quencher_dft.py` and `photocatalyst_dft.py` also write `radius_S0`,
the radius of the sphere with the molecule's van der Waals volume. The volume is that of
the union of atomic spheres with Bondi radii around the optimized S0 geometry
(`S0/opt_geometry.xyz`), counted on a cubic grid; Ir and Ru, which have no Bondi value, get
one by hand. The radii and the grid spacing are in `molecular_size.py`. The radius reaches
the workbooks as `radius_S0` and `exp_dataset_ent` as `pc_radius_dft` and `q_radius_dft`;
the benchmark adds the two to the contact distance of a pair.

**Quencher geometry change.** `quencher_geometry.py` compares the S0 and T1 minima of
each quencher (`opt_geometry.xyz`) and writes `quencher_geometry_S0_T1.csv`: the largest
relative change of an S0 bond length, the largest change of the deviation from coplanarity
about a non-aromatic double bond and about any heavy-atom bond, and the heavy-atom RMSD
after Kabsch alignment. The script's docstring defines each column.

**Molecular extent.** `molecular_extent.py` writes `molecular_extent.csv`, the extent
radius r_extent = max_ij (d_ij + R_i + R_j)/2 of each photocatalyst in the data and each
quencher, from the S0 geometry with the Bondi radii of `molecular_size.py`. It is an
alternative to `radius_S0` for the contact distance.

**Reorganization energies by mode.** `reorganization_modes.py` projects the displacement
between the S0 and T1 minima onto the normal modes of the surface it is evaluated on and
writes `reorganization_modes.csv`: per photocatalyst and quencher, surface (S0 or T1) and
displacement (internal or Cartesian), the sum of the mode contributions λ_j, the fraction of
λ in the modes below 3 k_BT (classical) and the λ-weighted mean frequency of the modes at or
above it (quantum).

**Literature triplet energies.** `literature_benchmark/photocatalyst_triplet_energies.csv`
holds literature E_T of 15 photocatalysts, at 77 K and at room temperature, each with its
measurement conditions and DOI.

**Photocatalyst triplet.** `T1/` holds one triplet per photocatalyst. Triplets with the
excited electron on each kind of ligand were searched. The lowest minimum is kept. Where two
lie within 1 kcal/mol, the character assigned by experiment is kept, or the original one if
none is known. Some of these T1 jobs, and their `S0/sp_vert_T1`, start from a `guess.gbw`
that is not deposited.

**Organophotocatalysts.** `organophotocatalysts/` is a separate library of organic
photocatalysts with no measured pairs. `exp_dataset_ent` and the benchmark do not read it.
Its level of theory is wB97X-D4 / def2-TZVP // def2-TZVPP, CPCM(acetonitrile), without ZORA.
There are no oxidized or reduced states, so there are no excited-state potentials.

**Dissociative states.** In some quencher states the optimisation breaks a bond, so the
optimized structure is not the intact molecule: the T1 states of Q10 and Q11 (S–S), Q17
(Se–Se) and Q45 (Se–C), and the reduced states of Q17 (Se–Se) and Q25, Q26 and Q27
(C–Br). Reruns of Q17 and Q45 from the S0 minimum broke a different bond, so the starting
geometry is not the cause. Their energies are kept, but the
reorganization energies involving these states are not those of an intact molecule, and
the adiabatic triplet energies of Q10, Q11, Q17 and Q45 are closer to fragmentation
energies.

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
one prediction per quencher and of all quenchers in one call, on the CPU, measured with
`../ml/data/model_final.pt`; the machine, versions and checkpoint are in its `#` header
lines.

**Calculations.** The computational folders hold one calculation per state. A quencher
state folder, `Q<n>/<state>/`, holds the geometry optimisation (`opt`), the single point
at that geometry (`sp`) and the vertical single points of the other states
(`sp_vert_<kind>`); the lowest-energy conformer is kept. A photocatalyst state folder,
`PC_<n>/<state>/`, holds the optimisation and the vertical single points
(`sp_vert_<state>`), and `PC_<n>/conf/` the conformer search. An organophotocatalyst
folder, `OPC<n>/{S0,T1}/`, is laid out the same way. In Q17 S0, `opt_output.out`
(the thermal correction) and `sp_output.out` come from two runs of the same conformer; in
Q18 S0, the reduced-state vertical single point comes from a second run.

## Git and Zenodo

From `computational_data_raw/`, only the derived tables are in git. The ORCA/Milo job
folders are on Zenodo.
