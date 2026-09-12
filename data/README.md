# data

Measurements and calculations, from instrument files and ORCA jobs to one property
workbook per compound class, and from the experimental side of those to the pair-level
dataset `exp_dataset.csv`, which is where `../benchmark/` starts.

```
data/
├── experimental_data_raw/           instrument files, plus the tables derived from them
│   ├── pc_emission/                 emission scans -> emission_spectra.csv
│   ├── pc_tcspc/                    PC<n>.txt/.FL, one trace each -> lifetimes_tcspc.csv
│   ├── quencher_absorption/         absorption scans -> absorption_extinction.csv
│   ├── quencher_cyclic_voltammetry/ CV traces -> cyclic_voltammetry_redox.csv, _traces.csv
│   └── stern_volmer/                plate reads, pipetting maps -> plate_reader_filters.csv
├── experimental_data_processed/
│   ├── photocatalyst_properties_exp.xlsx
│   ├── quencher_properties_exp.xlsx
│   ├── stern_volmer_intensities.csv
│   └── stern_volmer_fits.csv
├── computational_data_raw/          ORCA/Milo jobs (Zenodo, not in git), plus derived tables
│   ├── photocatalysts/              PC_<n>/{S0,T1,ox,red,conf}/ -> photocatalyst_dft.csv
│   ├── quencher_adiabatic_DFT/      Q<n>/{S0,T1,ox,red}/ -> quencher_dft.csv
│   └── quencher_adiabatic_DvTE/     MD trajectories -> quencher_dvte.csv, _samples.csv.gz
├── computational_data_processed/
│   ├── photocatalyst_properties_calc.xlsx
│   ├── quencher_properties_calc.xlsx
│   └── quencher_properties_ml.xlsx  the same three quantities, predicted from the SMILES
├── experimental_data_scripts/       one script per raw folder, plus properties_exp.py
├── computational_data_scripts/      one script per raw folder, plus properties_calc.py
│                                    and quencher_ml.py, which runs the model
├── compound_identification.py/.xlsx which compounds are in the data, what they are,
│                                    and the old -> new photocatalyst id mapping
├── properties_workbook.py           layout shared by the workbooks
├── build_exp_dataset.py             the experimental processed folder -> one row per pair
└── exp_dataset.csv, .xlsx           the pair-level experimental dataset
```

**Order.** Each extraction script reads one raw folder and writes its table into that same
folder; only the two Stern–Volmer scripts write to `experimental_data_processed/`.
`properties_exp.py` and `properties_calc.py` then copy the per-compound values into
the workbooks; nothing is computed at that step. `build_exp_dataset.py` then turns the
experimental processed folder into `exp_dataset`, one row per pair. The calculated
workbooks are not joined here — that is the consumer's step, and `../benchmark/pairs.py`
does it. `uv run reproduce.py data` runs it all in order. `quencher_ml.py` is the one
script here it does not run — see below.

**Photocatalyst numbers.** The raw files and the extraction scripts use the old id, the
number from the screen (`PC10_QBatch1.csv`, `pc_tcspc/PC4.txt`). The new id, PC1–PC27 without
gaps, enters only at the end: the processed photocatalyst workbooks carry `old_id` and
`new_id`, `exp_dataset` carries `pc_old_id` and `pc_new_id`, and everything downstream uses
the new id. Old PC7, PC26, PC30 and PC31 are crossed out in the screen's photocatalyst
sheet and are not in the data. `compound_identification.xlsx` holds the mapping, and
`compound_identification.py` reads it with the standard library alone, so importing it
adds no dependency to any script.

**Workbooks.** One row per compound, sorted by new id (PC1–27) or quencher id (Q1–48). Columns in blocks —
identity first, then one block per property — separated by an empty column and named in
row 1; headers in row 2 carry their unit as `quantity / unit`. The `Column key` sheet
gives each column's description and the raw table it came from.

**exp_dataset.** One row per photocatalyst–quencher pair, 27 × 48 = 1296, sorted by
`pc_new_id` then `quencher_id`. It includes PC22 (old PC24), whose emission was too weak;
the benchmark drops it, the dataset does not. It holds twelve columns and they are all
experimental: the ids, K_SV, τ0, k_q = K_SV/τ0 and its floor 10/τ0, the two labels —
`quenches` and `quenches_strict` (K_SV ≥ 25 and r² ≥ 0.90, the published criterion) — and
`censored`. The xlsx is the same table in the workbook layout.

`quenches` is a ladder, in which the weaker the quenching the straighter the line has to
be: K_SV > 50 at r² > 0.75, > 20 at 0.80, > 10 at 0.90, > 0 at 0.99. Any one rung is
enough, and `stern_volmer_fits.csv` records which one carried each pair. So no fit worse
than r² 0.75 is ever called quenching however large its K_SV, and a K_SV of a few M⁻¹
needs an almost perfect line. 735 of 1296 pairs quench, 637 of them strictly.

The fit behind those numbers departs from the acquisition-time code in exactly one place:
that code skipped its linear-range scan whenever the all-points r² fell below 0.5, so a
pair that is linear and then saturates took its K_SV from the whole series. The scan now
runs for every pair. 58 pairs choose different points because of it and say so in their
`note`; the rest are unchanged. `stern_volmer_fits.py` gives the case that shows why.

Nothing calculated is in it, and neither driving force is. A DFT triplet energy is a
property of a compound and lives one row per compound in `computational_data_processed/`;
ΔG_EnT and ΔG_SET are differences across a pair and are built by whoever compares them,
which is `../benchmark/pairs.py`. ΔG_SET is the clearest case: half of it is a measured
quencher potential and half a calculated excited-state potential of the photocatalyst, so
a column of it would sit among the measurements without being one. The measured potentials
stay in `quencher_properties_exp.xlsx` with the bound flags that say which are limits.

**Stern–Volmer wavelengths** are the filter pairs in the plate-reader export headers,
collected with their bandwidths in `stern_volmer/plate_reader_filters.csv`. Old PC1 and PC3
were read with saved protocols that name no filters; their wavelengths come from the
screen's photocatalyst sheet. The pipetting maps' wavelength fields are placeholders and
are not read.

**Compound identity** has one source, `compound_identification.xlsx`: a sheet per compound
class with the ids, the photocatalyst old→new mapping, the names, the quencher SMILES and
the molar masses. Every script asks `compound_identification.py` for them, so there is one
answer to what a compound is. The quencher names are derived from the SMILES, each checked
by converting it back to a structure; the pipetting maps keep their own shorthand labels
(they call seven of the thiophenes and furans thiazoles and oxazoles — the recorded molar
masses confirm the SMILES), and nothing reads a name from them.

**Quencher redox potentials** carry a bound column (`>` or `<` where the value is a limit),
a `source` — `CV` (measured here), the DOI of the paper, or `estimated` — and a free-text
`comment`. Only Q32 and Q33 are estimated, and from one reading of the thioamide class for
both; every other value is a voltammogram or a paper.

**Photocatalyst lifetimes** come from one TCSPC trace per photocatalyst; τ0 and its error are
the value and standard error of a single-exponential fit. The fit starts automatically after
the pulse, except for old PC4, fitted from 1500 ns to skip a fast early component
(`FIT_START_NS` in `lifetimes_tcspc.py`).

**Photocatalyst DFT** uses Leon's arithmetic and reproduces his delivered file
except E_T for old PC10, PC13, PC25 and PC28 (new 8, 11, 23, 26), which that file carries
stale.

**Predicted quencher properties.** `quencher_properties_ml.xlsx` holds the same three
quantities as the DFT quencher workbook — E_T, λ_S_T, λ_T_S — predicted from the SMILES
alone by the graph mixture density network in `../ml/data/model.pt`, each
with the model's own uncertainty, which a DFT number does not have. It is written by
`computational_data_scripts/quencher_ml.py`, the only script in this folder that needs the
model environment (`uv sync --extra ml`, then
`uv run --extra ml python data/computational_data_scripts/quencher_ml.py`) and therefore
the only one `reproduce.py` does not run: the rest of the folder rebuilds without
installing torch, and the workbook is in git so a clone has the predictions anyway. The two
workbooks are meant to be read side by side, not merged — whoever compares DFT, DvTE and
the model does that where the comparison is made.

**One calculation per state, and no other.** Both computational folders hold only the job
each published number came from: `Q<n>/<state>/` and `PC_<n>/<state>/` carry one geometry
optimisation and its single points, named by what they are -- `opt`, `sp`, `sp_vert_<kind>`
-- with no conformer number and no pipeline number. The delivery held several conformers
per quencher state, and a few states were computed twice in parallel pipelines; the
conformer that won on energy is the one kept and the rest are not deposited, so nothing
downstream has to choose. Two exceptions are worth knowing when reading a folder: Q17's S0
takes its thermal correction, and Q18's S0 its reduced-state vertical single point, from a
second run of the same conformer, so in those two folders `opt_output.out` is not the run
that `sp_output.out` came from.

**In git from** `computational_data_raw/`**:** only the derived tables and the photocatalyst
`manifest.csv`, which records which delivered job was kept and which was dropped and why.
