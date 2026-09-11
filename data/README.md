# data

Measurements and calculations, from instrument files and ORCA jobs to one property
workbook per compound class, and from those to the pair-level dataset `exp_dataset.csv`,
which is all that `../benchmark/` reads.

```
data/
├── experimental_data_raw/           instrument files, plus the tables derived from them
│   ├── compound_records/            lab records (xlsx) -> compound_records_*.csv
│   ├── pc_emission/                 emission scans -> emission_spectra.csv
│   ├── pc_tcspc/                    PC<n>.txt/.FL, one trace each -> lifetimes_tcspc.csv
│   ├── quencher_absorption/         absorption scans -> absorption_extinction.csv
│   ├── quencher_cyclic_voltammetry/ CV traces -> cyclic_voltammetry_redox.csv, _traces.csv
│   └── stern_volmer/                plate reads and pipetting maps
├── experimental_data_processed/
│   ├── photocatalyst_properties_exp.xlsx
│   ├── quencher_properties_exp.xlsx
│   ├── stern_volmer_intensities.csv
│   └── stern_volmer_fits.csv
├── computational_data_raw/          ORCA/Milo jobs (Zenodo, not in git), plus derived tables
│   ├── photocatalysts/              PC_<n>/{S0,T1,ox,red,conf}/ -> photocatalyst_dft.csv
│   ├── quencher_adiabatic_DFT/      Q<n>/{S0,T1,ox,red}/ -> quencher_dft.csv, _conformers.csv
│   └── quencher_adiabatic_DvTE/     MD trajectories -> quencher_dvte.csv, _samples.csv.gz
├── computational_data_processed/
│   ├── photocatalyst_properties_calc.xlsx
│   └── quencher_properties_calc.xlsx
├── experimental_data_scripts/       one script per raw folder, plus properties_exp.py
├── computational_data_scripts/      one script per raw folder, plus properties_calc.py
├── photocatalyst_ids.py             which photocatalysts are in the data; old -> new id
├── properties_workbook.py           layout shared by the workbooks
├── build_exp_dataset.py             processed folders -> one row per pair
└── exp_dataset.csv, .xlsx           the pair-level dataset
```

**Order.** Each extraction script reads one raw folder and writes its table into that same
folder; only the two Stern–Volmer scripts write to `experimental_data_processed/`.
`properties_exp.py` and `properties_calc.py` then copy the per-compound values into
the workbooks; nothing is computed at that step. `build_exp_dataset.py` then joins the
processed folders into `exp_dataset`. `uv run reproduce.py data` runs it all in order.

**Photocatalyst numbers.** The raw files and the extraction scripts use the old id, the
number from the screen (`PC10_QBatch1.csv`, `pc_tcspc/PC4.txt`). The new id, PC1–PC27 without
gaps, enters only at the end: the processed photocatalyst workbooks carry `old_id` and
`new_id`, `exp_dataset` carries `pc_old_id` and `pc_new_id`, and everything downstream uses
the new id. Old PC7, PC26, PC30 and PC31 are crossed out in the screen's photocatalyst
sheet and are not in the data. `photocatalyst_ids.py` holds the mapping.

**Workbooks.** One row per compound, sorted by new id (PC1–27) or quencher id (Q1–48). Columns in blocks —
identity first, then one block per property — separated by an empty column and named in
row 1; headers in row 2 carry their unit as `quantity / unit`. The `Column key` sheet
gives each column's description and the raw table it came from.

**exp_dataset.** One row per photocatalyst–quencher pair, 27 × 48 = 1296, sorted by
`pc_new_id` then `quencher_id`. It includes PC22 (old PC24), whose emission was too weak;
the benchmark drops it, the dataset does not. It holds only what the
benchmark reads: ids, K_SV, τ0, k_q = K_SV/τ0 and its floor 10/τ0, the two labels —
`quenches` (K_SV > 10, or a significant slope below it) and `quenches_strict` (K_SV ≥ 25
and r² ≥ 0.90) — and `censored`, the DFT/MD inputs of the theories, and the EnT and SET
driving forces. The xlsx is the same table in the workbook layout.

**Quencher redox potentials** carry a bound column (`>` or `<` where the value is a limit),
a `source` — `CV` (measured here), the DOI of the paper, or `estimated` (Q19, Q32, Q33) —
and a free-text `comment`.

**Photocatalyst lifetimes** come from one TCSPC trace per photocatalyst; τ0 and its error are
the value and standard error of a single-exponential fit. The fit starts automatically after
the pulse, except for old PC4, fitted from 1500 ns to skip a fast early component
(`FIT_START_NS` in `lifetimes_tcspc.py`).

**Photocatalyst DFT** uses Leon's arithmetic and reproduces his delivered file
(`spectroscopy_data_processed/dataset/properties_photocatalysts_calc.xlsx`) except E_T for
old PC10, PC13, PC25 and PC28 (new 8, 11, 23, 26), which that file carries stale.

**In git from** `computational_data_raw/`**:** only the derived tables, the photocatalyst
`manifest.csv` (which delivered job was kept) and the quencher `provenance.csv` (which
pipeline and conformer each state came from).
