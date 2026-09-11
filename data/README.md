# data

Measurements and calculations, from instrument files and ORCA jobs to one property
workbook per compound class. Everything downstream (`../dataset/`) reads only the
`*_processed/` folders.

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
└── properties_workbook.py           layout shared by the four workbooks
```

**Order.** Each extraction script reads one raw folder and writes its table into that same
folder; only the two Stern–Volmer scripts write to `experimental_data_processed/`.
`properties_exp.py` and `properties_calc.py` then copy the per-compound values into
the workbooks; nothing is computed at that step. `uv run reproduce.py data` runs it all in
order.

**Workbooks.** One row per compound, sorted by id (PC1–29, Q1–48). Columns in blocks —
identity first, then one block per property — separated by an empty column and named in
row 1; headers in row 2 carry their unit as `quantity / unit`. The `Column key` sheet
gives each column's description and the raw table it came from.

**Quencher redox potentials** carry a bound column (`>` or `<` where the value is a limit),
a `source` — `CV` (measured here), the DOI of the paper, or `estimated` (Q19, Q32, Q33) —
and a free-text `comment`.

**Photocatalyst lifetimes** come from one TCSPC trace per photocatalyst; τ0 and its error are
the value and standard error of a single-exponential fit. The fit starts automatically after
the pulse, except for PC4, fitted from 1500 ns to skip a fast early component
(`FIT_START_NS` in `lifetimes_tcspc.py`).

**Photocatalyst DFT** uses Leon's arithmetic and reproduces his delivered file
(`spectroscopy_data_processed/dataset/properties_photocatalysts_calc.xlsx`) except E_T for
PC10, PC13, PC25 and PC28, which that file carries stale.

**In git from `computational_data_raw/`:** only the derived tables, the photocatalyst
`manifest.csv` (which delivered job was kept) and the quencher `provenance.csv` (which
pipeline and conformer each state came from).
