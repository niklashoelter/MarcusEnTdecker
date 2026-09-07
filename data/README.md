# data

Instrument output and job files, and the scripts that turn them into tables.

```
experimental_data_raw/         instrument output, read-only
experimental_data_scripts/     raw -> processed
experimental_data_processed/   generated tables

computational_data_raw/        job files, read-only, not in git (~11 GB)
computational_data_scripts/    raw -> processed
computational_data_processed/  generated tables
```

One rule, in both halves: **scripts read from `_raw`, write to `_processed`, and touch
nothing else.** Raw is never modified, processed is never hand-edited, and deleting
`_processed` and re-running `_scripts` gives the same files back.

Splitting experimental from computational means a computed value can never be mistaken
for a measured one, and either side can be replaced without disturbing the other.

## Where it stands

| | status |
|---|---|
| `experimental_data_raw/` | **complete** — 81 SV batches, 36 TCSPC traces, 25 emission, 5 absorption, 40 CV |
| `experimental_data_scripts/` | **7 of 8 written and verified**; only cyclic voltammetry is left |
| `experimental_data_processed/` | **8 tables**, all regenerated from raw by those scripts |
| `computational_data_raw/` | **partly in** — quencher DFT and DvTE complete; no photocatalyst job files |
| `computational_data_scripts/` | **2 of 3 written** — quencher DFT and DvTE; photocatalysts blocked |
| `computational_data_processed/` | quencher DFT and DvTE tables, rebuilt from the job files |

The experimental half round-trips: delete `experimental_data_processed/`, run the
scripts, and the same files come back byte for byte. They are not merely plausible —
each was checked against the output of the code that produced the published numbers,
matching on all 1296 Stern-Volmer point selections, all 35 lifetime fit windows, all
864 extinction coefficients, and 26 of the 27 screened lifetimes. See
`experimental_data_scripts/README.md`.

On the computational side the quencher half is rebuilt from the ORCA outputs: 517 of
the 520 published DFT values reproduce exactly, and the three that do not are errors in
the published spreadsheet rather than here.

## What is missing, and what it blocks

| missing | blocks |
|---|---|
| photocatalyst DFT job files | `photocatalyst_dft.py`; the PC24 and PC19 decisions; the four disputed triplet energies |
| a CV sample-number to quencher mapping | `cyclic_voltammetry_redox.py` — nothing else, the paper's redox values are literature |
| EnTdecker predictions | the model-versus-theory comparison, if the paper makes one |

## The screen

27 photocatalysts x 48 quenchers = **1296 pairs**, complete. Automated Stern-Volmer
quenching in acetonitrile on an Opentrons Flex and a BMG VantaStar plate reader;
96-well glass-coated plates, six-point dilution series.

At a k_q cutoff of 1e7 M-1 s-1, **695 pairs quench** and **601 do not**.
(694/602 before tau_0 was re-derived from the TCSPC traces; the single pair that
moved is PC4/Q45, which sits on the threshold. See `../dataset/README.md`.)

## Three things to know before using this

1. **Half the dataset is left-censored.** 662 pairs (51.1 %) fall below the detection
   floor `K_SV >= 25 M-1 and r2 >= 0.90`, and 379 of those have a negative fitted
   slope. They are one-sided observations, not missing values, and they belong in any
   analysis as non-quenchers — no redox value and no DFT energy can turn a non-quencher
   into missing data. In rate terms the floor is photocatalyst-specific, 3.9e6 to
   3.4e8 M-1 s-1, because it scales as `25 / tau_0`.

2. **Label on k_q, not K_SV.** `K_SV = k_q * tau_0` and `tau_0` spans 86x across the
   screen, so a fixed K_SV cutoff would demand an 86x higher rate constant from the
   short-lived photocatalysts. Marcus and Rehm-Weller predict k_q.

3. **The largest rate constants rest on the fewest points.** Range scanning drops
   high-concentration points where I0/I goes non-linear, which is exactly where
   quenching is strongest. 810 of 7757 measured points (10.4 %) are excluded from
   their fit.

## Column naming

| suffix | meaning |
|---|---|
| `_exp` | measured here, or from the literature with a source recorded |
| `_dft` | computed by DFT (four-point scheme) |
| `_md` | from Milo molecular-dynamics sampling (DvTE only) |
| `_ml` | predicted by a model |

Columns with no suffix are identifiers, instrument settings, or arithmetic
derivations. Units follow the value: `_ns`, `_V`, `_kcal`, `_M-1s-1`.

## Size

`computational_data_raw/` is about 11 GB and is not in git — see its README.
Everything else in `data/` together is under 300 MB.
