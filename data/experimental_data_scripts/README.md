# Experimental data — scripts

Each script reads `../experimental_data_raw/`, writes `../experimental_data_processed/`,
and touches nothing else. Run any of them with `uv run <script>.py` — no arguments, no
install step, no configuration. Dependencies are declared in a PEP 723 header at the top
of each file.

Scripts and their outputs share a measurement prefix.

| script | reads | writes | status |
|---|---|---|---|
| `stern_volmer_intensities.py` | `stern_volmer/` (81 plate exports + 81 pipetting maps) | `stern_volmer_intensities.csv` (7776 wells) | **verified** |
| `stern_volmer_fits.py` | `stern_volmer_intensities.csv` | `stern_volmer_fits.csv` (1296 pairs) | **verified** |
| `lifetimes_tcspc.py` | `pc_tcspc/` (36 traces) | `lifetimes_tcspc.csv` (36 rows) | **verified** |
| `lifetimes_tau0.py` | `lifetimes_tcspc.csv` | `lifetimes_tau0.csv` (28 photocatalysts) | **verified** — 26/27 screened match the curated table exactly |
| `absorption_extinction.py` | `quencher_absorption/` (1 scan + plate map) | `absorption_extinction.csv` (48 quenchers) | **verified** |
| `emission_spectra.py` | `pc_emission/` (1 of 25 scans) | `emission_spectra.csv` (14067 points) | **written** |
| `compound_records.py` | `compound_records/` (3 spreadsheets) | `compound_records_quenchers.csv` (48), `compound_records_photocatalysts.csv` (29) | **written** |
| `cyclic_voltammetry_redox.py` | `quencher_cyclic_voltammetry/` (40 traces) | — | blocked, no sample map |

### Which absorption scan is the real one

`quencher_absorption/` holds five scans, and four of them are the wrong ones. The four
originally copied in are 96-well, semicolon-delimited exports with German decimal commas.
Recomputing epsilon from each and comparing against the published values shows none of
them is the source:

| scan | wells | median deviation from published epsilon |
|---|---:|---:|
| `10-29-03 Absorption_Scan_300_1000` | 96 | 104 % |
| `10-38-19 Absorption_Scan_220_1000` | 96 | 58 % |
| `10-44-59 Absorption_Scan_360_650` | 96 | 42 % |
| `10-53-28 Absorption_Scan_360_650_no_path_correction` | 96 | 25 % |
| **`10-53-28 Absorption_Scan_360_650_56well_published`** | **56** | **0.000 %** |

The last one was copied in from
`Quencher/Absorption/Absorption_Scan_360_650/` and is the file the published pipeline was
actually run on: a comma-delimited export carrying a `Content` column, holding exactly the
56 wells in use (48 quenchers in A1-H6 plus 8 solvent blanks in column 7) rather than the
full plate. It reproduces `combined_quencher_absorption.csv` on 864 of 864 values to
better than 1e-9, and the current table matches it to 7.8e-7 median, which is the
six-significant-figure rounding and nothing else.

The trap is that the discarded 96-well file carries the **same name and the same
timestamp** as the correct one but different numbers, so it cannot be told apart by name.
The four unused scans are left in place as raw records; `SCAN` at the top of the script
names the one that is read.

### Why emission uses one file out of 25

`pc_emission/` contains no plate map, and for most of its files that is fatal — they are
labelled by well with no record of which well held which photocatalyst. One file is
different: `Emission_Scan_Exc_380_405_425_450_Gain_1500.csv` labels its rows with the
photocatalyst number directly, so it needs no map. That is the file the script reads, and
it is the source of the `Emission_Spectra_Gain_1500` sheet in
`Photocatalysts/photocatalysts.xlsx`.

Everything else in the folder is ignored on purpose: six autofluorescence tests of plate
materials, files marked `quick`, `DRY` and `old_`, and single-excitation
method-development scans that are well-labelled and therefore unresolvable. One row in the
380 nm block is labelled `96`, outside the photocatalyst range, and is skipped and
reported rather than silently dropped.

### Where tau_0 comes from

`tau_0` turns a Stern-Volmer constant into a rate constant, `k_q = K_SV / tau_0`, so it
sets the scale of every rate in the paper. It is **measured**, not inherited: all 27
screened photocatalysts have a usable TCSPC trace, so nothing has to be filled in from a
curated table.

`lifetimes_tau0.py` collapses the per-trace table to one lifetime per photocatalyst.
Duplicates and failed fits are dropped, then a trace is dropped if its acquisition window
spans fewer than five lifetimes — a window that short cannot separate the decay from the
baseline. That removes exactly one trace across the whole set, PC4's 200 ns range, which
returns 51 ns against roughly 1100 ns from the same compound on microsecond ranges.

Where a photocatalyst was measured more than once, the **longest** surviving measurement
is taken, not the mean. These are triplet lifetimes under argon and the dominant error is
incomplete degassing, which can only ever shorten the result — the error is one-sided, so
averaging a good measurement with a poor one does not cancel it but biases tau_0 down.
PC22 shows it directly: three measurements at 3095, 3135 and 3305 ns, the longest being
the one the experimenter labelled `betterDegas`.

This reproduces **26 of the 27** curated `t0 / ns` values in
`spectroscopy_data_processed/dataset/properties_photocatalysts.xlsx` exactly, which
establishes that the curated table was itself derived from these traces.

**PC4 is the exception and is not reconciled.** Its curated 1494.54 ns matches none of its
measurements — the two usable ones give 1228.49 and 1039.14 ns, both at unusually low
amplitude (133 and 123 counts against 900+ for a typical trace here). The rule gives
1228.49 ns, 18 % below the curated figure, which scales k_q for all 48 of PC4's pairs by
1.22. It needs a decision, not a rule, and `consistent = 0` marks it.

### Why CV is still blocked

`quencher_cyclic_voltammetry/` has 19 distinct sample numbers (009-048) plus blanks and no
map from those numbers to compounds. The numbers are plausibly quencher IDs, but nothing
in the data says so, and the manuscript's redox values are literature rather than
measured, so there is no published output against which a guess could be checked.
The mapping has to come from the lab record.

## The one exception to the raw-only rule

`stern_volmer_fits.py` reads a generated file rather than raw instrument output. The
replicate QC that turns repeat reads into one intensity per well is deliberately
implemented once, in `stern_volmer_intensities.py`, rather than duplicated in the fitting
script where the two copies could drift apart. A simpler QC rule moved four K_SV values
by up to 13 %, so having one definition of it matters more than the purity of the rule.

## Where the algorithms come from

Neither fitting procedure was invented here. Both are ports of the code that produced the
published numbers, and both were checked against that code's own output.

**Stern-Volmer** — `Auto_Quenching_Backup/auto_quenching/analysis/stern_volmer.py` and
`regression_utils.py` (read-only, outside this tree). Verified against the 1296 per-pair
workbooks written at acquisition time, in
`spectroscopy_data_processed/raw/<batch>/stern_volmer/SV_*.xlsx`, which record for every
point whether it entered the fit:

```
point-selection masks   1296 / 1296 identical
K_SV                    agrees to 5.7e-14 relative
fit method              1296 / 1296 identical
```

**TCSPC** — `Photocatalysts/plot_lifetimes.py`. All 35 fittable traces reproduce with
identical fit windows (channel start, channel end, point count) and tau agreeing to
rounding.

## What the Stern-Volmer procedure actually does

The all-points fit of `I0/I` against `[Q]` decides the branch:

```
slope < 0        -> report it, not quenching   "negative initial slope"        374 pairs
r2 < 0.50        -> report it, not quenching   "initial r2 below ..."           88 pairs
r2 > 0.995       -> report it                  "perfect measurement"           256 pairs
otherwise        -> range scan                 "range scanned"                 578 pairs
```

The scan walks contiguous intervals anchored at `[Q] = 0`, minimum 3 points. Within each
interval at most one internal point is excluded on a Theil-Sen residual criterion, and
intervals are ranked by `score_fn(residuals, n) = n**0.5 / (1 + MAE)**4`. The winner is
refitted by ordinary least squares with a free intercept — that refit is what is reported.

A pair counts as quenching when `K_SV >= 25 M-1` **and** `r2 >= 0.90`. 634 pairs quench,
662 do not.

### Two quirks kept on purpose

Both are marked `QUIRK` in `stern_volmer_fits.py`. They look like defects, they are
reproduced deliberately, and changing either would silently move published numbers.

1. `exclude_outlier_theil_sen` returns the trimmed arrays whether or not the point
   qualified as an outlier. Every interval is therefore *scored* on a fit that has its
   worst point removed, while the exclusion is only *recorded* when it qualified.
2. The exclusion is applied under `if outlier_index:`, a plain truthiness test, so an
   outlier at position 0 is never removed. That is why every fit keeps the `[Q] = 0`
   anchor point.

### One dependency the original did not declare

The original imported scikit-learn for `LinearRegression` and `TheilSenRegressor` without
listing it in its dependencies, so the version behind the published values is not
recorded. Both estimators are reimplemented here in numpy — ordinary least squares, and
Theil-Sen as the spatial median of the exact lines through every pair of points, which is
what `TheilSenRegressor` computes at `n_subsamples = n_features + 1 = 2`. That removes
scikit-learn and scipy from the pipeline entirely and makes the result independent of
library versions. It reproduces the original on all 1296 pairs.

## Precision

Computed floats are rounded to **six significant figures**, well beyond the experimental
precision. Two exceptions, both deliberate:

* `intensity_au` is a mean of integer counts and is rounded to **one decimal place**.
  Six significant figures sits exactly at the boundary of a six-digit count and was
  silently discarding the half-count in means such as `229026.5`.
* Identifiers, channel numbers and point counts stay integers.

Rounding at the `stern_volmer_intensities.csv` handover changes no decision anywhere
downstream — identical point selections, identical notes, identical quenching flags on all
1296 pairs. It moves K_SV by at most `1.3e-5` relative across the 634 quenching pairs
(median `1.3e-6`). The larger figures seen on non-quenchers are relative deviations on
values near zero, e.g. `0.0140186 -> 0.0140661`, and carry no meaning.

## Two things the raw data gets wrong

Found by the scripts, not assumed:

* **Two TCSPC files are duplicate measurements.**
  `4_ACN_Ar_20us_-5M_3mL_IrdFppy.txt` is byte-identical to
  `4_ACN_Ar_10us_-5M_3mL_IrdFppy.txt`, and `3_ACN_Ar_20us_-5M_3mL_Irppy.txt` differs from
  `3_ACN_Ar_10us_-5M_3mL_Irppy.txt` only in its `Labels` line — all 2048 data rows are
  identical. So 36 files hold 34 distinct measurements; PC3 has one real trace, not two,
  and PC4 has three, not four. `lifetimes_tcspc.py` hashes the numeric rows and reports
  this in `duplicate_of` and `n_distinct_traces` rather than double-counting.
* **Quencher identity comes from the well position, not the plate label.** Ten plates
  label their quenchers with bare names and two of those names are misleading
  (`Bpin-Benzoxazol` is Bpin-benzofuran, `Br-Thiazol` is Br-benzothiophene). All 71
  labelled plates agree on the layout of each QBatch, and every bare-name plate matches
  its batch layout 96 wells out of 96, so position resolves identity with no alias table
  and no string matching.

## Style

Minimal, not general. Only the options actually used are implemented — no alternative
fitting modes, no configuration files, no command-line switches. Paths are resolved
relative to the script's own location, so a script can only ever write where it should.
