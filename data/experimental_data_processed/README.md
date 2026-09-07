# Experimental data — processed

Output of `../experimental_data_scripts/`. Everything here is generated; nothing is edited
by hand. Deleting this folder and re-running the scripts gives the same files back.

| file | rows | produced by |
|---|---|---|
| `stern_volmer_intensities.csv` | 7776 | `stern_volmer_intensities.py` |
| `stern_volmer_fits.csv` | 1296 | `stern_volmer_fits.py` |
| `lifetimes_tcspc.csv` | 36 | `lifetimes_tcspc.py` |
| `lifetimes_tau0.csv` | 28 | `lifetimes_tau0.py` |
| `absorption_extinction.csv` | 48 | `absorption_extinction.py` |
| `emission_spectra.csv` | 14067 | `emission_spectra.py` |
| `compound_records_quenchers.csv` | 48 | `compound_records.py` |
| `compound_records_photocatalysts.csv` | 29 | `compound_records.py` |

All of them reproduce the published analysis. The Stern-Volmer fits match the 1296 per-pair
workbooks written at acquisition time on every point-selection mask, and K_SV to 5.7e-14
relative; the lifetimes match on every fit window; the extinction coefficients match the
published epsilon values to the six-significant-figure rounding. See the scripts README
for the verification detail and for the two acquisition-code quirks reproduced on purpose.

## Still to come

| file | source |
|---|---|
| `cyclic_voltammetry_redox.csv` | the 40 CV traces, blocked on a sample-number to quencher mapping |

## Reading these tables

**Non-quenchers are data.** 662 of the 1296 pairs (51.1 %) fall below the detection floor
and 379 of those have a negative fitted slope. They stay in as one-sided observations with
their negative K_SV intact. No missing redox value and no missing DFT energy converts a
measured non-quencher into a missing row.

**The floor is photocatalyst-specific.** `K_SV >= 25 M-1` is a fixed cutoff, but tau_0
spans 86x across the screen, so in rate terms the floor runs from 3.9e6 to 3.4e8 M-1 s-1.
Any labelling or thresholding downstream happens on k_q, not on K_SV.

**19 wells have no usable reading.** Every repeat read hit the detector rail at 260000, so
`intensity_au` is empty and the well is excluded from its pair's fit. `n_measured` in
`stern_volmer_fits.csv` records how many of the six points survived.

**`lifetimes_tcspc.csv` has one row per trace file; `lifetimes_tau0.csv` has one row per
photocatalyst.** The first is the fit of each measurement, the second is the decision
about which measurement to believe. Keeping them apart means the selection rule is
visible and arguable rather than baked into the fit.

* `duplicate_of` flags two files that are the same measurement stored twice. 36 files
  hold 34 distinct measurements. Use `n_distinct_traces`, not `n_traces`, if you want to
  say how often a photocatalyst was actually measured.
* One trace, `24_ACN_Ar_10ns_-5M_3mL`, has no usable channels and is not fitted. Its row
  is kept with an empty `tau_ns` and the reason in `note`. PC24 has a second trace.

**Every screened photocatalyst has a lifetime.** The screen covers PC 1-6, 8-25 and
27-29; all 27 have a usable trace. PC7 is measured but not screened, and PC26 appears in
neither — so no photocatalyst in the dataset depends on an externally curated tau_0.

`lifetimes_tau0.csv` reproduces **26 of those 27** curated `t0 / ns` values exactly.
**PC4 is the one that does not**: its curated 1494.54 ns matches none of its measurements
(1228.49 and 1039.14 ns usable, both at unusually low amplitude), and appears to come from
a re-fit that is not in the data. Taking the longest gives 1228.49 ns, which scales k_q
for all 48 of PC4's pairs by 1.22. `consistent = 0` flags it; it needs a decision.

Read `tau0_std_ns` as one-sided. Where it comes from repeats it is the scatter between
measurements whose dominant error — incomplete degassing — can only shorten the result, so
the true value sits at or above `tau0_ns` rather than centred on it. Where
`n_measurements = 1` it is that fit's own standard error, which is a lower bound on the
real uncertainty.
