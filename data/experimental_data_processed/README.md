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
| `compound_records_photocatalysts.csv` | 29 | `compound_records.py`, `photocatalyst_smiles.py` |
| `cyclic_voltammetry_redox.csv` | 19 | `cyclic_voltammetry_redox.py` |
| `cyclic_voltammetry_traces.csv` | 25821 | `cyclic_voltammetry_redox.py` |

All of them reproduce the published analysis. The Stern-Volmer fits match the 1296 per-pair
workbooks written at acquisition time on every point-selection mask, and K_SV to 5.7e-14
relative; the lifetimes match on every fit window; the extinction coefficients match the
published epsilon values to the six-significant-figure rounding. See the scripts README
for the verification detail and for the two acquisition-code quirks reproduced on purpose.

## Cyclic voltammetry

`cyclic_voltammetry_redox.csv` is one row per CV sample; `cyclic_voltammetry_traces.csv`
is the tidy long form of the traces themselves — 25821 points, one row per exported
point, carrying the raw current, the blank and the blank-corrected current so that any
plot or re-extraction can start from the same numbers the summary did.

**The sample number is the quencher id, and this table is where that was checked.** The
file names carry a three-digit sample number and nothing else; the mapping came from the
lab record and is verified here against the potentials recorded per quencher in
`compound_records_quenchers.csv`. Of the 38 branches — 19 samples, oxidation and
reduction — 32 carry a plain recorded measurement and 5 carry a bound:

| | |
|---|---|
| plain measurements compared | 32 |
| reproduced within 0.15 V | **29** |
| median absolute deviation | **0.050 V** |
| mean signed deviation over those 29 | +0.046 V |
| recorded bounds, checked as bounds | 5 of 5 consistent |

0.050 V is about one exported point on the coarse traces. A wrong mapping would have
scattered by a volt or more — the recorded oxidations alone span 0.86 to 2.5 V — and a
mis-referenced potential axis would have shifted every value by the ~0.4 V of a missed
ferrocene correction. Neither is present. Read `delta_E_ox_V` / `delta_E_red_V` and
`ox_agrees` / `red_agrees` per row for the per-sample version of this.

**Bounds are checked as bounds, never as numbers.** Five recorded values are limits
(`> 2.4`, `< -2.5`), meaning the compound was not oxidized or reduced anywhere in the
window. `delta_*_V` is deliberately empty for those; what is checked instead is that the
trace shows no wave inside the window, or one only beyond the limit. All five pass.

**`E_ox_V` and `E_red_V` are the reported potentials; `E_ox_basis` / `E_red_basis` says
what they are.** 35 of the 38 branches are irreversible — the wave runs into the solvent
limit with no return peak — and for those the reported number is the tangent onset,
which is what the experimenters read off the plots. Three branches, all reductions
(Q28, Q39, Q42), do show a return peak, and there the reported number is the half-wave
potential, which is the thermodynamic quantity. On all three the half-wave lands closer
to the recorded value than the onset does.

Treat the other 35 as kinetic. An irreversible onset is not a redox equilibrium and
should not enter a driving-force calculation as though it were one. The peak and
half-peak potentials are carried alongside for anyone who prefers a different
convention.

**Four branches are not reproduced, and each row says why in `note`.** 017's reduction
(recorded -0.2 V, extracted -1.57 V; its lab comment reads "ugly CV measured"), 046's
reduction (-2.0 V against -2.29 V, on a trace whose oxidation is a shapeless few-uA
hump), 023's oxidation (-0.153 V, three thousandths outside the tolerance) and 022's
oxidation, which is **empty**: its two repeats disagree by more than a volt, so no value
is reported rather than an average that describes neither. Only 021 and 022 were
measured twice.

**`scan_rate_V_s` is empty for every row and will stay that way.** The exports have no
time column and no instrument metadata beyond a run counter, so the scan rate is not in
the data. The window, step, point count and sweep direction are.

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


## The photocatalyst SMILES are hand-authored

`smiles` in `compound_records_photocatalysts.csv` is the one column in this folder that
nobody measured. Every other value here comes out of an instrument file or a calculation;
a structure comes out of neither. The source spreadsheets record the 31 photocatalysts
only as names — `[Ir(dF(CF3)ppy)2(dtbbpy)][PF6]`, `Ir_dCF3_CF3_ppy2_d55_OMe_bpy` — and a
name is not a machine-readable structure. The ML model in `../../ml/` takes a SMILES and
nothing else, so the photocatalyst side of every pair was simply absent without it.

So the 31 strings in `photocatalyst_smiles.py` were written by hand. That is a weaker
provenance than the rest of the package and it is said here rather than left to be
discovered.

**Convention: donor-pair (ionic) model, one rule for all of them.** Every metal–ligand
bond is a dative bond written `->` toward the metal; the metal carries a formal charge
equal to its oxidation state, `[Ir+3]` or `[Ru+2]`; an anionic donor carries its own
charge, so the cyclometalated aryl carbon is `[c-]` and an N-heterocyclic carbene is
written in its imidazolium-ylide form, `[n+] … [c-]`, with the ylide carbon as the donor.
The net charge then falls out instead of being asserted: `+1` for `[Ir(C^N)2(N^N)]+`, `0`
for `Ir(C^N)3` and for the tris-carbenes, `+2` for `[Ru(N^N)3]2+`. The PF6⁻ counter-ion is
not part of the structure — it is a spectator, it is not in the DFT model, and the species
the model has to see is the cation. Stereochemistry is not encoded, because the model
consumes a molecular graph and the experiments were run on racemates.

**Validated against the DFT geometry, not against the name.** A SMILES that is a plausible
reading of a ligand abbreviation is worth very little: `dF(Me)ppy` and `dF(4-Me)ppy` differ
by one ring position, and PC12/PC25 and PC20/PC21 are pairs of structural isomers with
identical formulas. Each string is therefore checked against
`../computational_data_raw/photocatalysts/PC_<n>/S0/opt_geometry.xyz`, the optimized
ground-state geometry of the species that was actually computed, on four points:

1. molecular formula, hydrogens included, equals the element counts in the xyz;
2. heavy-atom count equals the heavy-atom count of the xyz;
3. net formal charge equals the charge on the `*xyzfile <charge> <mult>` line of the
   matching `opt_input.inp`;
4. the element-labeled heavy-atom connectivity graph of the SMILES is isomorphic to the
   graph derived from the interatomic distances in the xyz — covalent radii for the light
   atoms, a separate 2.6 Å shell for the metal, with the clean gap that cutoff relies on
   checked rather than assumed.

The fourth check is the one that does the work. It pins every ring position, every
substitution pattern and every metal–ligand contact, and it is what turns "a plausible
SMILES for a compound with this name" into "the SMILES of the molecule that was computed".
Swapping the PC12 and PC25 strings, or the PC20 and PC21 strings, or moving one methyl,
fails it. All 31 pass. Any failure aborts the script before it writes, because a wrong
photocatalyst structure does not announce itself downstream — it just moves the predicted
rate.

**Run order.** `photocatalyst_smiles.py` adds a column to a file that
`compound_records.py` writes from scratch, so re-running `compound_records.py`
afterwards drops `smiles` again — silently, because it knows nothing about the
column. Re-running `photocatalyst_smiles.py` restores it and is idempotent.

**Two of the 31 have no row.** PC30 (`Ir(p-F-pmb)3`) and PC31 (`Ir(pmi)3`) were computed
and appear in the photocatalyst spreadsheet, but they were never screened, so
`properties_photocatalysts.xlsx` has no row for them. Their SMILES are validated and
printed by the script; no row is invented for them here, because the row set of this file
belongs to `compound_records.py`.
