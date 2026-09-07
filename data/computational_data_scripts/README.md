# Computational data — scripts

Each script reads `../computational_data_raw/`, writes `../computational_data_processed/`,
and touches nothing else. Run with `uv run <script>.py` — no arguments, no install step.
Dependencies are declared in a PEP 723 header at the top of each file.

| script | reads | writes | status |
|---|---|---|---|
| `quencher_dft.py` | `quencher_adiabatic_DFT/` (48 quenchers, 2467 files) | `quencher_dft.csv`, `quencher_dft_conformers.csv` | **verified** — 517/520 published values, the 3 exceptions being spreadsheet errors |
| `quencher_dvte.py` | `quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz` | `quencher_dvte.csv`, `quencher_dvte_samples.csv.gz` | **verified** — all 10 columns reproduce |
| `photocatalyst_dft.py` | `photocatalysts/` | — | blocked, folder is empty |

## `quencher_dft.py`

The per-quencher arithmetic is a port of
`Quencher/Adiabatic DFT Pipeline 5/extract_parameters_T1.py`, which produced the published
values. That script began from a spreadsheet of per-conformer energies; the step that built
that spreadsheet from the ORCA outputs was never in the tree, so it is implemented here and
written out as `quencher_dft_conformers.csv`.

### What is read from where

```
E_singlepoint       last "FINAL SINGLE POINT ENERGY" in <conf>_sp_output.out
E_correction        last "G-E(el)" (the Eh field) in <conf>_opt_output.out
E_combined          E_singlepoint + E_correction
E_vert_*            last "FINAL SINGLE POINT ENERGY" in <conf>_sp_vert_*_output.out
```

Per state, the conformer with the lowest `E_combined` supplies the values.

The original also produced a `_sim` variant of every reorganization energy, using the
conformer geometrically closest to the lowest-energy S0 structure rather than the
lowest-energy one. Those columns are not produced here: nothing in the project reads them,
neither assembled dataset carries them, and their published values did not reproduce from
the geometries. Dropping them also removed the only dependency, so this script is pure
standard library.

```
E_T          = E_combined(T1)  - E_combined(S0)
lambda_S_T   = E_vert_spin_change(T1) - E_singlepoint(S0)
lambda_T_S   = E_vert_spin_change(S0) - E_singlepoint(T1)
dG_Ox        =  E_combined(ox)  - E_combined(S0)
dG_Red       = -(E_combined(red) - E_combined(S0))
lambda_S0_D+ = E_vert_red(ox) - E_singlepoint(S0)
lambda_D+_S0 = E_vert_ox(S0)  - E_singlepoint(ox)
lambda_S0_D- = E_vert_ox(red) - E_singlepoint(S0)
lambda_D-_S0 = E_vert_red(S0) - E_singlepoint(red)
E (V vs SCE) = dG/23.061 - 4.442
```

### Rounding

The original's roundings are preserved deliberately, because they define the published
numbers: **1 decimal in kcal/mol, 2 decimals in V**. This is the one place in the data
package that does not follow the six-significant-figure rule. Note that `23.061` here and
`23.0605` in the driving-force expressions are different constants in the source material
and are both kept as they were.

### Reruns

A few quenchers were computed more than once and their jobs carry a calculation prefix as
well as a conformer number — Q17's S0 holds `calc1_conf1` and `calc3_conf1`, Q18's holds
`calc1_conf1` and `calc5_conf1`. These are one conformer computed twice, not two
conformers, and must not compete against each other for lowest energy.

A rerun is treated as **filling gaps, not replacing**: the earliest calculation supplies
the values and any quantity it lacks is taken from the next run that has it. Q18 is the
case that fixes this rule — its `calc1` never ran the reduced-state vertical single point,
which `lambda_D-_S0` needs, while every other published quantity for that quencher follows
`calc1` rather than `calc5`. Taking the latest run wholesale instead reproduces four fewer
published values.

### Agreement with the published table

517 of the 520 published values reproduce exactly. The three that do not are errors in the
published spreadsheet, not here: each is a potential that contradicts the free energy
printed beside it in its own row, and recomputing from the ORCA outputs gives the value
that ΔG implies.

| | ΔG published | E published | E implied by that ΔG, and computed here |
|---|---:|---:|---:|
| Q9 | 134.8 | 1.41 | **1.40** |
| Q14 | 159.0 | 2.46 | **2.45** |
| Q18 | 44.0 | −2.54 | **−2.53** |

One quencher needed an explicit exception. Q17's published `dG_Red` of 54.2 is only
reproduced with `calc3`'s thermal correction, while every other published Q17 quantity
needs `calc1`'s single points. There is no rule in the data that predicts this, so it is
recorded as a named entry in `RERUN_FIELD_SOURCE` at the top of the script rather than
dressed up as one. Removing that entry gives 54.1 and shifts `E_Red_vs_SCE_calc` to −2.10.

### Two quenchers have no reduced state

Q10 and Q11 have no `red/` jobs at all, so `dG_Red`, `E_Red_vs_SCE_calc` and the four
reduction reorganization energies are empty for them. The published table is empty there
too.

## `quencher_dvte.py`

At every sampled step of every Milo trajectory, ORCA was run twice on the same geometry --
`output_1.out` as the singlet, `output_3.out` as the triplet, both M062X/6-311++G(d,p) with
CPCM(acetonitrile). The vertical gap is the difference of their final single point
energies, and the collection of those gaps is the DvTE distribution. One pass over the
archive reads 472964 outputs and yields 236482 paired geometries, about 4900 per quencher.

```
dvte_mu, dvte_std   mean and standard deviation of the sampled gaps
dvte_<p>            mu + Phi^-1(p) * std           normal quantile
dvte_<p>_kde        the same quantile from a Gaussian KDE, Scott's bandwidth
```

The definitions are not documented anywhere and were recovered by testing candidates
against `Quencher/DvTE/milo_summary_detailed.csv`, which collects the values from
per-quencher `dvte_*.txt` files that no longer exist. All ten columns reproduce: mu and std
to 0.005 kcal/mol, the plain thresholds to 0.009, the KDE thresholds to 0.062. The first
two are the published file's own two-decimal rounding.

### The archive is not a duplicate

`quencher_adiabatic_DvTE/Milo_MD/` as unpacked contains only the trajectory jobs --
`job.in`, `job.out`, `job.xyz`, `milo.in`, the scheduler log and `submit.cmd`. The per-step
`step_<n>/` folders holding the two ORCA outputs exist ONLY inside
`quenchers_vertical_gaps.tar.xz` (3.4 GB, ~1.95 million entries). That archive is the only
surviving record of the per-step energies and must not be deleted.

It is read as a stream, so nothing is unpacked to disk, at the cost of decompressing the
whole file on each run -- about five minutes. The per-step gaps are written out to
`quencher_dvte_samples.csv.gz` (4.7 MB) so that nothing downstream needs to read the
archive again.
