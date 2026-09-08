# Computational data — processed

Output of `../computational_data_scripts/`. Everything here is generated; nothing is edited
by hand. Deleting this folder and re-running the scripts gives the same files back.

| file | rows | produced by |
|---|---|---|
| `quencher_dft.csv` | 48 | `quencher_dft.py` |
| `quencher_dft_conformers.csv` | 239 | `quencher_dft.py` |
| `quencher_dvte.csv` | 48 | `quencher_dvte.py` |
| `quencher_dvte_samples.csv.gz` | one row per sampled geometry | `quencher_dvte.py` |
| `photocatalyst_dft.csv` | 31 | `photocatalyst_dft.py` |

`photocatalysts_computed_properties.xlsx` and its `.md` are **not** generated, and they are
superseded. The spreadsheet is a copyedit of the published photocatalyst table and the `.md`
is the record of what that copyedit changed. `photocatalyst_dft.csv` now rebuilds all of it
from the job files, so nothing reads either any more.

They are kept rather than deleted because the spreadsheet is the only copy in the tree, and
they sit here rather than in `../computational_data_raw/photocatalysts/`, where they used to,
because that directory is for job files the extraction scripts read and these are neither.

## Still to come

| file | source | blocked on |
|---|---|---|
| `entdecker_predictions.csv` | EnTdecker 1.0 (E_T) and 2.0 (E_T + lambdas) | not in the tree at all |

## `photocatalyst_dft.csv`

One row per photocatalyst, 31 of them, from the four-point scheme: S0, T1, oxidized and
reduced optimizations plus the six single points that cross them over.

```
E_T                                  adiabatic triplet energy, Gibbs-corrected
lambda_S0_to_T1, lambda_T1_to_S0     energy-transfer reorganization energies
lambda_S0_to_ox, lambda_ox_to_S0     hole-transfer reorganization energies
lambda_S0_to_red, lambda_red_to_S0   electron-transfer reorganization energies
E_PCplus_PC, E_PC_PCminus            ground-state couples, V vs SCE
E_PCstar_PCminus, E_PCplus_PCstar    excited-state couples, V vs SCE
```

Potentials are named by their couple and never by direction. `E_PCstar_PCminus` is how
strong an oxidant the triplet is; calling it "the excited-state oxidation potential" is
how its sign gets flipped downstream, and the source spreadsheet has exactly that pair of
labels crossed.

### It reproduces the published table

Every column of this file was matched against every column of `computed_PCs.xlsx`, across
all 31 photocatalysts. Twenty-two quantities have a counterpart and all twenty-two agree:
the triplet energy, all six reorganization energies and all four potentials to **3e-5 or
better**, the ten raw electronic energies to 5e-10 Ha, the four Gibbs corrections exactly.

Two constants were recovered from that comparison rather than assumed: 627.509
kcal mol⁻¹ Ha⁻¹, and an absolute SCE potential of 4.422 V, both with essentially no
scatter across the set. The match also confirms which spreadsheet label means what:
`E_PCplus_PC` is the sheet's "Reduction Potential" and `lambda_S0_to_T1` its
"lambda_donor_triplet" — the sheet's own naming is crossed in both places.

The one column that does **not** reproduce is `Adiabatic ET`, stale in five cells (PC10 by
+2.23, PC13/25/28 by 0.5–0.7, PC19 by 0.07 kcal/mol). Its sibling `Adiabatic Energy`
reproduces exactly for all 30 rows that carry it, so this is a hand-carried duplicate that
fell behind rather than a disagreement about method. **Nothing here or downstream reads
it.** It existed to compare against literature triplet energies, a benchmark of the method
that this work does not make, so it is noted and left alone.

### Sanity

Ir(ppy)₃ comes out at +0.88 / −2.34 V for the ground-state couples and +0.28 / −1.74 V for
the excited-state ones, within about 0.1 V of the accepted values on three of the four. The
absolute scale is therefore not obviously displaced, though the whole set moves rigidly
with the choice of `SCE_ABSOLUTE_V`.

### PC19 is incomplete

Four of its ten jobs were never run — `S_geometry_T_energy`, `S_geometry_ox_energy`,
`S_geometry_red_energy` and `red_geometry_S_energy` — so four of its reorganization
energies are empty and its 48 pairs drop out of every barrier method downstream. No
re-optimisation fixes that; the jobs have to be run. Open with Leon.

Its S0 and reduced optimisations also carry an imaginary mode, and `minimum_confirmed = 0`
records it, but the mode is **−5.44 cm⁻¹** and **−10.02 cm⁻¹** — a floppy rotor, not a
saddle point. These are the only imaginary modes across all 124 optimisations in the
deposit. The geometry is converged and the numbers are usable.

The spreadsheet's PC19 singlet energy is 7.8e-7 Ha from the one here, the only Ha
disagreement in the whole comparison worse than 1e-9. That is the sheet lagging the data:
`PC_19/singlet_opt` was re-run on 2026-04-24, after the sheet was filled, and the re-run
is what is deposited.

## `quencher_dft.csv`

One row per quencher, in kcal/mol except the two potentials in V vs SCE.

```
E_T                                adiabatic triplet energy
lambda_S_T, lambda_T_S             energy-transfer reorganization energies
dG_Ox, E_Ox_vs_SCE_calc            oxidation
lambda_S0_D+, lambda_D+_S0         hole-transfer reorganization energies
dG_Red, E_Red_vs_SCE_calc          reduction
lambda_S0_D-, lambda_D-_S0         electron-transfer reorganization energies
```

517 of the 520 published values reproduce exactly. The three that do not are errors in the
published spreadsheet, each a potential that contradicts the free energy printed beside it
in its own row: Q9 1.41 against an implied 1.40, Q14 2.46 against 2.45, Q18 −2.54 against
−2.53. The values here are the implied ones. See the scripts README.

**Q10 and Q11 have no reduced state.** There are no `red/` jobs for them, so their four
reduction columns are empty. The published table is empty there too.

**No `_sim` columns.** The original also computed every reorganization energy for the
conformer geometrically closest to the lowest-energy S0 structure. Those are not produced
here: nothing in the project reads them, neither assembled dataset carries them, and their
published values did not reproduce from the geometries.

`quencher_dft_conformers.csv` is the per-conformer intermediate — the raw ORCA energies
before any conformer is selected — with a `runs` column naming the job prefixes that fed
each row, so a rerun is visible rather than silently folded in.

## `quencher_dvte.csv`

The distribution of vertical singlet–triplet gaps sampled along the Milo trajectories.

```
dvte_mu, dvte_std      mean and standard deviation of the sampled gaps, kcal/mol
dvte_<p>               mu + Phi^-1(p) * std, for p in 0.005, 0.008, 0.01, 0.015
dvte_<p>_kde           the same quantile from a Gaussian KDE of the samples
```

Every column reproduces `Quencher/DvTE/milo_summary_detailed.csv`, from 236482 sampled
geometries across the 48 quenchers:

| column | max difference | median |
|---|---:|---:|
| `dvte_mu`, `dvte_std` | 0.005 | 0.003 |
| `dvte_0.005` … `dvte_0.015` | 0.009 | 0.003 |
| `dvte_*_kde` | 0.062 | 0.025 |

all in kcal/mol. The published file carries two decimals, so the first two rows are its
own rounding. The `_kde` agreement also settles the bandwidth question: the original's
choice is not recorded anywhere, and Scott's rule matches it to within 0.07.

`quencher_dvte_samples.csv.gz` holds the per-step singlet and triplet energies and their
gap. It exists so that nothing downstream ever has to read the 3.4 GB archive again.

## Sign conventions for the pair-level quantities

Computed in `../../dataset/pairs.csv`, not here — this is the convention it uses. All
quencher-centric, negative = exergonic, F = 23.0605 kcal mol⁻¹ V⁻¹:

```
dG_EnT     = E_T(Q) - E_T(PC)
dG_SET_ox  = F * (E_Ox(Q) - E_ox_ES(PC))      Q oxidized, PC* is the oxidant
dG_SET_red = F * (E_red_ES(PC) - E_Red(Q))    Q reduced,  PC* is the reductant
dG_SET     = whichever is more accessible, with dG_SET_channel naming it
```

Potentials are named by their couple, not by direction, because the downstream naming is
quencher-centric and inverts easily: `E_ox_ES` is the potential at which PC\* *oxidizes a
substrate*, i.e. the reduction potential of the PC\*/PC⁻ couple.

Note that `23.061` in `quencher_dft.py` and `23.0605` here are different constants in the
source material. Both are kept as they were rather than reconciled.

## Photocatalysts still to resolve

* **PC19** is missing four single points that were never run, so four of its reorganization
  energies are empty and its 48 pairs are silently predicted "does not quench" by every
  barrier method. Open with Leon. Its imaginary modes are *not* the problem — see above.
* **PC24** came out at λ_T→S = 0.89 kcal mol⁻¹, impossible for an Ir(ppy) complex — the
  singlet and triplet geometries collapsed onto each other. It quenched nothing across all
  48 quenchers, and a near-zero λ flattens the Marcus barrier, so it produced roughly 30
  false positives on its own. It is excluded from the assembled dataset.
* **PC20** gave λ_T→S = 33.67 against λ_S→T = 12.70, a 2.7x forward/reverse asymmetry no
  other non-pmb complex shows. Never flagged or verified.

PC10, PC13, PC25 and PC28 were on this list because of the `Adiabatic ET` column. They are
off it: that column is stale and unread, and their triplet energies recompute exactly.
