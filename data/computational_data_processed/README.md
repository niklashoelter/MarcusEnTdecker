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

### It reproduces the published table, and finds five stale cells in it

Eleven quantities were checked against `computed_PCs.xlsx` across all 31 photocatalysts.
Every one agrees to **6e-5 or better** — the triplet energy, all six reorganization
energies and all four potentials. Two constants were recovered from that comparison rather
than assumed: 627.509 kcal mol⁻¹ Ha⁻¹, and an absolute SCE potential of 4.422 V, both with
essentially no scatter across the set.

The exception is the spreadsheet's `Adiabatic ET` column, which is **not** a consistent
function of the deposited jobs — it scatters where every other column is exact. Five
photocatalysts differ:

| | sheet | recomputed | difference |
|---|---:|---:|---:|
| PC10 | 58.174 | 55.943 | **+2.230** |
| PC13 | 64.969 | 65.670 | −0.701 |
| PC25 | 65.232 | 64.621 | +0.610 |
| PC28 | 79.404 | 79.950 | −0.546 |
| PC19 | 59.169 | 59.237 | −0.069 |

The spreadsheet's own `Adiabatic Energy` column reproduces exactly for all 31, so this is
a stale hand-carried duplicate rather than a disagreement about method. It matters because
`Adiabatic ET` is the column the experimental triplet energies are compared against: for
PC28 the calculated-minus-experimental gap is 3.65 kcal/mol, not the 3.10 the sheet shows.

### Sanity

Ir(ppy)₃ comes out at +0.88 / −2.34 V for the ground-state couples and +0.28 / −1.74 V for
the excited-state ones, within about 0.1 V of the accepted values on three of the four. The
absolute scale is therefore not obviously displaced, though the whole set moves rigidly
with the choice of `SCE_ABSOLUTE_V`.

### PC19 is incomplete and is not a minimum

Four of its ten jobs are absent in the source, so its energy-transfer and hole/electron
reorganization energies are empty. Its S0 and reduced optimizations also carry imaginary
frequencies. `minimum_confirmed = 0` and `note` say both.

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

## Four photocatalysts to resolve before the photocatalyst table is rebuilt

* **PC10, PC13, PC25, PC28** — the reported triplet energy disagrees with the one implied
  by the electronic energies in the same row, by 2.23, 0.70, 0.61 and 0.55 kcal mol⁻¹.
  E_T feeds ΔG_EnT directly, so on PC10 this moves the driving force for all 48 of its pairs.
* **PC19** has imaginary frequencies in both the S₀ and the reduced optimisation. It has a
  triplet energy but no reorganization energies and no potentials, and drops out of every
  downstream table. It needs re-optimising, not re-fitting.
* **PC24** came out at λ_T→S = 0.89 kcal mol⁻¹, impossible for an Ir(ppy) complex — the
  singlet and triplet geometries collapsed onto each other. It quenched nothing across all
  48 quenchers, and a near-zero λ flattens the Marcus barrier, so it produced roughly 30
  false positives on its own.
* **PC20** gave λ_T→S = 33.67 against λ_S→T = 12.70, a 2.7x forward/reverse asymmetry no
  other non-pmb complex shows. Never flagged or verified.
