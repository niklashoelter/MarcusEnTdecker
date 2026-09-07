# Photocatalysts — computed properties

`photocatalysts_computed_properties.xlsx` — 31 photocatalysts, sorted by id.

Copyedited from `Photocatalysts/computed_PCs - Kopie.xlsx` in the parent project
tree, which is left untouched. 45 columns in, 40 out.

Two sheets:

* **Photocatalysts** — the data, with a grouped and colour-coded header band:
  identity, electronic energies, Gibbs corrections, frequency check, triplet
  energy, reorganization energies, excited-state redox free energies, redox
  potentials vs SCE. Header rows frozen, filter enabled.
* **Column key** — every column with its unit and a one-line description.

## What was changed

**Dropped** (10 columns): the cluster job path, six entirely empty columns, and
the duplicate id column. Nothing carrying data was removed.

**Renamed** so that direction and reference state are explicit. The source sheet
had most headers missing entirely — only 14 of 45 columns were labelled — so
the rest were identified by matching values against `computed_PCs.xlsx`, which
carries the full header row.

**Energies** now say which species is evaluated at which geometry:
`E_T1_at_S0geom_Ha` rather than `S_geometry_T_energy`. The four-point terms say
which direction they belong to: `lambda_S0_to_T1_kcal`, `lambda_T1_to_S0_kcal`.

**Potentials are named by their couple**, which cannot be misread:

| column | couple | meaning |
|---|---|---|
| `E_PCplus_PC_V` | PC⁺/PC | ground-state oxidation |
| `E_PC_PCminus_V` | PC/PC⁻ | ground-state reduction |
| `E_PCstar_PCminus_V` | PC*/PC⁻ | how strong an oxidant PC\* is — `E_ox_ES` downstream |
| `E_PCplus_PCstar_V` | PC⁺/PC* | how strong a reductant PC\* is — `E_red_ES` downstream |

This matters because the downstream naming is quencher-centric: `E_ox_ES` is the
potential at which PC\* *oxidizes a substrate*, i.e. the reduction potential of
the PC\*/PC⁻ couple. Reading it as "the potential at which PC\* is oxidized"
inverts the chemistry.

**One name corrected.** PC1 is listed as `[Ir(dF(CF)3ppy)2][PF6]` in the
compound records — the CF₃ is mistyped and the dtbbpy ligand is missing. The
calculation folder name (`Ir_dF_CF3_ppy2_dtbbpy`) and the experimental property
table both give `[Ir(dF(CF3)ppy)2(dtbbpy)][PF6]`, which is what this file uses.

## Three things worth checking before this is used

**1. Four reported triplet energies do not match their own energies.**
`E_T_reported_kcal` is the value that was carried into the downstream property
tables. `E_T_from_energies_kcal` recomputes the same quantity from the electronic
energies and Gibbs corrections in the same row. They agree for 26 of the 30
photocatalysts with results, to within a rounding-constant difference of
0.0006 kcal mol⁻¹. They disagree for four:

| | discrepancy / kcal mol⁻¹ |
|---|---:|
| PC10 | 2.23 |
| PC13 | 0.70 |
| PC25 | 0.61 |
| PC28 | 0.55 |

Both values are in the file so the difference is visible. E_T feeds ΔG_EnT
directly, so on PC10 this shifts the driving force for all 48 of its pairs by
2.2 kcal mol⁻¹. Either the reported value was overridden deliberately — a better
conformer, a corrected calculation — or it is stale. Worth resolving before
these numbers go into the paper.

**2. The labels in `computed_PCs.xlsx` are swapped.** That sheet calls the
+1.85 V column for PC1 "Reduction Potential" and the −1.41 V column "Oxidation
Potential". They are the other way round: for Ir(ppy)₃ the two values are
+0.88 and −2.34 V against a literature +0.77 and −2.19 V vs SCE. The columns
here are assigned by couple and are correct; the source labels are not.

**3. PC19 has imaginary frequencies** in both the S₀ and the reduced
optimisation. It has a reported triplet energy but no reorganization energies
and no potentials, which is why it drops out of every downstream table. It needs
re-optimising, not re-fitting.

Also present but not acted on: `structure_tag` for PC27 reads `Ir_ppy3_phen`
while the compound is `[Ir(ppy)2(phen)]PF6`. The tag is the calculation folder
name, so it is left as recorded.

## Coverage

| | populated |
|---|---|
| electronic energies, Gibbs corrections | 31 / 31 |
| triplet energy (reported) | 31 / 31 |
| reorganization energies, redox potentials | 30 / 31 (PC19 missing) |
| experimental triplet energy | 18 / 31 |
| source for the experimental value | 4 / 31 |
