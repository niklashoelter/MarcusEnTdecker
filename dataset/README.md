# dataset — the pair-level table

Where processed measurements and processed calculations become one row per
photocatalyst–quencher pair. Everything downstream — benchmarks, models, figures —
reads `pairs.csv` and nothing else.

```
data/experimental_data_processed/   ┐
                                    ├──> build_dataset.py ──> pairs.csv   (1296 rows)
data/computational_data_processed/  ┘
```

Nothing is read from outside `data/`. No XLSX is consulted at run time and no file is
copied in as an import side effect.

## The central quantity

```
k_q = K_SV / tau_0
```

`K_SV` per pair from `stern_volmer_fits.csv`, `tau_0` per photocatalyst from
`lifetimes_tau0.csv`. Both measured here, both traceable to instrument files.

**At the 1e7 M⁻¹ s⁻¹ cutoff, 695 pairs quench and 601 do not.**

That is one more than the 694 the earlier analysis reported, and the difference is
fully accounted for: PC4's τ₀ is now derived from its TCSPC traces (1228.49 ns) rather
than taken from a curated value that matches none of them (1494.54 ns). The shorter
lifetime raises k_q by 1.22× for all 48 of PC4's pairs, and exactly one of them —
**PC4/Q45**, at K_SV = 12.78 — crosses the threshold, moving from 8.5e6 to 1.04e7.
It sits on the boundary either way and should not be leaned on.

## The 1e7 threshold is below the detection limit for 20 of 27 photocatalysts

This is the most consequential thing in the table and it needs a decision.

The ground truth is defined by a single global cutoff, `k_q > 1e7 M⁻¹ s⁻¹`. The
Stern–Volmer detection floor is `K_SV >= 25`, which in rate terms is `25 / tau_0` —
photocatalyst-specific, and for a short-lived photocatalyst it sits **above** the
global cutoff:

| | τ₀ / ns | floor / M⁻¹ s⁻¹ | vs. the 1e7 cutoff |
|---|---:|---:|---:|
| PC10 | 74.7 | 3.35e8 | **33× above** |
| PC17 | 277.4 | 9.01e7 | 9× above |
| PC2 | 557.1 | 4.49e7 | 4.5× above |
| … | | | |
| PC1 | 2387.8 | 1.05e7 | 1.0× above |

Twenty of the 27 photocatalysts are in that list, covering **960 of the 1296 pairs**.
For those, "did not quench" does not mean `k_q < 1e7`. It means `k_q` was not shown to
exceed a floor that is itself well above 1e7 — PC10 could have a pair at 3×10⁸ M⁻¹ s⁻¹
and still record it as a non-quencher.

Sorting the pairs by what the experiment actually established:

| `label_class` | count | meaning |
|---|---:|---|
| `positive` | 627 | above the cutoff **and** quenching was actually detected |
| `negative` | 156 | below the cutoff **and** this photocatalyst could have detected it |
| `indeterminate` | 513 | the measurement does not answer this question |

**Only 156 of the 601 apparent non-quenchers are established as non-quenchers.** The
other 445 are indeterminate, and 68 pairs currently labelled quenchers never passed the
detection criterion at all.

That matters because the benchmark scores every method against this negative class. A
classifier that predicts "quencher" for one of those 445 is not necessarily wrong — the
experiment cannot say — but it is currently scored as a false positive. Specificity, and
therefore MCC, is being measured partly against pairs that carry no information.

`label_class` and `threshold_resolvable` are written so this is visible rather than
buried. Nothing is dropped: all 1296 rows are present with `quenches` and `censored`
unchanged, so the previous labelling is still exactly reproducible.

**The options, none of which I have taken:**

1. **Restrict the benchmark to `label_class != indeterminate`** — 783 pairs, honest but
   loses 40 % of the screen and most of the short-lived photocatalysts.
2. **Raise the cutoff to the worst floor** (3.35e8, set by PC10) — makes every negative
   real, but reclassifies a large number of genuine quenchers as negatives.
3. **Use a per-photocatalyst cutoff** at each one's own floor — keeps every pair, but
   then the label means something different for each photocatalyst, which is the exact
   problem that moving from K_SV to k_q was meant to solve.
4. **Censored regression on k_q** with `k_q_floor` as the detection limit — uses all
   1296 pairs and all the information in them, and is what the data is actually shaped
   for. It changes the benchmark from classification to regression.

My recommendation is (4), with (1) reported alongside as the classification sensitivity
check. But this is a scientific call about what the paper claims, not a data-processing
one, so the table carries the information and takes no position.

## Two things this table refuses to smooth over

**Half the dataset is left-censored.** 662 pairs (51.1 %) fall below the Stern–Volmer
detection floor, and 379 of those have a negative fitted slope. Every one is present
with `censored = 1` and its fitted `K_SV` intact. They are one-sided observations, not
missing values.

`k_q_floor` gives the smallest rate constant each pair could have shown, `25 / tau_0`.
It runs from **3.9e6 to 3.4e8 M⁻¹ s⁻¹ — an 86× range**, because τ₀ does. That column
is what a survival analysis needs, and it is per-row for exactly that reason.

**Nineteen quenchers have a redox potential that is a bound, not a value.** Recorded as
`> 2.0` or `< -2.10` in the source, meaning the compound was not oxidized or reduced
within the accessible window. `q_E_Ox_bound` and `q_E_Red_bound` carry that as `lower`
or `upper`; the number is still there because it is the only quantitative information
available, but it never travels without the flag.

This matters for mechanism assignment specifically. Ruling out single-electron transfer
works by comparing these potentials against the photocatalyst's, and reading `> 2.0` as
`2.0` turns "we could not oxidize this" into "this oxidizes at 2.0 V" — which makes SET
look more feasible than the data supports, in the direction that would wrongly exclude
pairs from the energy-transfer set.

## What is not here yet

Every photocatalyst-side computed quantity: `E_T(PC)`, its reorganization energies, and
its excited-state redox potentials. The photocatalyst DFT job files are not in the
repository, and this package does not inherit numbers from a spreadsheet it cannot
regenerate.

So the pair-level driving forces — `dG_EnT`, `dG_SET_ox`, `dG_SET_red` — are **not
computed and no column for them is written**. They are not blank cells awaiting a
merge; they are quantities this table is not yet entitled to state. The quencher side
is complete, so they become a short addition once those job files arrive.

The sign conventions they will use, all quencher-centric, negative = exergonic,
`F = 23.0605 kcal mol⁻¹ V⁻¹`:

```
dG_EnT     = E_T(Q) - E_T(PC)
dG_SET_ox  = F * (E_Ox(Q) - E_ox_ES(PC))      Q oxidized, PC* is the oxidant
dG_SET_red = F * (E_red_ES(PC) - E_Red(Q))    Q reduced,  PC* is the reductant
dG_SET     = min(dG_SET_ox, dG_SET_red)
```

Potentials are named by their couple, not by direction, because the naming is
quencher-centric and inverts easily: `E_ox_ES` is the potential at which PC\*
*oxidizes a substrate*, i.e. the reduction potential of the PC\*/PC⁻ couple.

## Columns

| group | columns |
|---|---|
| identity | `pc_id`, `pc_name`, `pc_compound`, `quencher_id`, `quencher_name`, `quencher_compound`, `quencher_smiles` |
| measurement | `K_SV / M-1`, `r2`, `n_points_fitted`, `n_points_measured`, `fit_method`, `fit_note`, `tau0 / ns`, `tau0_rsd`, `tau0_consistent` |
| rate | `k_q / M-1 s-1`, `log10_k_q`, `quenches`, `censored`, `k_q_floor / M-1 s-1`, `label_class`, `threshold_resolvable` |
| quencher, experimental | `q_E_Ox_vs_SCE_exp / V`, `q_E_Ox_bound`, `q_E_Red_vs_SCE_exp / V`, `q_E_Red_bound`, `q_redox_source_type`, `q_redox_doi`, `q_epsilon_at_pc_excitation / mM-1 cm-1`, `pc_excitation / nm` |
| quencher, computed | `q_E_T_dft`, `q_lambda_S_T_dft`, `q_lambda_T_S_dft`, `q_E_Ox_vs_SCE_calc`, `q_E_Red_vs_SCE_calc`, `q_dvte_mu_md`, `q_dvte_std_md` |

`_exp` and `_calc` redox potentials are kept as separate columns and never merged. A
measured potential and a DFT potential are not interchangeable, and which one an
analysis used is part of what it has to report.

`q_epsilon_at_pc_excitation` is the quencher's extinction coefficient at the wavelength
its photocatalyst was excited at, for judging inner-filter effects. The paper's K_SV
values are **uncorrected**; this column is what a correction would be based on, not
evidence that one was applied.

## Coverage

| | of 1296 |
|---|---|
| `k_q` | 1296 |
| quencher SMILES | 1296 |
| quencher DFT `E_T` | 1296 |
| quencher DvTE | 1296 |
| experimental `E_Ox` | 1134 (6 quenchers have none) |
