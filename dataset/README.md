# dataset — the pair-level table

Where processed measurements and processed calculations become one row per
photocatalyst–quencher pair. Everything downstream — benchmarks, models, figures —
reads `pairs.csv` and nothing else.

```
data/experimental_data_processed/   ┐   stern_volmer_fits.csv
                                    │   photocatalyst_properties_exp.xlsx
                                    │   quencher_properties_exp.xlsx
                                    ├──> build_dataset.py ──> pairs.csv   (1248 rows)
data/computational_data_processed/  ┘   photocatalyst_properties_calc.xlsx
                                        quencher_properties_calc.xlsx
```

Nothing is read from outside the processed folders of `data/`, and no file is copied in
as an import side effect. The workbooks are generated, not curated: each one's `Column
key` sheet names the intermediate table in `data/*_raw/` that every column was copied
from.

## The central quantity

```
k_q = K_SV / tau_0
```

`K_SV` per pair from `stern_volmer_fits.csv`, `tau_0` per photocatalyst from
`photocatalyst_properties_exp.xlsx`. Both measured here, both traceable to instrument
files.

**The label is `quenches`: K_SV > 10, or K_SV ≤ 10 with a slope significant across the
points the fit used. 790 pairs quench, 458 do not.**

## The label is set on K_SV, not on k_q

k_q is the physically meaningful quantity and it is still carried, but it cannot define
the ground truth, because `k_q = K_SV / tau_0` divides by a photocatalyst-specific
lifetime. The Stern–Volmer detection floor is fixed in K_SV; converted to a rate it
becomes `K_SV_floor / tau_0`, which moves with the photocatalyst. Under the old
`k_q > 1e7` cutoff that floor sat **above** the cutoff for 19 of the 26 photocatalysts:

| | τ₀ / ns | floor at K_SV = 25 | vs. a 1e7 cutoff |
|---|---:|---:|---:|
| PC10 | 74.7 | 3.35e8 M⁻¹ s⁻¹ | **33× above** |
| PC17 | 277.4 | 9.01e7 | 9× above |
| PC2 | 557.1 | 4.49e7 | 4.5× above |
| PC1 | 2387.8 | 1.05e7 | 1.0× above |

For 912 of the 1248 pairs, "did not quench" therefore did not mean `k_q < 1e7`. It meant
nothing had been shown above a floor well above 1e7 — PC10 could have had a pair running
at 3×10⁸ M⁻¹ s⁻¹ and still recorded it as a non-quencher. 465 pairs could not be labelled
at all, and the benchmark was scoring specificity partly against pairs that carried no
information.

On K_SV the floor is one number for every photocatalyst, every pair is labelled by the
same criterion, and the indeterminate class does not arise. The cost is stated rather
than hidden: τ₀ spans 86× across the screen, so a fixed K_SV cutoff does correspond to
different rate constants for different photocatalysts. That is the trade — a threshold on
what was measured, whose conversion to a rate varies, against a threshold on a derived
rate that half the screen could not have detected.

## Why the rule has two parts

The published criterion was `K_SV ≥ 25 and r² ≥ 0.90` — recovered exactly, 1296 of 1296,
from the acquisition-time summary files. It is kept as `quenches_strict`.

The threshold moved to 10 because weak quenching in the K_SV 5–25 band is real signal
here rather than noise. Nineteen pairs with K_SV ≤ 10 clear r² > 0.95 where a
selection-aware null — best 5 of 6 points, no true slope — expects 1.1, and their traces
are monotone 3–4 % intensity drops well above the read noise.

The rescue below 10 is a **slope-significance test, not an r² cut**, because the fitted r²
is not comparable between pairs. `stern_volmer_fits.py` only runs its outlier scan when
the all-points r² clears `R2_NO_QUENCHING = 0.50`, so a pair sitting at 0.45 can never be
cleaned while one at 0.55 can be cleaned to 0.99. The result is stark:

| low-K_SV pairs | n | r²(fit) > 0.95 | p_slope < 0.01 |
|---|--:|--:|--:|
| a point was dropped | 50 | 19 | 28 |
| none dropped | 103 | **0** | 11 |

An r²-based rescue would therefore have selected on whether the pipeline happened to drop
a point. The slope test reads the points the fit used and applies a Bonferroni penalty
over the drop candidates when one was dropped, which makes trimmed and untrimmed pairs
comparable. 26 pairs are carried by it.

It runs on the fit's own points and not on all six deliberately: the range scan exists to
cut a collapsed well loose, and putting it back lets one bad reading carry the label.
PC11/Q44 fits at K_SV = 0.67 on its own points and 7276 across all six, because its last
well reads near zero.

## What the two labels cost

`quenches` (790) and `quenches_strict` (634) differ on 156 pairs, and the benchmark scores
both. Every method scores **better** against the strict label — λ-scaled symmetric Marcus
reaches MCC 0.604 strict against 0.474 inclusive, the adiabatic rule 0.462 against 0.373.
That gap is not an argument for the strict label. It says the 156 weak quenchers are the
hard cases, which is what a marginal quencher should be, and a criterion that scores well
by excluding them has not predicted them.

Which one the paper claims is a scientific question about what "quenching" means. Neither
this table nor the benchmark takes a position; both are always computed and reported.


## Two things this table refuses to smooth over

**458 pairs (36.7%) are left-censored.** They fall below the label,
and 331 of them have a negative fitted slope. Every one is present with `censored = 1`
and its fitted `K_SV` intact. They are one-sided observations, not missing values, and
`K_SV_floor` is the limit a survival analysis needs.

`k_q_floor` is the same limit as a rate, the smallest rate constant each pair could have
shown: `K_SV_floor / tau_0`. It runs from **1.55e+06 to 1.34e+08 M⁻¹ s⁻¹ —
an 86× range**, because τ₀ does. That column is per-row for exactly that reason,
and its spread is why the label is not set on k_q.

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

## The driving forces

Both channels a triplet photocatalyst can quench through, per pair. All quencher-centric,
all negative when exergonic, `F = 23.0605 kcal mol⁻¹ V⁻¹`:

```
dG_EnT     = E_T(Q) - E_T(PC)
dG_SET_ox  = F * (E_Ox(Q)  - E(PC*/PC⁻))     Q oxidized, PC* is the oxidant
dG_SET_red = F * (E(PC⁺/PC*) - E_Red(Q))     Q reduced,  PC* is the reductant
dG_SET     = whichever is more accessible, with dG_SET_channel saying which
```

Potentials are named by their couple, not by direction, because the naming here is
quencher-centric and inverts easily: `E(PC*/PC⁻)` is the potential at which PC\* *oxidizes
a substrate*, which is the **reduction** potential of the excited state.

Both SET columns exist twice, `_exp` from the measured quencher potentials and `_calc`
from the computed ones. They are never merged. A measured potential and a DFT potential
are not interchangeable, and which one an analysis used is part of what it reports.

`dG_EnT` behaves the way the experiment says it should, which is the first real check on
the photocatalyst side of the calculation:

| | n | median `dG_EnT` | exergonic |
|---|---:|---:|---:|
| established quenchers (`positive`) | 627 | −9.4 kcal/mol | 94 % |
| established non-quenchers (`negative`) | 156 | +0.3 kcal/mol | 49 % |

### Bounded potentials become bounded driving forces

A quencher whose oxidation is recorded as `> 2.0` puts a **floor** under `dG_SET_ox`, and
one whose reduction is recorded as `< -2.10` puts a floor under `dG_SET_red`. Both point
the same way, so `dG_SET_ox_bound = lower` means SET is *at least* this endergonic and the
true value can only be worse.

**165 pairs have a `dG_SET` that is a lower bound in exactly this sense.** Reading those at
face value understates how inaccessible the electron-transfer channel is — in the direction
that would wrongly exclude pairs from the energy-transfer set, which is the specific
mistake the bound flags exist to prevent.

## PC24 is excluded, and not by any threshold

`pairs.csv` is 26 photocatalysts × 48 quenchers, not 27 × 48. PC24 was measured, its 48
Stern–Volmer fits exist in `stern_volmer_fits.csv`, and they are dropped here by name.

This is a scientific decision taken outside the pipeline and it is worth being plain that
no data-quality rule produces it: PC24's median r² is 0.677, which is unremarkable —
PC6 (0.472), PC2 (0.554) and PC27 (0.651) are worse and are kept — and its τ₀ rests on a
single measurement, as 24 of the 27 do. The exclusion is recorded as
`EXCLUDED_PHOTOCATALYSTS` in `build_dataset.py` with its reason, so that it can be
disagreed with rather than discovered.

## What is still not here

The photocatalyst reorganization energies are carried per pair but no **combined** λ is
formed. Marcus theory needs the pair's reorganization energy, and how the two partners'
terms add is a modelling decision — it belongs in `benchmark/`, where it can be stated and
varied, not frozen into the table.

## Columns

| group | columns |
|---|---|
| identity | `pc_id`, `pc_name`, `pc_compound`, `quencher_id`, `quencher_name`, `quencher_compound`, `quencher_smiles` |
| measurement | `K_SV / M-1`, `r2`, `n_points_fitted`, `n_points_measured`, `fit_method`, `fit_note`, `tau0 / ns`, `tau0_rsd`, `tau0_consistent` |
| rate | `k_q / M-1 s-1`, `log10_k_q`, `quenches`, `quenches_strict`, `censored`, `k_q_floor / M-1 s-1`, `K_SV_floor / M-1` |
| quencher, experimental | `q_E_Ox_vs_SCE_exp / V`, `q_E_Ox_bound`, `q_E_Red_vs_SCE_exp / V`, `q_E_Red_bound`, `q_redox_source`, `q_epsilon_at_pc_excitation / mM-1 cm-1`, `pc_excitation / nm` |
| quencher, computed | `q_E_T_dft`, `q_lambda_S_T_dft`, `q_lambda_T_S_dft`, `q_E_Ox_vs_SCE_calc`, `q_E_Red_vs_SCE_calc`, `q_dvte_mu_md`, `q_dvte_std_md` |
| photocatalyst, computed | `pc_E_T_dft`, `pc_lambda_S0_to_T1_dft`, `pc_lambda_T1_to_S0_dft`, `pc_E_PCstar_PCminus`, `pc_E_PCplus_PCstar` |
| driving forces | `dG_EnT`, `dG_SET_ox_exp`, `dG_SET_ox_exp_bound`, `dG_SET_red_exp`, `dG_SET_red_exp_bound`, `dG_SET_exp`, `dG_SET_exp_bound`, `dG_SET_exp_channel`, `dG_SET_ox_calc`, `dG_SET_red_calc`, `dG_SET_calc`, `dG_SET_calc_channel` |

`_exp` and `_calc` redox potentials are kept as separate columns and never merged. A
measured potential and a DFT potential are not interchangeable, and which one an
analysis used is part of what it has to report.

`q_epsilon_at_pc_excitation` is the quencher's extinction coefficient at the wavelength
its photocatalyst was excited at, for judging inner-filter effects. The paper's K_SV
values are **uncorrected**; this column is what a correction would be based on, not
evidence that one was applied.

## Coverage

| | of 1248 |
|---|---|
| `k_q` | 1248 |
| quencher SMILES | 1248 |
| quencher DFT `E_T` | 1248 |
| quencher DvTE | 1248 |
| experimental `E_Ox` | 1092 (6 quenchers have none) |
| photocatalyst DFT `E_T` | 1248 |
| `dG_EnT` | 1248 |
| `dG_SET` | 1248 |
