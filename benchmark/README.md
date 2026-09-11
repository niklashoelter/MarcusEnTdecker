# benchmark — physical-organic theory as a predictor

How well do adiabatic thermodynamics, Marcus theory, Rehm–Weller/Balzani, Sandros and
the DvTE distribution predict whether a pair quenches, and how fast? Each is scored
against the same ground truth on the same pairs, with the same sweep discipline, and
each carries the number of free parameters it needed to get there.

```
uv run benchmark/run_benchmark.py
```

Reads `../dataset/pairs.csv` and nothing else. Writes four tables into `results/`.

## Layout

| file | what it is |
|---|---|
| `pairs.py` | the only place a column name of `pairs.csv` appears; also what each one means |
| `theory.py` | the shared interface, the Balzani rate equation, and the λ-scaling grid |
| `adiabatic.py` | ΔG_EnT < 0, the zero-parameter baseline |
| `marcus.py` | the two-parabola barrier and the rate built on it |
| `rehm_weller.py` | the empirical barrier and the rate built on it |
| `sandros.py` | the barrierless pre-equilibrium null model |
| `dvte.py` | the quencher's triplet energy as a sampled distribution |
| `scoring.py` | metrics, leave-one-photocatalyst-out, censoring-aware rate agreement |
| `run_benchmark.py` | the harness |

Each theory is a module with the same five entry points — `required_columns`,
`prepare`, `grid`, `classify`, `rate_grid`/`predicted_rate` — so adding one means adding
a file and one import. Each runs on its own (`uv run benchmark/marcus.py`), and each
declares its dependencies in a PEP 723 header like everything else in the repository.
Nothing manipulates `sys.path`; the modules sit beside each other in one directory,
which is all a plain `import` needs.

## Where this came from

Ported from `phys_org_theories/`, which produced the group's earlier version of these
numbers. The physics is carried over as it stood. What changed is everything around it:

- **The data layer is gone.** The original's `constants` module copied four XLSX files
  in from a sibling directory as a side effect of being imported, which made its results
  depend on a path outside the project and on whatever state that path happened to be
  in. This reads `dataset/pairs.csv`, once, with no side effect.
- **Driving forces come from the table** rather than being recomputed from triplet
  energies at every call site.
- **The per-row scalar branches are masks**, so the whole table evaluates at once. The
  arithmetic is unchanged.
- **The reorganization-energy pairing was corrected.** See below; this is the one place
  the port is not faithful, deliberately.

The original's own audit directory recorded several things as unsound, and none of them
are here: the `ΔG_EnT < −5` labelling rule and the adiabatic threshold sweep that was
fitted to it, the `K_SV`-based label, the `k_diff = 5×10⁸` "calibrated" pre-factor, and
single-electron transfer as a classification target. Each is listed under
[What is deliberately not ported](#what-is-deliberately-not-ported).

## The theories

All energies kcal/mol, rate constants M⁻¹ s⁻¹, potentials V vs SCE.

### adiabatic — ΔG_EnT < 0

Inputs: `dG_EnT`. Free parameters: none.

Transfer happens if it is downhill. Nothing else. This is the baseline, and the point of
it is that it cannot be fitted, so whatever the barrier theories gain, they gain over
something that had no opportunity to be tuned.

The dataset optimum is not at zero — the original found roughly ΔG < −3 kcal/mol, worth
about +5 MCC, and put it down to a systematic bias in the DFT triplet energies. Sweeping
it is not done here. "Is transfer downhill" is a statement of thermodynamics; "is
transfer downhill by more than 3 kcal/mol" is a one-parameter model calibrated against
this screen. Both are defensible and only the first is a baseline. The consequence is
that the margin every other method shows over this rule is a margin over an intentionally
untuned rule, and anywhere that margin is quoted it has to say so.

### marcus — the crossing of two parabolas

Inputs: `dG_EnT`, `pc_lambda_S0_to_T1_dft`, `pc_lambda_T1_to_S0_dft`,
`q_lambda_S_T_dft`, `q_lambda_T_S_dft`. Free parameters: the barrier threshold, plus the
λ scale when it is not 1.

Asymmetric (the exact intersection of two parabolas of different curvature) and
symmetric (one effective λ, the geometric mean), with automatic fallback to the
symmetric form where the parabolas do not cross. The rate constant is the Balzani form,

```
k_EnT = k_diff / (1 + exp(ΔG/kT) + 0.25 exp(ΔG‡/kT))
```

with `k_diff = 2×10¹⁰ M⁻¹ s⁻¹`, the acetonitrile Smoluchowski limit. That is a solvent
property, not a theory parameter, so it is not swept.

Classification is a two-part gate: exergonic **and** barrier below a threshold. The
ΔG < 0 half is fixed at the adiabatic rule, because the question is whether the barrier
adds anything to thermodynamics and that only means something if thermodynamics is
inside it unchanged.

### rehm_weller — the empirical interpolation

Same inputs, same grid.

```
ΔG‡ = sqrt((ΔG/2)² + (λ_rxn/4)²) + ΔG/2
```

Not a parabola crossing: a hyperbolic interpolation that goes to λ_rxn/4 at ΔG = 0, to
ΔG when strongly endergonic, and to zero when strongly exergonic — it does not turn back
up in the inverted region the way Marcus does. That is why the two are worth running side
by side rather than treating one as a refinement of the other, and it is why its optimal
λ scale comes out lower: with the intrinsic barrier at λ_rxn/4, a smaller λ reproduces
the same effective barrier.

### sandros — the null model

Inputs: `dG_EnT`, `k_q`, `censored`. Free parameters: `k_max`.

```
k_EnT = k_max / (1 + exp(ΔG/kT))
```

No barrier at all: exergonic pairs go at the diffusion plateau, endergonic pairs are
slowed by the Boltzmann factor alone. If a Marcus barrier adds nothing over this, then
reorganization energy is not doing any work on this dataset.

`k_max` is fitted in log₁₀ space, which for a single multiplicative constant has a closed
form and needs no optimizer. **It is fitted on detected pairs only.** For the 614
censored pairs `k_q` is an upper bound, and least squares on a bound treats it as an
observation and drags `k_max` down towards the detection floor. The original fitted every
point.

There is only one classification setting, and the module explains why: as a classifier
Sandros is algebraically a driving-force cutoff, with `k_max` and the threshold perfectly
degenerate. Sweeping either would be sweeping the adiabatic rule's threshold under
another name. So it is scored at the ground truth's own cutoff instead — which, since
the label is K_SV > 10, is per-pair: `10/τ₀`, the `k_q_floor` column.

### dvte — a distribution instead of a number

Inputs: `pc_E_T_dft`, `q_dvte_mu_md`, `q_dvte_std_md`. Free parameters: none.

An adiabatic triplet energy is one number for one optimized geometry, and a molecule in
solution is not at that geometry. DvTE samples the vertical singlet–triplet gap along a
Milo trajectory and keeps the distribution. The classifier asks whether the quencher
spends any appreciable fraction of its time in a geometry whose triplet is accessible:

```
DvTE(p) = mu + Phi⁻¹(p) · sigma      quenches  ⟺  E_T(PC) ≥ DvTE(p)
```

for p ∈ {0.005, 0.008, 0.01, 0.015}. The original swept p and reported the best, which is
an in-sample choice its own audit found worth less than 0.02 MCC. All four are reported
here as separate methods, so the reader sees the spread rather than its maximum.

The `_kde` variant is not ported: it needs the sample distribution, which lives in
`data/computational_data_raw/quencher_adiabatic_DvTE/` rather than in `pairs.csv`, and everything
downstream in this repository reads `pairs.csv` and nothing else. It tracked the normal
form to within 0.07 kcal/mol. If it is wanted, the fix is to carry the four `_kde`
quantiles into `pairs.csv`.

## The λ pairing, which the port changed on purpose

`dataset/README.md` leaves the combination rule to this folder: "how the two partners'
terms add is a modelling decision — it belongs in `benchmark/`, where it can be stated and
varied, not frozen into the table." Stated, then.

The forward reaction is PC(T₁) + Q(S₀) → PC(S₀) + Q(T₁). For R → P the reorganization
energy is the product state at the reactant geometry minus the product state at its own
geometry, so per partner:

| partner | ends on | starting from | column |
|---|---|---|---|
| photocatalyst | the singlet surface | the T₁ geometry | `pc_lambda_T1_to_S0_dft` |
| quencher | the triplet surface | the S₀ geometry | `q_lambda_T_S_dft` |

```
λ_forward = pc_lambda_T1_to_S0 + q_lambda_T_S
λ_reverse = pc_lambda_S0_to_T1 + q_lambda_S_T
```

**This is the opposite pairing to the original**, which built `λ_forward = pc_lambda_T_S
+ q_lambda_S_T`. The two source tables read their subscripts in opposite orders: the
quencher's `lambda_S_T` is `E_singlet(T₁ geom) − E_singlet(S₀ geom)`, a difference on the
*singlet* surface, while the photocatalyst's `lambda_S0_to_T1` is a difference on the
*triplet* surface. Written out on one surface at a time they line up crosswise, and the
original's combination is the reverse reaction's reorganization energy. This is exactly
the trap `photocatalyst_dft.py` chose its geometry-explicit names to avoid, and the
original carried a whole section in its notes about call sites having to swap the
photocatalyst tuple by hand.

It matters in one place only. The symmetric Marcus form and all of Rehm–Weller use a
symmetric mean of the two, so the swap is invisible there — `marcus_barrier_sym` scores
identically either way, which is a useful check that the flag does what it claims.

**And the correction does not improve the score.** On the asymmetric form the original's
pairing scores higher than the physically correct one under both label regimes.
The physically correct assignment is the worse-scoring one. `--legacy-lambda-pairing`
restores the original so the difference can be measured rather than argued about, and
this should be settled on the physics rather than on the MCC.

## What the scoring means

### Two label regimes, both reported, neither chosen

The ground truth is set on K_SV, the observable, and not on k_q. That is not a
presentational choice: `k_q = K_SV / τ₀`, so a fixed k_q cutoff compares every pair
against a detection floor that moves with the photocatalyst. At the old 10⁷ M⁻¹ s⁻¹
cutoff the Stern–Volmer floor `25/τ₀` sat *above* the cutoff for 19 of the 26
photocatalysts, covering 912 of the 1248 pairs, and 465 pairs could not be labelled at
all. On K_SV the floor is one number for everything and every pair is labelled by the
same criterion, so the indeterminate class — and the old `resolvable` regime — is gone.

What is left is how permissive the criterion is, and both readings are scored:

| regime | positives | negatives | criterion |
|---|---:|---:|---|
| `inclusive` | 790 | 458 | K_SV > 10, or K_SV ≤ 10 with a slope significant across the points the fit used |
| `strict` | 634 | 614 | the published criterion, K_SV ≥ 25 and r² ≥ 0.90 |

Both cover all 1248 pairs; nothing is dropped. `strict` is a subset of `inclusive` by
construction, and `pairs.py` raises if that ever stops holding.

The 156 pairs between them are weak but real quenchers — monotone 3–4 % intensity drops
well above the read noise, at rate constants the published criterion never counted.
`dataset/README.md` sets out how they were established and why the rescue is a slope test
rather than an r² cut.

Note what the gap between the regimes is *not*: every method scores **higher** on `strict`
than on `inclusive`, by 0.01 to 0.14 MCC. That is not evidence that the strict label is
the right one. It says the 156 weak quenchers are the hard cases — which is what a
marginal quencher should be — and a criterion that scores well by leaving them out has
not predicted them. Which regime the paper reports is a scientific question about what
counts as quenching; this folder takes no position and scores both every time.

### Free parameters are counted, and tuned scores get a held-out counterpart

Every method's row carries `n_free`, the number of parameters chosen by maximizing MCC on
the data the MCC is then reported for. A barrier threshold is one; a barrier threshold
plus a λ scale is two; the adiabatic rule and DvTE have none.

For anything with a free parameter, `benchmark_best.csv` also carries
`MCC_leave_one_pc_out`: for each photocatalyst the setting is chosen on the other 25 and
applied to the held-out one, and the held-out predictions are pooled into a single
confusion matrix. Photocatalyst is the right grouping unit because every photocatalyst-side
quantity is constant across that photocatalyst's 48 pairs — a random split leaves the same
fingerprint on both sides. The original's audit measured that directly for its
machine-learning baselines: a random 5-fold random forest scored 0.805 and the same model
grouped by photocatalyst scored 0.640.

Pooling rather than averaging per-fold MCC matters. The original's leave-one-out study
averaged per fold and got 0.438 against an in-sample 0.617, then attributed the gap to the
variable positive fraction of small folds. Pooling removes that artifact, so what remains
is the real cost of choosing a threshold.

### Rate predictions are scored with the censoring respected

A single RMSE over all 1248 pairs would be wrong, and wrong in a flattering direction for
a theory that under-predicts: for the 614 censored pairs `k_q` is an upper bound and a
residual against a bound means nothing. So `results/rate_agreement.csv` reports two things
and does not combine them — RMSE and median bias in log₁₀ units on detected pairs, and on
censored pairs the fraction whose predicted rate falls at or below that pair's own
detection floor, which is the fraction of non-detections the theory is consistent with.

This is descriptive. It is not a censored-likelihood fit and it takes no position on
whether the benchmark should become a censored regression.

## Where it stands

Default run: all 26 photocatalysts, no mechanism filter, `k_diff = 2×10¹⁰`, geometric
mean λ. MCC.

| method | free | α | `inclusive` | held-out | `strict` |
|---|---:|---:|---:|---:|---:|
| adiabatic | 0 | — | 0.373 | 0.373 | 0.462 |
| dvte, p = 0.015 | 0 | — | 0.425 | 0.425 | 0.557 |
| dvte, p = 0.005 | 0 | — | 0.377 | 0.377 | 0.478 |
| sandros at the label cutoff | 1 | — | 0.370 | 0.370 | 0.384 |
| marcus barrier, symmetric | 1 | 1.00 | 0.427 | 0.427 | 0.547 |
| **marcus barrier, symmetric, λ-scaled** | 2 | 0.45 | **0.474** | 0.474 | **0.604** |
| marcus barrier, asymmetric, λ-scaled | 2 | 0.40 | 0.447 | 0.433 | 0.583 |
| marcus rate, λ-scaled | 2 | 0.40 | 0.459 | 0.445 | 0.590 |
| marcus rate at the label cutoff | 0 | 1.00 | 0.305 | 0.305 | 0.395 |
| rehm–weller barrier | 1 | 1.00 | 0.412 | 0.412 | 0.520 |
| rehm–weller barrier, λ-scaled | 2 | 0.30 | 0.442 | 0.442 | 0.574 |
| rehm–weller rate, λ-scaled | 2 | 0.25 | 0.448 | 0.448 | 0.581 |

Reading it:

- **λ scaling reproduces.** The optimum lands at α ≈ 0.40–0.45 for Marcus and α = 0.25–0.30
  for Rehm–Weller, which is what the original found (0.35–0.45 and 0.25–0.30) on a
  different version of the dataset. It is worth about +0.05 MCC over raw DFT λ, not the
  +0.15 to +0.20 the original reported against its own baseline.
- **It survives being held out.** Leave-one-photocatalyst-out costs the scaled methods
  0.00 to 0.01 MCC, and that selection includes α, not just the threshold. It is not an
  artifact of choosing 16 values and keeping the best.
- **The margin over the baseline is +0.10 MCC on `inclusive`, +0.14 on `strict`** — over a
  baseline that was deliberately not tuned, and with two free parameters against zero.
  The like-for-like zero-parameter comparison is the Marcus rate read at the label's own
  cutoff instead of at a fitted threshold, and it is *worse* than the adiabatic rule —
  0.305 against 0.373.
- **DvTE is not beaten by very much,** and it has no free parameter. On `inclusive` the
  best λ-scaled Marcus is 0.474 against DvTE's 0.425, on two fitted parameters against
  none; on `strict` it is 0.604 against 0.557.
- **Everything scored lower when the label moved to K_SV.** The old numbers were measured
  against a k_q cutoff that 19 of 26 photocatalysts could not resolve, which inflated
  specificity on pairs that carried no information. The ordering of the methods is
  unchanged; the absolute values are not comparable to the earlier table and should not
  be quoted against it.

The rate side is less flattering than the classification side, and worth stating plainly:

| | RMSE / dex | median bias / dex | consistent with the floor |
|---|---:|---:|---:|
| sandros, `k_max` = 1.17×10⁹ | 1.42 | +0.11 | 0.34 |
| marcus, α = 1.0 | 3.03 | −1.88 | 0.90 |
| marcus, α = 0.6 | 1.46 | +0.29 | 0.53 |
| marcus, α = 0.45 | 1.74 | +0.83 | 0.35 |
| rehm–weller, α = 1.0 | 3.39 | −2.47 | 0.91 |
| rehm–weller, α = 0.5 | 1.37 | +0.35 | 0.41 |

With raw DFT λ both theories under-predict the measured rate constants by roughly two
orders of magnitude. Scaling λ down fixes the bias, but the α that minimizes the residual
on detected pairs (≈ 0.5–0.6) is not the α that maximizes MCC (0.25–0.45), and the α that
best respects the non-detections is 1.0 — because a theory that under-predicts everything
is trivially consistent with every upper bound. No single λ scale satisfies all three at
once, and the one-parameter Sandros model beats both barrier theories on residual. That
is a symptom worth chasing before the rate numbers are used for anything, and it is not
something the λ-scaling argument accounts for.

None of the above is a conclusion about the chemistry. It is what the code produces today
on the table as it stands.

## Options

```
--theories adiabatic dvte sandros marcus rehm_weller
--mechanism-rules none strict relaxed dominant_ddG_5
--set-source exp calc
--bounds face|respect
--exclude-pc PC19 PC20
--k-diff 2e10
--mean-type geometric|arithmetic
--legacy-lambda-pairing
```

**Mechanism rules** admit a quenching pair as an energy-transfer positive only if the
competing single-electron-transfer channel can be ruled out: `strict` (ΔG_SET > +5),
`relaxed` (ΔG_SET > 0), `dominant_ddG_5` (ΔG_EnT < ΔG_SET − 5). Non-quenchers are never
dropped by any rule — they carry no mechanistic label, so they are legitimate negatives
either way. The default is `none`: every quenching pair is admitted, because deciding
which pairs are mechanistically clean is a scientific choice and not one this harness
should make silently. The original always filtered.

**Bounds.** 165 pairs have a `dG_SET_exp` that is a lower bound — the quencher was not
oxidized or reduced anywhere in the accessible window, so the true driving force can only
be *more* endergonic. All three rules ask whether electron transfer is sufficiently
uphill, so a lower bound always points towards "ruled out", and reading it at face value
understates how inaccessible the channel is — in the direction that wrongly drops pairs
out of the energy-transfer set. `--bounds face` reproduces the original; `--bounds
respect` admits those pairs. Neither is obviously right, so the default is `face` and
every run using a rule prints how many pairs the two policies disagree about (26 under
`strict/exp`, 22 under `relaxed/exp`).

**`--exclude-pc`** drops nothing by default. Two photocatalysts are flagged in `pairs.py`
and named in every run's header:

- **PC19** has imaginary frequencies in both the S₀ and the reduced optimization and no
  singlet-surface reorganization energy. Its 48 pairs fall out of every barrier and rate
  method automatically — the run reports this as 48 non-finite inputs — but they stay in
  the adiabatic and DvTE numbers, which need only E_T. So the barrier methods are already
  being scored on 1200 pairs while the baselines are scored on 1248, and every one of
  PC19's pairs is handed to the barrier methods as a predicted non-quencher. Excluding
  PC19 and PC20 moves the headline by about +0.03 MCC and changes no ordering.
- **PC20** has λ = 33.67 on the triplet surface against 12.70 on the singlet, a 2.7×
  asymmetry no other non-pmb complex shows, never verified. It is exactly the pair of
  numbers the asymmetric Marcus form is most sensitive to.

PC24 is not in this list. It was excluded upstream by name in `dataset/build_dataset.py`,
so `pairs.csv` is 26 photocatalysts and this folder never sees it.

## What is deliberately not ported

| | why |
|---|---|
| the `K_SV > 10` label | `K_SV = k_q·τ₀` biases the positive class by photocatalyst lifetime, which spans 86× here. `dataset/build_dataset.py` labels on `k_q` and the code path is gone. |
| the `ΔG_EnT < −5` feasibility rule, and the adiabatic threshold sweep fitted to it | the labels at the time were themselves generated by that rule, so matching the classifier to it was leakage. It was reported as "+17 MCC from one line of code" and it was not real. |
| `k_diff = 5×10⁸` as a second scenario | a pre-factor fitted to the data. The original's own audit put the `k_diff` sweep on its do-not-retest list: the rate methods absorb it into their own tuned threshold, so it buys nothing that is not already being counted. |
| single-electron transfer as a classification target | 9 to 55 unambiguously SET-only positives once the mechanism filters are applied — too few to carry a benchmark number. ΔG_SET is still used, as a filter. |
| the `_kde` DvTE variant | needs the sample distribution, which is not in `pairs.csv`. |
| machine-learning baselines | not this folder's job; `ml/` has the models. If one is ever added here it has to use `GroupKFold` by photocatalyst — a random split scored 0.805 against 0.640 grouped, which is memorization of the photocatalyst fingerprint. |
| parity and series plots | figures belong in `figures/`, which reads this folder's output. |

## Open

- **The label regime.** Not this folder's decision. Both are reported.
- **The bound policy on ΔG_SET.** Same. Both are available, the disagreement is counted.
- **The λ pairing.** The physically correct assignment scores worse than the original's.
  Someone should check the derivation in `marcus.py` against the raw job files before
  either number goes in a paper.
- **The rate residuals.** Three different α optima for three different questions, and a
  one-parameter Sandros fit that beats both barrier theories on residual. Unexplained.
- **Tuning and reporting on the same data.** Every `_scaled` row is a best-of-16-α, and
  the held-out column is the honest counterpart. If a single method is to be quoted in
  the paper, it should be quoted with its held-out number.
- **PC19's 48 pairs** are silently predicted "does not quench" by the barrier methods,
  which is a free pass on every true negative among them. The run warns; nothing corrects
  for it.
