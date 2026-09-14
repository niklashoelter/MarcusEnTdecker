# benchmark

Physical-organic criteria for triplet energy transfer, scored against the Stern–Volmer
screen: can they tell which photocatalyst–quencher pairs quench, and how well do they
predict the quenching rate constant?

```
uv run reproduce.py benchmark        all three scripts, in order
uv run benchmark/pairs.py            the pair table
uv run benchmark/classification.py   quenches or not
uv run benchmark/rates.py            rate constants
```

Inputs are read from `../data/` and nothing there is changed.

## Layout

```
benchmark/
├── pairs.py            the pair table: measurement, calculated and predicted
│                       properties, driving forces, and which pairs are scored
├── theory.py           barrier and rate expressions (SI Section 4.1)
├── scoring.py          metrics, leave-one-photocatalyst-out, rate residuals
├── classification.py   the classification criteria
├── rates.py            predicted against measured rate constants, and the fits
└── results/
    ├── pairs.csv
    ├── classification/
    │   ├── settings.csv    every criterion at every setting
    │   └── best.csv        best setting per criterion, with its held-out score
    └── rates/
        ├── agreement.csv   Marcus and Rehm–Weller rates against k_q
        └── fits.csv        Sandros and Rehm–Weller fits of k_q against ΔG_EnT
```

## The pairs

`pairs.py` starts from `exp_dataset.csv` (1296 pairs) and joins on:

| from | columns |
|---|---|
| `photocatalyst_properties_calc.xlsx` | E_T, λ_S0_T1, λ_T1_S0, excited-state potentials |
| `quencher_properties_calc.xlsx` | E_T, λ_S0_T1, λ_T1_S0 (`_dft`), DvTE mean and standard deviation |
| `quencher_properties_ml.xlsx` | E_T, λ_S0_T1, λ_T1_S0 (`_ml`) |
| `quencher_properties_exp.xlsx` | measured oxidation and reduction potentials |

The ML model covers organic molecules only, so photocatalyst values are always DFT.
ΔG_EnT = E_T(Q) − E_T(PC) is derived for both quencher sources.

A pair is scored unless its `excluded` column gives the reason:

- **PC22.** Its emission was too weak for the Stern–Volmer intensities to be trusted.
- **SET not ruled out.** Only for pairs that quench. Electron transfer is ruled out when
  every channel with a measured quencher potential is more than **0.217 V** uphill:
  ΔE_ox = E_ox(Q) − E(PC\*/PC⁻) and ΔE_red = E(PC⁺/PC\*) − E_red(Q). A measured limit
  (E_ox > x, E_red < x) is read as its value, which rules the channel out only if the
  limit itself is past 0.217 V.
- **Missing input.** A property one of the criteria needs is blank. Every criterion is
  scored on the same pairs.

Pairs that do not quench are never dropped by the SET rule.

## Classification

The ground truth is `quenches` from `exp_dataset`.

| criterion | predicts quenching when | tuned on the data |
|---|---|---|
| `adiabatic` | ΔG_EnT < 0 | nothing |
| `dvte` | E_T(PC) ≥ μ + Φ⁻¹(p)·σ, reported for p = 0.005, 0.008, 0.01, 0.015 | nothing |
| `marcus_barrier` | ΔG_EnT < 0 and ΔG‡ < threshold (0.5–15 kcal/mol) | threshold |
| `marcus_rate` | k·τ₀ > K_SV,crit = 10 M⁻¹ | nothing |
| `rehm_weller_rate` | k·τ₀ > K_SV,crit = 10 M⁻¹ | nothing |

- **Parameters.** `adiabatic` and the three kinetic criteria are scored twice, with the
  quencher's E_T and λ from DFT (`dft`) and from the ML model (`ml`).
- **λ scaling.** The kinetic criteria are scored with λ as calculated (`lambda raw`) and
  with λ multiplied by α = 0.25–0.95 (`lambda scaled`), where α is chosen on the data.
- **n_free** counts the parameters chosen by maximizing MCC.
- **Held-out score.** For each photocatalyst, the setting is chosen on the other
  photocatalysts and applied to the held-out one. The held-out predictions are pooled
  into one confusion matrix (`MCC_held_out`). For a criterion without free parameters it
  equals the in-sample score.

Metrics: accuracy, precision, recall (sensitivity), specificity, F1 and MCC, with the
confusion counts.

## Rates

Rate constants are compared only for scored pairs that quench. For the others, k_q is
an upper bound.

- **`agreement.csv`**: Marcus and Rehm–Weller rates, each with DFT and ML λ, at every α.
  Gives RMSE and median bias of log₁₀(k_predicted / k_q).
- **`fits.csv`**: one-parameter fits of log₁₀ k_q against the DFT driving force, over all
  pairs, per photocatalyst and per quencher (at least three pairs).
  - Sandros, k = k_max / (1 + exp(ΔG/RT)), fits k_max.
  - Rehm–Weller, the Balzani rate with the Rehm–Weller barrier, fits one effective λ.

## Physics

Energies in kcal/mol, T = 298.15 K, k_diff = 2 × 10¹⁰ M⁻¹ s⁻¹ (acetonitrile).

- **Pair reorganization energy.** Each partner contributes the surface it ends on, at the
  geometry it starts from:
  λ_forward = λ_S0_T1(PC) + λ_T1_S0(Q) and λ_reverse = λ_T1_S0(PC) + λ_S0_T1(Q).
- **Marcus barrier.** The crossing of the two parabolas. Where they do not cross, the
  symmetric form (ΔG + λ)² / 4λ with λ = √(λ_forward·λ_reverse).
- **Rehm–Weller barrier.** √((ΔG/2)² + (λ/4)²) + ΔG/2, with the same λ.
- **Rate.** k = k_diff / (1 + exp(ΔG/RT) + 0.25·exp(ΔG‡/RT)).

Reorganization energies are named `lambda_<surface>_<geometry>`, as in `../data/`.
