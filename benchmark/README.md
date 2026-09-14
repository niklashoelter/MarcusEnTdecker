# benchmark

Physical-organic criteria for triplet energy transfer, scored against the Stern–Volmer
screen: can they tell which photocatalyst–quencher pairs quench, and how well do they
predict the quenching rate constant?

```
uv run reproduce.py benchmark        both scripts, in order
uv run benchmark/classification.py   quenches or not
uv run benchmark/rates.py            rate constants
```

The only input is `../data/exp_dataset_ent.csv`.

## Layout

```
benchmark/
├── pairs.py            loads the dataset and names its columns; which pairs are scored
├── theory.py           barrier and rate expressions (SI Section 4.1)
├── scoring.py          metrics, leave-one-photocatalyst-out, rate residuals
├── classification.py   the classification criteria
├── rates.py            predicted against measured rate constants, and the fits
└── results/
    ├── classification/
    │   ├── settings.csv    every criterion at every setting
    │   └── best.csv        best setting per criterion, with its held-out score
    └── rates/
        ├── agreement.csv   Marcus and Rehm–Weller rates against k_q
        └── fits.csv        Sandros and Rehm–Weller fits of k_q against ΔG_EnT
```

## The pairs

`exp_dataset_ent` is built in `../data/` by `build_exp_dataset_ent.py`.
- **Non-quenching pairs:** all kept.
- **Quenching pairs:** kept only if every single-electron-transfer channel with a
  measured quencher potential is more than 0.25 V uphill.
- **Columns:** the measurement, DFT properties of both partners, ML-predicted quencher
  properties, DvTE, and ΔG_EnT from both sources.

`pairs.py` scores every pair in it except:

- **PC22.** Its emission was too weak for the Stern–Volmer intensities to be trusted.
- **Missing input.** A property one of the criteria needs is blank. Every criterion is
  scored on the same pairs.

## Classification

The ground truth is `quenches`.

| criterion | predicts quenching when | tuned on the data |
|---|---|---|
| `adiabatic` | ΔG_EnT < 0 | nothing |
| `dvte` | E_T(PC) ≥ μ + Φ⁻¹(p)·σ, reported for p = 0.005, 0.008, 0.01, 0.015 | nothing |
| `marcus_barrier` | ΔG_EnT < 0 and ΔG‡ < threshold (0.5–15 kcal/mol) | threshold |
| `marcus_rate` | k·τ₀ > K_SV,crit = 10 M⁻¹ | nothing |
| `rehm_weller_rate` | k·τ₀ > K_SV,crit = 10 M⁻¹ | nothing |

- **Parameters.** `adiabatic` and the three kinetic criteria are scored twice, with the
  quencher's E_T and λ from DFT (`dft`) and from the ML model (`ml`). Photocatalyst
  values are always DFT.
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
