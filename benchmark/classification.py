# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "openpyxl>=3.1", "pandas"]
# ///
"""
Scores the criteria for predicting whether a pair quenches.

Reads   the pair table (pairs.py)
Writes  results/classification/settings.csv   every criterion at every setting
        results/classification/best.csv       the best setting of each criterion, with
                                              its leave-one-photocatalyst-out score

    uv run benchmark/classification.py

Criterion            predicts quenching when
adiabatic            dG_EnT < 0
dvte                 E_T(PC) >= DvTE(p), for p = 0.005, 0.008, 0.01, 0.015
marcus_barrier       dG_EnT < 0 and Marcus barrier < threshold
marcus_rate          k_Marcus * tau0 > K_SV,crit
rehm_weller_rate     k_Rehm-Weller * tau0 > K_SV,crit

Adiabatic and the three kinetic criteria are scored with the quencher's E_T and lambda
from DFT and from the ML model. The kinetic criteria are scored with DFT lambda as is
(`lambda raw`) and with lambda scaled by a factor alpha (`lambda scaled`).

The ground truth is `quenches` from exp_dataset.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import pairs
import scoring
import theory

RESULTS = pairs.RESULTS / "classification"

ALPHAS = tuple(round(a, 2) for a in np.arange(0.25, 1.001, 0.05))
BARRIER_THRESHOLDS = tuple(round(t, 1) for t in np.arange(0.5, 15.01, 0.5))  # kcal/mol
K_SV_CRIT = 10.0                                                            # M-1
DVTE_P = (0.005, 0.008, 0.01, 0.015)

CRITERIA = ["adiabatic", "dvte", "marcus_barrier", "marcus_rate", "rehm_weller_rate"]

# The columns that identify one criterion; settings sharing them compete for "best".
CRITERION = ["criterion", "parameters", "variant"]


@dataclass
class Setting:
    criterion: str
    parameters: str         # "dft" or "ml"
    variant: str            # "", "lambda raw", "lambda scaled", or the DvTE p
    n_free: int             # parameters chosen by maximizing MCC
    predicted: np.ndarray
    alpha: float = np.nan
    barrier_threshold: float = np.nan


def settings(df):
    """Every setting of every criterion, as predictions for the rows of df."""
    tau0 = pairs.tau0_seconds(df)
    pc_e_t = df[pairs.PC_E_T].to_numpy(float)

    for source in pairs.SOURCES:
        yield Setting("adiabatic", source, "", 0, pairs.driving_force(df, source) < 0)

    for p in DVTE_P:
        threshold = theory.dvte(df[pairs.Q_DVTE_MU], df[pairs.Q_DVTE_STD], p)
        yield Setting("dvte", "dft", f"p = {p}", 0, pc_e_t >= threshold)

    for source in pairs.SOURCES:
        dG = pairs.driving_force(df, source)
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        for alpha in sorted(ALPHAS, key=lambda a: a != 1.0):     # raw lambda first
            scaled = alpha != 1.0
            variant = "lambda scaled" if scaled else "lambda raw"
            lf, lr = alpha * forward, alpha * reverse

            barrier = theory.marcus_barrier(dG, lf, lr)
            for threshold in BARRIER_THRESHOLDS:
                yield Setting("marcus_barrier", source, variant, 1 + scaled,
                              (dG < 0) & (barrier < threshold), alpha, threshold)

            rates = {"marcus_rate": theory.marcus_rate(dG, lf, lr),
                     "rehm_weller_rate": theory.rehm_weller_rate(
                         dG, theory.effective_lambda(lf, lr))}
            for name, k in rates.items():
                yield Setting(name, source, variant, int(scaled), k * tau0 > K_SV_CRIT, alpha)


def score(df):
    """(every setting, best setting per criterion) as DataFrames."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    groups = df[pairs.PC_NAME].to_numpy()
    every = list(settings(df))

    rows = [{**_describe(s), **scoring.metrics(y, s.predicted)} for s in every]
    all_settings = pd.DataFrame(rows)

    best_rows = []
    for _, members in all_settings.groupby(CRITERION, sort=False):
        matrix = np.vstack([every[i].predicted for i in members.index])
        pick = members.index[scoring.best_setting(y, matrix)]
        held_out, n_chosen = scoring.leave_one_out(y, matrix, groups)
        held = scoring.metrics(y, held_out)
        best_rows.append({**all_settings.loc[pick].to_dict(),
                          "n_settings": len(members),
                          "MCC_held_out": held["MCC"],
                          "accuracy_held_out": held["accuracy"],
                          "settings_chosen_held_out": n_chosen})
    best = pd.DataFrame(best_rows)
    best = best.sort_values(["parameters", "criterion", "variant"],
                            key=lambda c: c.map(CRITERIA.index) if c.name == "criterion" else c)
    return all_settings, best.reset_index(drop=True)


def _describe(s: Setting) -> dict:
    return {"criterion": s.criterion, "parameters": s.parameters, "variant": s.variant,
            "n_free": s.n_free, "alpha": s.alpha,
            "barrier_threshold / kcal mol-1": s.barrier_threshold,
            "K_SV_crit / M-1": K_SV_CRIT if s.criterion.endswith("_rate") else np.nan}


def main():
    table = pairs.load()
    print(pairs.summary(table))
    all_settings, best = score(pairs.scored(table))

    RESULTS.mkdir(parents=True, exist_ok=True)
    all_settings.to_csv(RESULTS / "settings.csv", index=False)
    best.to_csv(RESULTS / "best.csv", index=False)

    shown = CRITERION + ["n_free", "alpha", "barrier_threshold / kcal mol-1",
                         "accuracy", "recall", "specificity", "MCC", "MCC_held_out"]
    print()
    print(best[shown].rename(columns={"barrier_threshold / kcal mol-1": "barrier"})
          .round(3).to_string(index=False))
    print(f"\n-> {len(all_settings)} settings, {len(best)} criteria in "
          f"{RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
