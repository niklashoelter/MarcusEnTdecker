# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Compares predicted and measured quenching rate constants.

Reads   ../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/rates/agreement.csv   Marcus and Rehm-Weller rates against k_q, per
                                      lambda source and scale factor alpha
        results/rates/fits.csv        Sandros and Rehm-Weller fitted to k_q against
                                      dG_EnT, over all pairs, per photocatalyst and per
                                      quencher

    uv run benchmark/rates.py

Only scored pairs that quench are used: for the others k_q is an upper bound, not a
measurement. Residuals are in decades of k (dex).

The fits have one free parameter each:
    Sandros        k = k_max / (1 + exp(dG/RT))            k_max
    Rehm-Weller    k = Balzani rate with the RW barrier     one effective lambda
Both use the DFT driving force.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import pairs
import scoring
import theory
from classification import ALPHAS

RESULTS = pairs.RESULTS / "rates"

MIN_POINTS = 3                  # fewer detected pairs than this: no fit
LAMBDA_RANGE = (0.1, 400.0)     # kcal/mol, search range for the Rehm-Weller fit


def agreement(df):
    k_measured = df[pairs.K_Q].to_numpy(float)
    rows = []
    for source in pairs.SOURCES:
        dG = pairs.driving_force(df, source)
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        for alpha in ALPHAS:
            lf, lr = alpha * forward, alpha * reverse
            predicted = {"marcus": theory.marcus_rate(dG, lf, lr),
                         "rehm_weller": theory.rehm_weller_rate(
                             dG, theory.effective_lambda(lf, lr))}
            for model, k in predicted.items():
                rows.append({"model": model, "parameters": source, "alpha": alpha,
                             **scoring.log_residuals(k, k_measured)})
    return pd.DataFrame(rows)


def fit_sandros(dG, k):
    """k_max by least squares in log10 k; closed form for one multiplicative constant."""
    boltzmann = np.log10(1.0 + np.exp(np.clip(dG / theory.RT, -theory.EXP_LIMIT, theory.EXP_LIMIT)))
    return 10.0 ** np.mean(np.log10(k) + boltzmann)


def fit_rehm_weller(dG, k):
    """The effective lambda by least squares in log10 k."""
    def loss(lam):
        return np.sum((np.log10(theory.rehm_weller_rate(dG, lam)) - np.log10(k)) ** 2)

    grid = np.linspace(*LAMBDA_RANGE, 800)
    start = int(np.argmin([loss(lam) for lam in grid]))
    bounds = (grid[max(start - 1, 0)], grid[min(start + 1, len(grid) - 1)])
    return float(minimize_scalar(loss, bounds=bounds, method="bounded").x)


def fits(df):
    series = [("all", "", df)]
    series += [("photocatalyst", name, group) for name, group in df.groupby(pairs.PC_NAME, sort=False)]
    series += [("quencher", name, group) for name, group in df.groupby(pairs.Q_NAME, sort=False)]

    rows = []
    for kind, name, group in series:
        dG = pairs.driving_force(group, "dft")
        k = group[pairs.K_Q].to_numpy(float)
        row = {"series": kind, "name": name, "n": len(group)}
        if len(group) >= MIN_POINTS:
            k_max = fit_sandros(dG, k)
            lam = fit_rehm_weller(dG, k)
            row |= {
                "sandros_k_max / M-1 s-1": k_max,
                "sandros_rmse / dex": scoring.log_residuals(theory.sandros_rate(dG, k_max), k)["rmse / dex"],
                "rehm_weller_lambda / kcal mol-1": lam,
                "rehm_weller_rmse / dex": scoring.log_residuals(theory.rehm_weller_rate(dG, lam), k)["rmse / dex"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    table = pairs.load()
    print(pairs.summary(table))
    detected = pairs.scored(table)
    detected = detected[detected[pairs.QUENCHES] == 1].reset_index(drop=True)
    print(f"{len(detected)} scored pairs that quench are used")

    agreement_table = agreement(detected)
    fit_table = fits(detected)

    RESULTS.mkdir(parents=True, exist_ok=True)
    agreement_table.to_csv(RESULTS / "agreement.csv", index=False, lineterminator="\n")
    fit_table.to_csv(RESULTS / "fits.csv", index=False, lineterminator="\n")

    print("\nLowest RMSE per model and lambda source:")
    best = agreement_table.loc[agreement_table.groupby(["model", "parameters"])["rmse / dex"].idxmin()]
    print(best.round(3).to_string(index=False))
    print("\nFit over all pairs:")
    print(fit_table[fit_table["series"] == "all"].round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
