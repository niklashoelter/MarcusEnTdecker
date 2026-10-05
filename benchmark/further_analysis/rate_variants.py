# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Rate criteria with every barrier model and both rate laws, and the MCC over alpha and
K_SV,crit together.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/rate_variants/best.csv          k * tau0 >= 10 M-1 for every barrier model
                                                and rate law, lambda raw and scaled
        results/rate_variants/alpha_k_sv_crit.csv   MCC, recall and specificity on DFT
                                                parameters for every alpha and K_SV,crit
                                                (full rate law)
        results/rate_variants/alpha_k_sv_crit_best.csv   the best (alpha, K_SV,crit) per
                                                barrier model, and the best K_SV,crit per alpha

    uv run benchmark/further_analysis/rate_variants.py

Rate laws: theory.rate (full) and theory.rate_irreversible. Barrier models:
theory.BARRIERS. K_SV,crit grid: 1 to 1000 M-1 in steps of 0.05 in log10.
"""

import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "rate_variants"
RATE_LAWS = ("full", "irreversible")
K_SV_CRIT_GRID = tuple(float(f"{10 ** e:.4g}") for e in np.arange(0.0, 3.0001, 0.05))  # M-1


def variants(model, law):
    base = classification.Setting("rate", barrier_model=model, rate_law=law,
                                  k_sv_crit=classification.K_SV_CRIT)
    return {"lambda raw": [base],
            "lambda scaled": [replace(base, alpha=a) for a in classification.ALPHAS]}


def best_table(df):
    best = []
    for model in theory.BARRIERS:
        for law in RATE_LAWS:
            for variant, options in variants(model, law).items():
                _, rows, _ = classification.tune(df, f"{model} rate, {law}", variant, options,
                                                 every_setting=False)
                best += rows
    return pd.DataFrame(best)


def alpha_k_map(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for model in theory.BARRIERS:
        for alpha in classification.ALPHAS:
            setting = classification.Setting("rate", barrier_model=model, alpha=alpha,
                                             k_sv_crit=1.0)
            _, log_k_tau0 = classification.predict(df, setting, "dft")
            for k_sv_crit in K_SV_CRIT_GRID:
                m = scoring.metrics(y, log_k_tau0 >= np.log10(k_sv_crit))
                rows.append({"barrier": model, "alpha": alpha, "K_SV_crit / M-1": k_sv_crit,
                             "recall": m["recall"], "specificity": m["specificity"], "MCC": m["MCC"]})
    return pd.DataFrame(rows)


def map_best(grid):
    rows = []
    for model, part in grid.groupby("barrier", sort=False):
        best = part.iloc[scoring.best_setting_from_scores(part["MCC"].to_numpy())]
        rows.append({"barrier": model, "best_of": "alpha and K_SV_crit", **best.drop("barrier")})
        for alpha, by_alpha in part.groupby("alpha"):
            best = by_alpha.iloc[scoring.best_setting_from_scores(by_alpha["MCC"].to_numpy())]
            rows.append({"barrier": model, "best_of": "K_SV_crit at this alpha", **best.drop("barrier")})
    return pd.DataFrame(rows)


def main():
    df = pairs.scored(pairs.load())
    best, grid = best_table(df), alpha_k_map(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(best).to_csv(RESULTS / "best.csv", index=False, lineterminator="\n")
    scoring.rounded(grid).to_csv(RESULTS / "alpha_k_sv_crit.csv", index=False,
                                 lineterminator="\n")
    grid_best = map_best(grid)
    scoring.rounded(grid_best).to_csv(RESULTS / "alpha_k_sv_crit_best.csv", index=False,
                                      lineterminator="\n")

    shown = ["criterion", "parameters", "variant", "alpha", "recall", "specificity", "MCC", "AUC"]
    print(best[shown].round(3).to_string(index=False))
    print()
    print(grid_best[grid_best["best_of"] == "alpha and K_SV_crit"].round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
