# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Variants of the adiabatic criterion: a required margin, and a cutoff on the ML uncertainty.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/adiabatic_variants/margin.csv           E_T(PC) >= E_T(Q) + m for every m
        results/adiabatic_variants/ml_uncertainty.csv   the ML criterion with a cutoff on the
                                                        predicted uncertainty of E_T(Q)

    uv run benchmark/further_analysis/adiabatic_variants.py

Margin: m = 0 to 3 kcal/mol in steps of 0.1, with DFT and with ML quencher E_T.

Uncertainty cutoff u, taken at every distinct uncertainty of the 48 quenchers, in two
modes:
    excluded           pairs whose quencher uncertainty exceeds u are not scored
    no quenching       pairs whose quencher uncertainty exceeds u are predicted not to quench
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pairs  # noqa: E402
import scoring  # noqa: E402

RESULTS = HERE / "results" / "adiabatic_variants"
MARGINS = tuple(round(m, 1) for m in np.arange(0.0, 3.0001, 0.1))    # kcal/mol


def margin(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for source in pairs.SOURCES:
        dG = pairs.driving_force(df, source)
        for m in MARGINS:
            rows.append({"parameters": source, "margin / kcal mol-1": m,
                         **scoring.metrics(y, dG <= -m), "AUC": scoring.auc(y, -dG)})
    return pd.DataFrame(rows)


def ml_uncertainty(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    dG = pairs.driving_force(df, "ml")
    uncertainty = df[pairs.q_ml_uncertainty("E_T")].to_numpy(float)
    rows = []
    for u in np.unique(uncertainty):
        confident = uncertainty <= u
        rows.append({"mode": "excluded", "cutoff / kcal mol-1": u,
                     "fraction_scored": confident.mean(),
                     **scoring.metrics(y[confident], dG[confident] <= 0),
                     "AUC": scoring.auc(y[confident], -dG[confident])})
        rows.append({"mode": "no quenching", "cutoff / kcal mol-1": u,
                     "fraction_scored": 1.0,
                     **scoring.metrics(y, (dG <= 0) & confident),
                     "AUC": scoring.auc(y, np.where(confident, -dG, -np.inf))})
    return pd.DataFrame(rows)


def main():
    df = pairs.scored(pairs.load())
    margins, uncertainty = margin(df), ml_uncertainty(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    margins.to_csv(RESULTS / "margin.csv", index=False, lineterminator="\n")
    uncertainty.to_csv(RESULTS / "ml_uncertainty.csv", index=False, lineterminator="\n")

    for source, rows in margins.groupby("parameters"):
        best = rows.loc[rows["MCC"].idxmax()]
        print(f"margin {source}: MCC {rows.iloc[0]['MCC']:.3f} at 0, "
              f"best {best['MCC']:.3f} at {best['margin / kcal mol-1']} kcal/mol")
    for mode, rows in uncertainty.groupby("mode"):
        best = rows.loc[rows["MCC"].idxmax()]
        print(f"uncertainty, {mode}: best MCC {best['MCC']:.3f} at cutoff "
              f"{best['cutoff / kcal mol-1']:.2f} kcal/mol (scored {best['fraction_scored']:.2f})")
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
