# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Scores the `adiabatic` criterion with the quencher E_T of EnTdecker 1.0.

Reads   results/entdecker_1/quencher_E_T.csv     E_T of the triplet-energy model
        ../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/entdecker_1/adiabatic.csv        one row in the columns of
                                                 results/classification/best.csv,
                                                 parameters = entdecker

    uv run benchmark/entdecker_1.py

The model run is not part of the repository (see README.md).
dG_EnT = E_T(Q) - E_T(PC, DFT), rounded to 2 decimals, scored on pairs.scored().
"""

import sys

import numpy as np
import pandas as pd

import classification
import pairs
import scoring

RESULTS = pairs.RESULTS / "entdecker_1"
PREDICTIONS = RESULTS / "quencher_E_T.csv"
SOURCE = "entdecker"
DECIMALS = 2


def main():
    table = pairs.load()
    df = pairs.scored(table)
    e_t = pd.read_csv(PREDICTIONS).set_index(pairs.Q_ID)["E_T / kcal mol-1"]

    q_e_t = df[pairs.Q_ID].map(e_t).to_numpy(float)
    if np.isnan(q_e_t).any():
        missing = sorted(df.loc[np.isnan(q_e_t), pairs.Q_NAME].unique())
        sys.exit(f"scored pairs without an EnTdecker 1.0 E_T: {', '.join(missing)}")
    df = df.copy()
    df[pairs.dg_ent(SOURCE)] = np.round(q_e_t - df[pairs.PC_E_T].to_numpy(float), DECIMALS)

    rows = classification.evaluate(df, "adiabatic", "", SOURCE,
                                   [classification.Setting("adiabatic")], 0)
    row = scoring.rounded(pd.DataFrame([{**r, "n_settings": 1} for r in rows]))
    row.to_csv(RESULTS / "adiabatic.csv", index=False, lineterminator="\n")

    print(pairs.summary(table))
    print()
    print(row[["n", "accuracy", "MCC", "AUC", "AUC_within_PC_median"]].to_string(index=False))
    print(f"\n-> {(RESULTS / 'adiabatic.csv').relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
