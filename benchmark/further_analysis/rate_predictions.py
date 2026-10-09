# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Predicted quenching rate constants for every scored pair, source, rate law and barrier
model.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/rate_predictions/pairs.csv              the scored pairs with their
                                                        measurement
        results/rate_predictions/predictions.csv.gz     one row per pair, source, rate law
                                                        and barrier: log10 of the
                                                        predicted k

    uv run benchmark/further_analysis/rate_predictions.py

Rate law     barrier
full         Marcus symmetric, Marcus asymmetric, Rehm-Weller
irreversible Marcus symmetric, Marcus asymmetric, Rehm-Weller
sandros      none

The full rate law takes k_-d/nu = (k_d/K_A)/(k_B T/h), with K_A the encounter constant at
the contact distance r_PC + r_Q of each pair (pairs.CONTACT), as in the two_gate criterion.

Sources: DFT and ML quencher parameters; photocatalyst parameters are DFT. Nothing is
fitted. For pairs that do not quench (`censored`), the measured k_q is an upper bound, not a
rate.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pairs  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "rate_predictions"
PAIR_COLUMNS = [pairs.PC_NAME, pairs.Q_NAME, pairs.K_SV, pairs.TAU0, pairs.K_Q, pairs.QUENCHES]


def predictions(df):
    parts = []
    index = np.arange(len(df))
    escape_over_nu = theory.escape_over_frequency(pairs.contact_distances(df))
    for source in pairs.SOURCES:
        dG = pairs.driving_force(df, source)
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        parts.append(pd.DataFrame({"pair": index, "parameters": source, "rate_law": "sandros",
                                   "barrier": "none",
                                   "log10 k / M-1 s-1": np.log10(theory.sandros_rate(dG))}))
        for model in theory.BARRIERS:
            barrier = theory.barrier(model, dG, forward, reverse)
            for law, k in (("full", theory.rate(dG, barrier, escape_over_nu)),
                           ("irreversible", theory.rate_irreversible(barrier))):
                parts.append(pd.DataFrame({"pair": index, "parameters": source,
                                           "rate_law": law, "barrier": model,
                                           "log10 k / M-1 s-1": np.log10(k)}))
    return pd.concat(parts, ignore_index=True)


def main():
    df = pairs.scored(pairs.load())
    table = predictions(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df[PAIR_COLUMNS].rename_axis("pair").to_csv(RESULTS / "pairs.csv", lineterminator="\n")
    # mtime 0 keeps the gzip header, and so the file, the same from run to run.
    table.to_csv(RESULTS / "predictions.csv.gz", index=False, lineterminator="\n",
                 float_format="%.5g", compression={"method": "gzip", "mtime": 0})
    print(f"{len(df)} pairs, {len(table) // len(df)} predictions per pair "
          f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
