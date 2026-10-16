# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
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
        results/rate_predictions/parity.csv             full rate law, asymmetric barrier:
                                                        predicted against measured log10 k_q
                                                        on the quenching pairs, and a
                                                        constant predictor

    uv run benchmark/further_analysis/rate_predictions.py

Rate law          barrier
full              Marcus symmetric, Marcus asymmetric
full (r = 7 A)    Marcus symmetric, Marcus asymmetric
irreversible      Marcus symmetric, Marcus asymmetric
sandros           none

Full rate law: k_-d/nu = (k_d/K_A)/(k_B T/h), K_A = encounter constant at r = r_PC + r_Q
(pairs.CONTACT) or r = 7 A (theory.CONTACT_DISTANCE).

DFT and ML quencher parameters; photocatalyst parameters DFT; nothing fitted. Non-quenching
pairs (`censored`): measured k_q is an upper bound.

Parity (quenching pairs, k_q > 0): median and RMSE of predicted - measured log10 k_q, MAE,
Spearman rho, fraction within one decade, SD (ddof = 1) of predicted and measured.
Constant predictor: mean measured log10 k_q.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import pairs  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "rate_predictions"
PAIR_COLUMNS = [pairs.PC_NAME, pairs.Q_NAME, pairs.K_SV, pairs.TAU0, pairs.K_Q, pairs.QUENCHES]
FULL_LAWS = {"full": None, "full (r = 7 A)": theory.CONTACT_DISTANCE}   # None: r_PC + r_Q
PARITY_BARRIER = "marcus_asymmetric"


def escape_over_nu(df, distance=None):
    """k_-d/nu of every pair at its contact distance, or at `distance` in Angstrom."""
    r = pairs.contact_distances(df) if distance is None else np.full(len(df), float(distance))
    return theory.escape_over_frequency(r)


def log_k_full(df, source, model=PARITY_BARRIER, distance=None):
    """log10 k of the full rate law for every pair."""
    dG = pairs.driving_force(df, source)
    barrier = theory.barrier(model, dG, *theory.pair_lambdas(*pairs.lambdas(df, source)))
    return np.log10(theory.rate(dG, barrier, escape_over_nu(df, distance)))


def predictions(df):
    parts = []
    index = np.arange(len(df))
    escape = {law: escape_over_nu(df, distance) for law, distance in FULL_LAWS.items()}
    for source in pairs.SOURCES:
        dG = pairs.driving_force(df, source)
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        parts.append(pd.DataFrame({"pair": index, "parameters": source, "rate_law": "sandros",
                                   "barrier": "none",
                                   "log10 k / M-1 s-1": np.log10(theory.sandros_rate(dG))}))
        for model in theory.BARRIERS:
            barrier = theory.barrier(model, dG, forward, reverse)
            laws = {law: theory.rate(dG, barrier, escape[law]) for law in FULL_LAWS}
            laws["irreversible"] = theory.rate_irreversible(barrier)
            for law, k in laws.items():
                parts.append(pd.DataFrame({"pair": index, "parameters": source,
                                           "rate_law": law, "barrier": model,
                                           "log10 k / M-1 s-1": np.log10(k)}))
    return pd.concat(parts, ignore_index=True)


def parity_row(predicted, measured):
    d = predicted - measured
    rho = spearmanr(predicted, measured).statistic if np.ptp(predicted) > 0 else np.nan
    return {"n": len(d), "median bias / decades": np.median(d),
            "RMSE / decades": np.sqrt(np.mean(d ** 2)), "MAE / decades": np.mean(np.abs(d)),
            "Spearman rho": rho, "within 1 decade": np.mean(np.abs(d) <= 1.0),
            "SD predicted / decades": np.std(predicted, ddof=1),
            "SD measured / decades": np.std(measured, ddof=1)}


def parity(df):
    """Parity of the full rate law with the asymmetric barrier, and of a constant."""
    k_q = df[pairs.K_Q].to_numpy(float)
    on = df[pairs.QUENCHES].to_numpy(bool) & (k_q > 0)
    measured = np.log10(k_q[on])
    rows = []
    for law, distance in FULL_LAWS.items():
        for source in pairs.SOURCES:
            predicted = log_k_full(df, source, distance=distance)[on]
            rows.append({"predictor": f"{law}, {PARITY_BARRIER}", "parameters": source,
                         **parity_row(predicted, measured)})
    rows.append({"predictor": "constant (mean measured)", "parameters": "",
                 **parity_row(np.full(len(measured), measured.mean()), measured)})
    return pd.DataFrame(rows)


def main():
    df = pairs.scored(pairs.load())
    table = predictions(df)
    agreement = parity(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df[PAIR_COLUMNS].rename_axis("pair").to_csv(RESULTS / "pairs.csv", lineterminator="\n")
    # mtime 0: byte-identical gzip output
    table.to_csv(RESULTS / "predictions.csv.gz", index=False, lineterminator="\n",
                 float_format="%.5g", compression={"method": "gzip", "mtime": 0})
    agreement.round(3).to_csv(RESULTS / "parity.csv", index=False, lineterminator="\n")
    print(agreement.round(3).to_string(index=False))
    print(f"{len(df)} pairs, {len(table) // len(df)} predictions per pair "
          f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
