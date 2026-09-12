# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "openpyxl>=3.1", "pandas"]
# ///
"""
Sandros: a barrierless rate, thermodynamics and a diffusion plateau and nothing else.

Reads   ../data/exp_dataset.csv (through pairs.py)
Writes  nothing -- run_benchmark.py collects the predictions

    uv run benchmark/sandros.py       score this one theory on its own

WHAT IT COMPUTES

        k_EnT = k_max / (1 + exp(dG / kT))

    The classical description of triplet-triplet transfer as a pre-equilibrium with no
    activation barrier at all: exergonic pairs go at the diffusion plateau k_max,
    endergonic pairs are slowed by the Boltzmann factor alone. Sandros, Acta Chem.
    Scand. 1964, 18, 2355. It is the null model the barrier theories have to beat -- if
    a Marcus barrier adds nothing over this, then reorganization energy is not doing any
    work on this dataset.

    k_max is the one free parameter. It is fitted here in log10 space by least squares,
    which for a single multiplicative constant has a closed form,

        log10 k_max = mean( log10 k_q + log10(1 + exp(dG/kT)) )

    so no optimizer and no scipy is involved, and the answer is exact rather than
    dependent on a starting guess. The original implementation fitted the same parameter
    with `scipy.optimize.curve_fit` per photocatalyst and per quencher, for its series
    plots.

    THE FIT USES DETECTED PAIRS ONLY. Half the screen is left-censored: for those pairs
    the recorded k_q is an upper bound, not a measurement. Least squares on a bound
    treats it as an observation and drags k_max down towards the detection floor. Rows
    with `censored = 1` are therefore excluded from the fit, and the count that went into
    it is reported. This is a departure from the original, which fitted every point.

WHY THERE IS ONLY ONE CLASSIFICATION SETTING HERE
    As a classifier, Sandros is algebraically a driving-force cutoff and nothing more.
    Predicting `log10 k >= t` means

        log10 k_max - log10(1 + exp(dG/kT)) >= t   <=>   dG <= some function of (t, k_max)

    -- k_max and the threshold are perfectly degenerate, and what survives is a monotone
    rule on dG alone. Sweeping either one would just be sweeping the adiabatic rule's
    threshold under a different name, which is the one thing this project decided not to
    do to its baseline (see adiabatic.py).

    So Sandros is scored at the ground truth's own cutoff instead: it predicts quenching
    where its predicted rate constant exceeds the 1e7 M-1 s-1 that defines the label. No
    threshold is fitted; k_max is, which is why n_free is 1 and not 0.
"""

import numpy as np
import pandas as pd

import pairs
import theory
from theory import Config

NAME = "sandros"
DESCRIPTION = "k_max / (1 + exp(dG/kT)), the barrierless pre-equilibrium null model"


def required_columns():
    return [pairs.DG_ENT, pairs.K_Q, pairs.CENSORED]


def sandros_rate(dG, k_max, temperature=theory.TEMPERATURE):
    """k_EnT in M-1 s-1."""
    dG = np.asarray(dG, dtype=float)
    kbT = theory.R_KCAL * temperature
    return k_max / (1.0 + np.exp(np.clip(dG / kbT, -500.0, 500.0)))


def fit_k_max(dG, k_q, usable, temperature=theory.TEMPERATURE):
    """Least squares in log10 space for the single multiplicative constant.

    `usable` selects the rows the fit may see -- detected pairs with a positive rate
    constant. Returns (k_max, n_points).
    """
    dG = np.asarray(dG, dtype=float)
    k_q = np.asarray(k_q, dtype=float)
    kbT = theory.R_KCAL * temperature
    ok = usable & np.isfinite(dG) & np.isfinite(k_q) & (k_q > 0)
    if ok.sum() < 2:
        raise ValueError(
            "Fewer than two detected pairs with a positive k_q and a finite dG; "
            "Sandros has nothing to fit k_max on."
        )
    boltzmann = np.log10(1.0 + np.exp(np.clip(dG[ok] / kbT, -500.0, 500.0)))
    log_k_max = float(np.mean(np.log10(k_q[ok]) + boltzmann))
    return 10.0 ** log_k_max, int(ok.sum())


def prepare(df: pd.DataFrame, **_kwargs):
    pairs.require(df, required_columns(), "the Sandros model")
    dG = df[pairs.DG_ENT].to_numpy(dtype=float)
    k_q = df[pairs.K_Q].to_numpy(dtype=float)
    detected = df[pairs.CENSORED].to_numpy(dtype=float) == 0
    k_max, n_fit = fit_k_max(dG, k_q, detected)
    k = sandros_rate(dG, k_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_k = np.log10(np.maximum(k, 1e-30))
    return {"dG": dG, "k": k, "log_k": log_k, "k_max": k_max, "n_fit": n_fit}


def grid():
    return [Config(family="sandros_at_cutoff", name="sandros_at_cutoff", n_free=1,
                   params={})]


def classify(prep, cfg: Config) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return prep["k"] > pairs.K_Q_THRESHOLD


def rate_grid():
    return [Config(family="sandros_rate", name="sandros_rate", n_free=1, params={})]


def predicted_rate(prep, cfg: Config) -> np.ndarray:
    return prep["k"]


def note(prep) -> str:
    return (f"k_max fitted to {prep['k_max']:.3g} M-1 s-1 on {prep['n_fit']} detected "
            f"pairs (censored rows excluded from the fit)")


if __name__ == "__main__":
    import sys

    import run_benchmark

    # Extra flags pass straight through, so `--exclude-pc PC17` and the rest work here
    # exactly as they do on the harness.
    run_benchmark.main(["--theories", NAME, *sys.argv[1:]])
