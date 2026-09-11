# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
DvTE: the quencher's triplet energy as a distribution over geometries, not a number.

Reads   ../data/exp_dataset.csv (through pairs.py)
Writes  nothing -- run_benchmark.py collects the predictions

    uv run benchmark/dvte.py       score this one theory on its own

WHAT IT COMPUTES
    An adiabatic triplet energy is one number for one optimized geometry. A molecule in
    solution at room temperature is not at that geometry. The dynamic vertical triplet
    energy samples the vertical singlet-triplet gap along a Milo molecular-dynamics
    trajectory and keeps the whole distribution, summarized as a mean and a standard
    deviation (`../data/computational_data_scripts/quencher_dvte.py`, from 236482 sampled
    geometries across the 48 quenchers).

    The classifier asks a different question from the adiabatic rule. Not "is the
    quencher's relaxed triplet below the photocatalyst's", but "does the quencher spend
    any appreciable fraction of its time in a geometry whose triplet is accessible":

        DvTE(p) = mu + Phi^-1(p) * sigma          the p-quantile of the sampled gaps
        quenches  <=>  E_T(PC) >= DvTE(p)

    p is small -- 0.005 to 0.015 -- so DvTE(p) sits about 2.2 to 2.6 standard deviations
    below the mean gap. A quencher with a wide distribution therefore looks accessible at
    a photocatalyst energy well below its mean gap, which is what lets this method
    separate pairs the adiabatic rule cannot. In the original benchmark it had markedly
    better specificity than the adiabatic rule for exactly that reason.

    Kanai et al., Chem 2024, report R^2 = 0.97 and MAE = 1.7 kcal/mol for DvTE against
    experimental triplet energies, versus R^2 = 0.51 and MAE = 9.5 for adiabatic E_T.
    That is a literature result on 24 small organics, not a result on this screen, and
    should not be quoted as if it were.

WHAT IS AND IS NOT PORTED
    The normal-quantile form is ported. It is exact from what exp_dataset.csv carries, since
    `q_dvte_mu_md` and `q_dvte_std_md` are the mean and standard deviation of the sampled
    gaps and the quantile is a closed-form function of them.

    The `_kde` variant is NOT ported. It takes the same quantile from a Gaussian kernel
    density estimate of the samples rather than from a normal fit, so it needs the sample
    distribution, which exp_dataset.csv does not carry -- it lives in
    `../data/computational_data_processed/quencher_properties_calc.xlsx` and, sample by
    sample, `../data/computational_data_raw/quencher_adiabatic_DvTE/quencher_dvte_samples.csv.gz`. Everything downstream in this repository reads
    exp_dataset.csv and nothing else, so rather than reach around that rule for a variant that
    tracked the normal one to within 0.07 kcal/mol, it is left out. If it turns out to be
    wanted, the right fix is to carry the four `_kde` quantiles into exp_dataset.csv.

    The cutoff is not swept. The original swept the four values and reported the best,
    which is an in-sample choice, and its own audit recorded that a finer grid moved MCC
    by less than 0.02. All four are reported here as separate methods instead, so the
    reader sees the spread rather than its maximum.
"""

import statistics

import numpy as np
import pandas as pd

import pairs
from theory import Config

NAME = "dvte"
DESCRIPTION = "quencher triplet energy as a sampled distribution, thresholded at a low quantile"

# The four quantiles the DvTE table was built for. Not extended: see the module
# docstring on why the cutoff is reported rather than tuned.
CUTOFFS = (0.005, 0.008, 0.01, 0.015)


def required_columns():
    return [pairs.Q_DVTE_MU, pairs.Q_DVTE_STD, pairs.PC_E_T]


def dvte_threshold(mu, sigma, cutoff):
    """The p-quantile of a normal distribution with that mean and standard deviation.

    `statistics.NormalDist().inv_cdf` is the standard library's exact inverse normal CDF,
    so this needs no scipy. It reproduces the `dvte_<p>` columns of
    `data/computational_data_processed/quencher_properties_calc.xlsx` to the rounding of
    that file.
    """
    z = statistics.NormalDist().inv_cdf(cutoff)
    return np.asarray(mu, dtype=float) + z * np.asarray(sigma, dtype=float)


def prepare(df: pd.DataFrame, **_kwargs):
    pairs.require(df, required_columns(), "the DvTE classifier")
    pc_E_T = df[pairs.PC_E_T].to_numpy(dtype=float)
    mu = df[pairs.Q_DVTE_MU].to_numpy(dtype=float)
    sigma = df[pairs.Q_DVTE_STD].to_numpy(dtype=float)
    return {"pc_E_T": pc_E_T,
            "threshold": {c: dvte_threshold(mu, sigma, c) for c in CUTOFFS}}


def grid():
    # One family per cutoff, each with a single setting, so nothing is selected by MCC.
    return [Config(family=f"dvte_p{c:g}", name=f"dvte_p{c:g}", n_free=0,
                   params={"cutoff": c})
            for c in CUTOFFS]


def classify(prep, cfg: Config) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return prep["pc_E_T"] >= prep["threshold"][cfg.params["cutoff"]]


def rate_grid():
    """DvTE states an accessibility criterion, not a rate constant."""
    return []


def predicted_rate(prep, cfg: Config):
    return None


if __name__ == "__main__":
    import sys

    import run_benchmark

    # Extra flags pass straight through, so `--exclude-pc PC17` and the rest work here
    # exactly as they do on the harness.
    run_benchmark.main(["--theories", NAME, *sys.argv[1:]])
