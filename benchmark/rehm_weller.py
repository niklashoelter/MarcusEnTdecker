# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
Rehm-Weller: the empirical barrier, and the Balzani rate built on it.

Reads   ../data/exp_dataset.csv (through pairs.py)
Writes  nothing -- run_benchmark.py collects the predictions

    uv run benchmark/rehm_weller.py       score this one theory on its own

WHAT IT COMPUTES
    Rehm and Weller's empirical interpolation between the barrierless and the
    activation-controlled regimes,

        dG_act = sqrt((dG/2)^2 + dG_act_intrinsic^2) + dG/2

    with the intrinsic barrier taken from Marcus theory as lambda_rxn / 4, and lambda_rxn
    the geometric mean of the forward and reverse reorganization energies -- the same two
    sums Marcus uses, assembled by marcus.pair_lambdas so the two theories cannot end up
    disagreeing about what the pair's reorganization energy is.

    The expression is not a crossing of parabolas. It is a hyperbolic interpolation with
    the right behavior at both ends: it goes to dG_act_intrinsic at dG = 0, to dG for a
    strongly endergonic pair, and to zero for a strongly exergonic one. That last part is
    the point of it -- it does not turn back up in the inverted region the way the Marcus
    parabola does, which is what the Rehm-Weller plot was drawn to capture.

    That difference is why the two are worth benchmarking side by side rather than
    treating one as a refinement of the other, and it is also why the empirically optimal
    lambda scaling comes out lower for Rehm-Weller than for Marcus: with the intrinsic
    barrier at lambda_rxn/4, a smaller lambda reproduces the same effective barrier.

    Because only the symmetric mean of the two reorganization energies appears, nothing
    here depends on which of them is called forward and which reverse. That question,
    which does matter for the asymmetric Marcus form, is settled in marcus.py.

WHAT IT PREDICTS
    The same two things Marcus does, on the same grid, so the comparison is like for
    like: an exergonic-AND-surmountable classification gate at a swept barrier threshold,
    and a rate constant through theory.balzani_rate. Both crossed with the lambda-scaling
    sweep, with alpha = 1.0 always scored as its own family.
"""

import numpy as np
import pandas as pd

import marcus
import pairs
import theory
from theory import Config

NAME = "rehm_weller"
DESCRIPTION = "Rehm-Weller empirical barrier, and the Balzani rate built on it"


def required_columns():
    return marcus.required_columns()


# ---------------------------------------------------------------------------------
# The physics. Ported from
# phys_org_theories/rehm_weller_theory/rehm_weller_theory.py, with the driving force
# taken from the pair table and the reorganization energies arriving already combined.
# The arithmetic is unchanged.
# ---------------------------------------------------------------------------------

def rehm_weller_barrier(dG, lambda_forward, lambda_reverse,
                        mean_type="geometric", lam_scale=1.0):
    """Return (forward barrier, backward barrier) in kcal/mol."""
    dG = np.asarray(dG, dtype=float)
    lam_f = lam_scale * np.asarray(lambda_forward, dtype=float)
    lam_r = lam_scale * np.asarray(lambda_reverse, dtype=float)
    lam_rxn = theory.symmetric_lambda_rxn(lam_f, lam_r, mean_type)

    intrinsic = lam_rxn / 4.0
    dG_act = np.sqrt((dG / 2.0) ** 2 + intrinsic ** 2) + dG / 2.0
    return dG_act, dG_act - dG


def rehm_weller_rate(dG, lambda_forward, lambda_reverse,
                     k_diff=theory.K_DIFF_ACETONITRILE, temperature=theory.TEMPERATURE,
                     mean_type="geometric", lam_scale=1.0):
    """k_EnT in M-1 s-1, from the Rehm-Weller barrier through the Balzani rate equation."""
    dG_act, _ = rehm_weller_barrier(dG, lambda_forward, lambda_reverse,
                                    mean_type=mean_type, lam_scale=lam_scale)
    return theory.balzani_rate(dG, dG_act, k_diff=k_diff, temperature=temperature)


# ---------------------------------------------------------------------------------
# The benchmark interface.
# ---------------------------------------------------------------------------------

def prepare(df: pd.DataFrame, k_diff=theory.K_DIFF_ACETONITRILE, mean_type="geometric",
            legacy_pairing=False):
    dG = df[pairs.DG_ENT].to_numpy(dtype=float)
    lam_f, lam_r = marcus.pair_lambdas(df, legacy_pairing=legacy_pairing)
    prep = {"dG": dG, "exergonic": dG < 0, "barrier": {}, "log_k": {}, "k": {}}
    # The per-pair rate cutoff the K_SV label implies: K_SV > 10 is k_q > 10 / tau_0,
    # and that is k_q_floor. Carried here so `classify` can compare a predicted rate
    # against the cutoff the ground truth is actually set at.
    with np.errstate(divide="ignore", invalid="ignore"):
        prep["log_k_q_floor"] = np.log10(df[pairs.K_Q_FLOOR].to_numpy(dtype=float))
    for alpha in theory.LAMBDA_SCALES:
        act, _ = rehm_weller_barrier(dG, lam_f, lam_r, mean_type=mean_type,
                                     lam_scale=alpha)
        prep["barrier"][alpha] = act
        k = rehm_weller_rate(dG, lam_f, lam_r, k_diff=k_diff, mean_type=mean_type,
                             lam_scale=alpha)
        prep["k"][alpha] = k
        with np.errstate(divide="ignore", invalid="ignore"):
            prep["log_k"][alpha] = np.log10(np.maximum(k, 1e-30))
    return prep


def grid():
    out = []
    for alpha in theory.LAMBDA_SCALES:
        tag = theory.scale_tag(alpha)
        tuned_lambda = 0 if alpha == 1.0 else 1
        for thr in theory.BARRIER_THRESHOLDS:
            out.append(Config(
                family=f"rw_barrier{tag}",
                name=f"rw_barrier{tag}_act{thr:g}",
                n_free=1 + tuned_lambda,
                params={"kind": "barrier", "lam_scale": alpha, "threshold": thr},
            ))
        for thr in theory.LOG_RATE_THRESHOLDS:
            out.append(Config(
                family=f"rw_rate{tag}",
                name=f"rw_rate{tag}_logk{thr:g}",
                n_free=1 + tuned_lambda,
                params={"kind": "rate", "lam_scale": alpha, "threshold": thr},
            ))
        # The rate compared against the label's own cutoff -- per-pair, 10 / tau_0 --
        # instead of a threshold chosen by maximizing MCC. See the block in marcus.py.
        out.append(Config(
            family=f"rw_rate_at_cutoff{tag}",
            name=f"rw_rate_at_cutoff{tag}",
            n_free=tuned_lambda,
            params={"kind": "rate", "lam_scale": alpha,
                    "threshold": "floor"},
        ))
    return out


def classify(prep, cfg: Config) -> np.ndarray:
    p = cfg.params
    if p["kind"] == "barrier":
        with np.errstate(invalid="ignore"):
            return prep["exergonic"] & (prep["barrier"][p["lam_scale"]] < p["threshold"])
    with np.errstate(invalid="ignore"):
        # A scalar threshold is one chosen by maximizing MCC. "floor" instead means the
        # cutoff the label itself is set at, which is per-pair: the label is K_SV > 10,
        # and K_SV = k_q * tau_0, so the matching rate cutoff is 10 / tau_0 = k_q_floor.
        cut = prep["log_k_q_floor"] if p["threshold"] == "floor" else p["threshold"]
        return prep["log_k"][p["lam_scale"]] >= cut


def rate_grid():
    return [Config(family=f"rw_rate{theory.scale_tag(a)}",
                   name=f"rw_rate{theory.scale_tag(a)}",
                   n_free=0 if a == 1.0 else 1,
                   params={"lam_scale": a})
            for a in theory.LAMBDA_SCALES]


def predicted_rate(prep, cfg: Config) -> np.ndarray:
    return prep["k"][cfg.params["lam_scale"]]


if __name__ == "__main__":
    import sys

    import run_benchmark

    # Extra flags pass straight through, so `--exclude-pc PC17` and the rest work here
    # exactly as they do on the harness.
    run_benchmark.main(["--theories", NAME, *sys.argv[1:]])
