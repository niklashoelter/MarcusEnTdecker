# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
Marcus theory: the crossing of two harmonic parabolas, as a barrier and as a rate.

Reads   ../data/exp_dataset.csv (through pairs.py)
Writes  nothing -- run_benchmark.py collects the predictions

    uv run benchmark/marcus.py       score this one theory on its own

HOW THE PAIR'S REORGANIZATION ENERGY IS ASSEMBLED
    The dataset carries each partner's lambda and leaves the combination to the benchmark
    on purpose: how the two terms add is a modelling decision, and it belongs where it can
    be stated and varied, not frozen into the table. So here it is, stated.

    The forward reaction is PC(T1) + Q(S0) -> PC(S0) + Q(T1). For a reaction R -> P the
    reorganization energy is the product state evaluated at the reactant geometry, minus
    the product state at its own geometry. Applied to each partner separately:

        photocatalyst   ends on the SINGLET surface, starting from the T1 geometry
                        -> E_S0(T1 geom) - E_S0(S0 geom)  =  pc_lambda_T1_to_S0
        quencher        ends on the TRIPLET surface, starting from the S0 geometry
                        -> E_T1(S0 geom) - E_T1(T1 geom)  =  q_lambda_T_S

        lambda_forward = pc_lambda_T1_to_S0 + q_lambda_T_S      (singlet-PC + triplet-Q)
        lambda_reverse = pc_lambda_S0_to_T1 + q_lambda_S_T      (triplet-PC + singlet-Q)

    Both partners contribute additively because the two molecules relax independently;
    there is no cross term in the classical two-parabola picture.

    THIS IS THE OPPOSITE PAIRING TO THE CODE THIS WAS PORTED FROM. The original built
    `lambda_forward = pc_lambda_T_S + quencher_lambda_S_T`, which in that project's own
    column definitions is the photocatalyst's TRIPLET-surface term plus the quencher's
    SINGLET-surface term -- the reverse reaction's reorganization energy. The two naming
    schemes read their subscripts in opposite orders, which is exactly the trap the
    photocatalyst script warned about when it chose geometry-explicit names, and the
    original carried a whole section in its own notes about call sites having to swap the
    photocatalyst tuple by hand.

    It matters in one place only. The symmetric form and the whole of Rehm-Weller combine
    the two into a symmetric mean, so swapping them changes nothing there. The asymmetric
    form is the exception, and it is most sensitive on exactly the photocatalysts with
    the largest forward/reverse asymmetry -- PC18, PC26, PC27. `--legacy-lambda-pairing`
    restores the original assignment so the difference can be measured rather than
    argued about.

WHAT IT COMPUTES
    Asymmetric, the exact intersection of two parabolas of different curvature,

        dG_act = lambda_f * ((sqrt(lambda_f*lambda_r + (lambda_f - lambda_r)*dG)
                              - lambda_r) / (lambda_f - lambda_r))^2

    falling back to the symmetric form where the radicand goes negative (the parabolas do
    not cross) or the two lambdas coincide to within 1e-10.

    Symmetric, the textbook expression with one effective lambda,

        dG_act = (dG + lambda_rxn)^2 / (4 * lambda_rxn)

    with lambda_rxn the geometric mean of forward and reverse.

    The rate constant is the Balzani form, k = k_diff / (1 + exp(dG/kT) +
    0.25 exp(dG_act/kT)); see theory.balzani_rate.

WHAT IT PREDICTS
    Classification, as a two-part gate: the pair quenches if transfer is exergonic AND
    its barrier is below a threshold. The dG < 0 half is fixed, because the question
    being asked is whether the barrier adds anything to the thermodynamic rule, and that
    only means something if the thermodynamic rule is inside it unchanged. The threshold
    half is swept.

    Rate, as log10 k_EnT -- thresholded for classification, and compared against the
    measured k_q in the rate-agreement table.

    Both are crossed with the lambda-scaling sweep documented in theory.LAMBDA_SCALES.
    alpha = 1.0 (raw DFT lambda) is always scored as its own family, so the tuned result
    never quietly replaces the untuned one.
"""

import numpy as np
import pandas as pd

import pairs
import theory
from theory import Config

NAME = "marcus"
DESCRIPTION = "Marcus barrier from two harmonic parabolas, and the Balzani rate built on it"


def required_columns():
    return [pairs.DG_ENT,
            pairs.PC_LAMBDA_S0_TO_T1, pairs.PC_LAMBDA_T1_TO_S0,
            pairs.Q_LAMBDA_S_T, pairs.Q_LAMBDA_T_S]


def pair_lambdas(df: pd.DataFrame, legacy_pairing: bool = False):
    """(lambda_forward, lambda_reverse) in kcal/mol, at alpha = 1.

    See the module docstring for the derivation and for what `legacy_pairing` restores.
    """
    pairs.require(df, required_columns(), "Marcus theory")
    pc_singlet = df[pairs.PC_LAMBDA_T1_TO_S0].to_numpy(dtype=float)
    pc_triplet = df[pairs.PC_LAMBDA_S0_TO_T1].to_numpy(dtype=float)
    q_singlet = df[pairs.Q_LAMBDA_S_T].to_numpy(dtype=float)
    q_triplet = df[pairs.Q_LAMBDA_T_S].to_numpy(dtype=float)
    forward = pc_singlet + q_triplet
    reverse = pc_triplet + q_singlet
    return (reverse, forward) if legacy_pairing else (forward, reverse)


# ---------------------------------------------------------------------------------
# The physics. Ported from phys_org_theories/marcus_theory/marcus_theory.py, with three
# changes: the driving force is taken from the pair table instead of recomputed from two
# triplet energies, the per-row scalar branch is a mask so the whole table evaluates at
# once, and the reorganization energies arrive already combined into a forward and a
# reverse term (see pair_lambdas). The arithmetic is unchanged.
# ---------------------------------------------------------------------------------

def marcus_barrier(dG, lambda_forward, lambda_reverse,
                   symmetric_approx=False, mean_type="geometric", lam_scale=1.0):
    """Return (forward barrier, backward barrier) in kcal/mol.

    `lam_scale` multiplies both reorganization energies before they enter the barrier,
    and is 1.0 for raw DFT lambda. See theory.LAMBDA_SCALES for what it is for and for
    the two caveats that have to travel with it.
    """
    dG = np.asarray(dG, dtype=float)
    lam_f = lam_scale * np.asarray(lambda_forward, dtype=float)
    lam_r = lam_scale * np.asarray(lambda_reverse, dtype=float)

    # The radicand of the intersection of the two parabolas.
    root_arg = lam_f * lam_r + (lam_f - lam_r) * dG

    # Fall back to the symmetric form where the parabolas do not intersect, or where the
    # two curvatures coincide, exactly as the original did per row.
    with np.errstate(invalid="ignore"):
        degenerate = (root_arg < 0) | (np.abs(lam_f - lam_r) < 1e-10)
    use_symmetric = np.full(np.shape(dG), bool(symmetric_approx)) | degenerate

    lam_rxn = theory.symmetric_lambda_rxn(lam_f, lam_r, mean_type)
    with np.errstate(divide="ignore", invalid="ignore"):
        act_symmetric = (dG + lam_rxn) ** 2 / (4.0 * lam_rxn)
    # A non-positive effective lambda means a barrierless crossing, not a division. NaN
    # inputs stay NaN rather than collapsing to zero.
    act_symmetric = np.where(np.isnan(lam_rxn) | np.isnan(dG), np.nan,
                             np.where(lam_rxn > 0, act_symmetric, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.where(use_symmetric, 1.0, lam_f - lam_r)
        root = np.sqrt(np.where(root_arg < 0, np.nan, root_arg))
        act_asymmetric = lam_f * ((root - lam_r) / denominator) ** 2

    dG_act = np.where(use_symmetric, act_symmetric, act_asymmetric)
    return dG_act, dG_act - dG


def marcus_rate(dG, lambda_forward, lambda_reverse,
                k_diff=theory.K_DIFF_ACETONITRILE, temperature=theory.TEMPERATURE,
                symmetric_approx=False, mean_type="geometric", lam_scale=1.0):
    """k_EnT in M-1 s-1, from the Marcus barrier through the Balzani rate equation."""
    dG_act, _ = marcus_barrier(dG, lambda_forward, lambda_reverse,
                               symmetric_approx=symmetric_approx,
                               mean_type=mean_type, lam_scale=lam_scale)
    return theory.balzani_rate(dG, dG_act, k_diff=k_diff, temperature=temperature)


# ---------------------------------------------------------------------------------
# The benchmark interface.
# ---------------------------------------------------------------------------------

def prepare(df: pd.DataFrame, k_diff=theory.K_DIFF_ACETONITRILE, mean_type="geometric",
            legacy_pairing=False):
    """Barriers and rates for every lambda scale, computed once for the whole grid."""
    dG = df[pairs.DG_ENT].to_numpy(dtype=float)
    lam_f, lam_r = pair_lambdas(df, legacy_pairing=legacy_pairing)
    prep = {"dG": dG, "exergonic": dG < 0, "barrier": {}, "log_k": {}, "k": {}}
    # The per-pair rate cutoff the K_SV label implies: K_SV > 10 is k_q > 10 / tau_0,
    # and that is k_q_floor. Carried here so `classify` can compare a predicted rate
    # against the cutoff the ground truth is actually set at.
    with np.errstate(divide="ignore", invalid="ignore"):
        prep["log_k_q_floor"] = np.log10(df[pairs.K_Q_FLOOR].to_numpy(dtype=float))
    for alpha in theory.LAMBDA_SCALES:
        for form, symmetric in (("asym", False), ("sym", True)):
            act, _ = marcus_barrier(dG, lam_f, lam_r, symmetric_approx=symmetric,
                                    mean_type=mean_type, lam_scale=alpha)
            prep["barrier"][(form, alpha)] = act
        k = marcus_rate(dG, lam_f, lam_r, k_diff=k_diff, mean_type=mean_type,
                        lam_scale=alpha)
        prep["k"][alpha] = k
        with np.errstate(divide="ignore", invalid="ignore"):
            prep["log_k"][alpha] = np.log10(np.maximum(k, 1e-30))
    return prep


def grid():
    """Every classification setting: barrier gates and rate cutoffs, across alpha."""
    out = []
    for alpha in theory.LAMBDA_SCALES:
        tag = theory.scale_tag(alpha)
        tuned_lambda = 0 if alpha == 1.0 else 1
        for form in ("asym", "sym"):
            for thr in theory.BARRIER_THRESHOLDS:
                out.append(Config(
                    family=f"marcus_barrier_{form}{tag}",
                    name=f"marcus_barrier_{form}{tag}_act{thr:g}",
                    n_free=1 + tuned_lambda,
                    params={"kind": "barrier", "form": form, "lam_scale": alpha,
                            "threshold": thr},
                ))
        for thr in theory.LOG_RATE_THRESHOLDS:
            out.append(Config(
                family=f"marcus_rate{tag}",
                name=f"marcus_rate{tag}_logk{thr:g}",
                n_free=1 + tuned_lambda,
                params={"kind": "rate", "lam_scale": alpha, "threshold": thr},
            ))
        # The same rate compared against the cutoff that defines the label itself
        # rather than against a threshold chosen by maximizing MCC. The label is
        # K_SV > 10, so that cutoff is per-pair -- 10 / tau_0, which is k_q_floor. At
        # alpha = 1 this carries no free parameters at all, which makes it the only
        # entry in the folder besides the adiabatic rule that can be set beside that
        # rule without a tuning asterisk.
        out.append(Config(
            family=f"marcus_rate_at_cutoff{tag}",
            name=f"marcus_rate_at_cutoff{tag}",
            n_free=tuned_lambda,
            params={"kind": "rate", "lam_scale": alpha,
                    "threshold": "floor"},
        ))
    return out


def classify(prep, cfg: Config) -> np.ndarray:
    p = cfg.params
    if p["kind"] == "barrier":
        barrier = prep["barrier"][(p["form"], p["lam_scale"])]
        with np.errstate(invalid="ignore"):
            return prep["exergonic"] & (barrier < p["threshold"])
    with np.errstate(invalid="ignore"):
        # A scalar threshold is one chosen by maximizing MCC. "floor" instead means the
        # cutoff the label itself is set at, which is per-pair: the label is K_SV > 10,
        # and K_SV = k_q * tau_0, so the matching rate cutoff is 10 / tau_0 = k_q_floor.
        cut = prep["log_k_q_floor"] if p["threshold"] == "floor" else p["threshold"]
        return prep["log_k"][p["lam_scale"]] >= cut


def rate_grid():
    """One entry per distinct rate prediction: the rate depends on alpha and nothing else."""
    return [Config(family=f"marcus_rate{theory.scale_tag(a)}",
                   name=f"marcus_rate{theory.scale_tag(a)}",
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
