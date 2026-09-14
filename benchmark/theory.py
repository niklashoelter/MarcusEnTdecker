"""
The rate and barrier expressions, as plain functions of arrays.

Energies kcal/mol, rate constants M-1 s-1, 298.15 K. Equation numbers refer to SI
Section 4.1.

Not a script; imported by classification.py and rates.py.
"""

import statistics

import numpy as np

R_KCAL = 1.9872e-3              # kcal mol-1 K-1
TEMPERATURE = 298.15            # K
RT = R_KCAL * TEMPERATURE
K_DIFF = 2.0e10                 # M-1 s-1, diffusion limit in acetonitrile
EXP_LIMIT = 500.0               # exponents are capped here to avoid overflow


def pair_lambdas(pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0):
    """(lambda_forward, lambda_reverse) of PC(T1) + Q(S0) -> PC(S0) + Q(T1), Eqs. S4-S5.

    Each partner contributes the surface it ends on, at the geometry it starts from: the
    photocatalyst returns to the singlet surface from its T1 geometry, the quencher
    reaches the triplet surface from its S0 geometry.
    """
    return pc_s0_t1 + q_t1_s0, pc_t1_s0 + q_s0_t1


def effective_lambda(lambda_forward, lambda_reverse):
    """The geometric mean of forward and reverse."""
    return np.sqrt(np.maximum(np.asarray(lambda_forward) * lambda_reverse, 0.0))


def marcus_barrier(dG, lambda_forward, lambda_reverse):
    """Marcus barrier at the crossing of two parabolas, Eq. S6.

    Where the parabolas do not cross, or have the same curvature, the symmetric form with
    the geometric-mean lambda is used instead, Eq. S7.
    """
    dG = np.asarray(dG, float)
    lf = np.asarray(lambda_forward, float)
    lr = np.asarray(lambda_reverse, float)

    radicand = lf * lr + (lf - lr) * dG
    with np.errstate(invalid="ignore", divide="ignore"):
        symmetric = (radicand < 0) | (np.abs(lf - lr) < 1e-10)
        lam = effective_lambda(lf, lr)
        barrier_symmetric = np.where(lam > 0, (dG + lam) ** 2 / (4.0 * lam), 0.0)
        barrier_symmetric = np.where(np.isnan(lam) | np.isnan(dG), np.nan, barrier_symmetric)
        root = np.sqrt(np.where(symmetric, 0.0, radicand))
        barrier_crossing = lf * ((root - lr) / np.where(symmetric, 1.0, lf - lr)) ** 2
    return np.where(symmetric, barrier_symmetric, barrier_crossing)


def rehm_weller_barrier(dG, lam):
    """Rehm-Weller barrier for one effective lambda, Eq. S8."""
    dG = np.asarray(dG, float)
    return np.sqrt((dG / 2.0) ** 2 + (np.asarray(lam, float) / 4.0) ** 2) + dG / 2.0


def balzani_rate(dG, barrier, k_diff=K_DIFF):
    """k = k_diff / (1 + exp(dG/RT) + 0.25 exp(barrier/RT)), Eq. S9."""
    exp_dG = np.exp(np.minimum(np.asarray(dG, float) / RT, EXP_LIMIT))
    exp_barrier = np.exp(np.minimum(np.asarray(barrier, float) / RT, EXP_LIMIT))
    return k_diff / (1.0 + exp_dG + 0.25 * exp_barrier)


def marcus_rate(dG, lambda_forward, lambda_reverse):
    return balzani_rate(dG, marcus_barrier(dG, lambda_forward, lambda_reverse))


def rehm_weller_rate(dG, lam):
    return balzani_rate(dG, rehm_weller_barrier(dG, lam))


def sandros_rate(dG, k_max):
    """k = k_max / (1 + exp(dG/RT)): no barrier, only the Boltzmann term."""
    return k_max / (1.0 + np.exp(np.clip(np.asarray(dG, float) / RT, -EXP_LIMIT, EXP_LIMIT)))


def dvte(mu, std, p):
    """The p-quantile of the sampled vertical gaps, taken as normally distributed."""
    return np.asarray(mu, float) + statistics.NormalDist().inv_cdf(p) * np.asarray(std, float)
