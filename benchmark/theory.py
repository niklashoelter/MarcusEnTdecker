"""
Barrier and rate expressions for triplet energy transfer, as functions of arrays.

Energies in kcal/mol, rate constants in M-1 s-1, 298.15 K.

Not a script; imported by the benchmark and further_analysis scripts.
"""

import numpy as np

R_KCAL = 1.9872e-3              # kcal mol-1 K-1
TEMPERATURE = 298.15            # K
RT = R_KCAL * TEMPERATURE
K_DIFF = 2.0e10                 # M-1 s-1, diffusion-limited rate constant in acetonitrile
K_ESCAPE_OVER_NU = 0.25         # k_-d / nu: escape from the encounter complex over the
                                # frequency factor of transfer inside it
K_DIFF_OVER_Z = 0.2             # k_d / Z in the irreversible form, Z = 1e11 M-1 s-1
EXP_LIMIT = 500.0               # exponents are capped here to avoid overflow

BARRIERS = ("marcus_symmetric", "marcus_asymmetric", "rehm_weller")


def pair_lambdas(pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0):
    """(lambda_forward, lambda_reverse) of PC(T1) + Q(S0) -> PC(S0) + Q(T1).

    Forward: the product state (PC on S0, Q on T1) at the reactant geometries, relative
    to its minimum. Reverse: the reactant state at the product geometries.
    """
    return pc_s0_t1 + q_t1_s0, pc_t1_s0 + q_s0_t1


def geometric_mean(lambda_forward, lambda_reverse):
    return np.sqrt(np.asarray(lambda_forward, float) * np.asarray(lambda_reverse, float))


def marcus_barrier_symmetric(dG, lam):
    """(dG + lambda)^2 / (4 lambda)."""
    dG, lam = np.asarray(dG, float), np.asarray(lam, float)
    return (dG + lam) ** 2 / (4.0 * lam)


def marcus_barrier_asymmetric(dG, lambda_forward, lambda_reverse):
    """Crossing of two parabolas of different curvature.

    Along x (reactant minimum at 0, product minimum at 1) the reactant surface is
    lambda_r x^2 and the product surface dG + lambda_f (x - 1)^2. The barrier is
    lambda_r x^2 at their crossing. Where they do not cross, or lambda_f = lambda_r, the
    symmetric barrier with the geometric-mean lambda is returned.
    """
    dG = np.asarray(dG, float)
    lf = np.asarray(lambda_forward, float)
    lr = np.asarray(lambda_reverse, float)

    radicand = lf * lr + (lr - lf) * dG
    crossing = (radicand >= 0) & ~np.isclose(lf, lr)
    x = (np.sqrt(np.where(crossing, radicand, 0.0)) - lf) / np.where(crossing, lr - lf, 1.0)
    symmetric = marcus_barrier_symmetric(dG, geometric_mean(lf, lr))
    return np.where(crossing, lr * x ** 2, symmetric)


def rehm_weller_barrier(dG, lam):
    """dG/2 + sqrt((dG/2)^2 + (lambda/4)^2)."""
    dG, lam = np.asarray(dG, float), np.asarray(lam, float)
    return dG / 2.0 + np.sqrt((dG / 2.0) ** 2 + (lam / 4.0) ** 2)


def barrier(model, dG, lambda_forward, lambda_reverse):
    """The barrier of one of BARRIERS; symmetric forms use the geometric-mean lambda."""
    if model == "marcus_symmetric":
        return marcus_barrier_symmetric(dG, geometric_mean(lambda_forward, lambda_reverse))
    if model == "marcus_asymmetric":
        return marcus_barrier_asymmetric(dG, lambda_forward, lambda_reverse)
    if model == "rehm_weller":
        return rehm_weller_barrier(dG, geometric_mean(lambda_forward, lambda_reverse))
    raise ValueError(f"unknown barrier model {model!r}; choose from {', '.join(BARRIERS)}")


def _exp(energy):
    return np.exp(np.minimum(np.asarray(energy, float) / RT, EXP_LIMIT))


def rate(dG, barrier_height):
    """k_d / (1 + exp(dG/RT) + (k_-d/nu) exp(barrier/RT)).

    Steady state of diffusion into an encounter complex, reversible transfer inside it,
    and separation of the successor complex at k_-d.
    """
    return K_DIFF / (1.0 + _exp(dG) + K_ESCAPE_OVER_NU * _exp(barrier_height))


def rate_irreversible(barrier_height):
    """k_d / (1 + (k_d/Z) exp(barrier/RT)): no back transfer."""
    return K_DIFF / (1.0 + K_DIFF_OVER_Z * _exp(barrier_height))


def sandros_rate(dG):
    """k_d / (1 + exp(dG/RT)): no barrier."""
    return K_DIFF / (1.0 + _exp(dG))
