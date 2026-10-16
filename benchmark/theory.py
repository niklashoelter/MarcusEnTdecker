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
K_DIFF_OVER_Z = 0.2             # k_d / Z in the irreversible form, Z = 1e11 M-1 s-1
EXP_LIMIT = 500.0               # cap on exponents (overflow)

K_BOLTZMANN = 1.380649e-23      # J K-1
PLANCK = 6.62607015e-34         # J s
AVOGADRO = 6.02214076e23        # mol-1
CONTACT_DISTANCE = 7.0          # Angstrom, fixed contact distance of the encounter complex

BARRIERS = ("marcus_symmetric", "marcus_asymmetric")


def pair_lambdas(pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0):
    """(lambda_forward, lambda_reverse) of PC(T1) + Q(S0) -> PC(S0) + Q(T1).

    Forward: the reactant state (PC on T1, Q on S0) at the product geometries, relative
    to its minimum. Reverse: the product state at the reactant geometries.
    """
    return pc_t1_s0 + q_s0_t1, pc_s0_t1 + q_t1_s0


def geometric_mean(lambda_forward, lambda_reverse):
    return np.sqrt(np.asarray(lambda_forward, float) * np.asarray(lambda_reverse, float))


def marcus_barrier_symmetric(dG, lam):
    """(dG + lambda)^2 / (4 lambda)."""
    dG, lam = np.asarray(dG, float), np.asarray(lam, float)
    return (dG + lam) ** 2 / (4.0 * lam)


def marcus_barrier_asymmetric(dG, lambda_forward, lambda_reverse):
    """Crossing of two parabolas of different curvature.

    Along x (reactant minimum at 0, product minimum at 1) the reactant surface is
    lambda_f x^2 and the product surface dG + lambda_r (x - 1)^2. The barrier is
    lambda_f x^2 at their crossing. Where they do not cross, or lambda_f = lambda_r, the
    symmetric barrier with the geometric-mean lambda is returned.
    """
    dG = np.asarray(dG, float)
    lf = np.asarray(lambda_forward, float)
    lr = np.asarray(lambda_reverse, float)

    radicand = lf * lr + (lf - lr) * dG
    crossing = (radicand >= 0) & ~np.isclose(lr, lf)
    x = (np.sqrt(np.where(crossing, radicand, 0.0)) - lr) / np.where(crossing, lf - lr, 1.0)
    symmetric = marcus_barrier_symmetric(dG, geometric_mean(lf, lr))
    return np.where(crossing, lf * x ** 2, symmetric)


def barrier(model, dG, lambda_forward, lambda_reverse):
    """The barrier of one of BARRIERS; the symmetric form uses the geometric-mean lambda."""
    if model == "marcus_symmetric":
        return marcus_barrier_symmetric(dG, geometric_mean(lambda_forward, lambda_reverse))
    if model == "marcus_asymmetric":
        return marcus_barrier_asymmetric(dG, lambda_forward, lambda_reverse)
    raise ValueError(f"unknown barrier model {model!r}; choose from {', '.join(BARRIERS)}")


def _exp(energy):
    return np.exp(np.minimum(np.asarray(energy, float) / RT, EXP_LIMIT))


def rate(dG, barrier_height, escape_over_nu):
    """k_d / (1 + exp(dG/RT) + (k_-d/nu) exp(barrier/RT)).

    Steady state of diffusion into an encounter complex, reversible transfer inside it,
    and separation of the successor complex at k_-d. escape_over_nu is k_-d / nu, from
    escape_over_frequency.
    """
    escape_over_nu = np.asarray(escape_over_nu, float)
    return K_DIFF / (1.0 + _exp(dG) + escape_over_nu * _exp(barrier_height))


def encounter_constant(distance=CONTACT_DISTANCE):
    """Eigen association constant 4 pi N_A r^3 / 3 in M-1, r in Angstrom."""
    volume = 4.0 / 3.0 * np.pi * (distance * 1e-10) ** 3          # m3
    return volume * AVOGADRO * 1e3


def escape_over_frequency(distance=CONTACT_DISTANCE):
    """k_-d / nu with k_-d = k_d / K_A and nu = k_B T / h."""
    return K_DIFF / encounter_constant(distance) / (K_BOLTZMANN * TEMPERATURE / PLANCK)


def lifetime_barrier(tau0, k_sv_crit, escape_over_nu):
    """The barrier at which k * tau0 = K_SV,crit when exp(dG/RT) is negligible.

    From k_d / (1 + (k_-d/nu) exp(barrier/RT)) * tau0 = K_SV,crit, with tau0 in s and
    K_SV,crit in M-1: RT ln((k_d tau0 / K_SV,crit - 1) / (k_-d/nu)).
    """
    tau0 = np.asarray(tau0, float)
    return RT * np.log((K_DIFF * tau0 / k_sv_crit - 1.0) / escape_over_nu)


def rate_irreversible(barrier_height):
    """k_d / (1 + (k_d/Z) exp(barrier/RT)): no back transfer."""
    return K_DIFF / (1.0 + K_DIFF_OVER_Z * _exp(barrier_height))


def sandros_rate(dG):
    """k_d / (1 + exp(dG/RT)): no barrier."""
    return K_DIFF / (1.0 + _exp(dG))
