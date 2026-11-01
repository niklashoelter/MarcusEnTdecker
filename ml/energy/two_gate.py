"""The two-gate verdict: does a pair quench within the photocatalyst's lifetime?

    from ml.energy.two_gate import lifetime_barrier, quenches

    t_pc = lifetime_barrier(tau0_ns)          # kcal/mol
    quenches(dG, barrier, t_pc)               # dG <= 0 and barrier <= t_pc

A pair is predicted to quench when the transfer is downhill (dG <= 0) and its Marcus
barrier is at most t_PC, the barrier at which k * tau0 = K_SV,crit in the exergonic limit
of the rate expression

    k = k_d / (1 + exp(dG/RT) + (k_-d/nu) exp(barrier/RT)),  exp(dG/RT) -> 0
    t_PC = RT ln((k_d tau0 / K_SV,crit - 1) / (k_-d/nu))

with k_-d = k_d / K_A, K_A = 4 pi N_A r^3 / 3 the Eigen encounter constant at the contact
distance r = 7 A, and nu = k_B T / h. tau0: measured excited-state lifetime of the
photocatalyst. escape_over_frequency gives k_-d/nu.
"""

from __future__ import annotations

import math

K_DIFF = 2.0e10                 # M-1 s-1, diffusion limit in acetonitrile
K_SV_CRIT = 10.0                # M-1, the smallest Stern-Volmer constant counted as quenching
CONTACT_DISTANCE = 7.0          # Angstrom
TEMPERATURE = 298.15            # K

R_KCAL = 1.9872e-3              # kcal mol-1 K-1
K_BOLTZMANN = 1.380649e-23      # J K-1
PLANCK = 6.62607015e-34         # J s
AVOGADRO = 6.02214076e23        # mol-1


def encounter_constant(distance: float = CONTACT_DISTANCE) -> float:
    """Eigen association constant 4 pi N_A r^3 / 3 in M-1, r in Angstrom."""
    return 4.0 / 3.0 * math.pi * (distance * 1e-10) ** 3 * AVOGADRO * 1e3


def escape_over_frequency(distance: float = CONTACT_DISTANCE,
                          temperature: float = TEMPERATURE, k_diff: float = K_DIFF) -> float:
    """k_-d / nu with k_-d = k_d / K_A and nu = k_B T / h."""
    return k_diff / encounter_constant(distance) / (K_BOLTZMANN * temperature / PLANCK)


def lifetime_barrier(tau0_ns: float, distance: float = CONTACT_DISTANCE,
                     k_sv_crit: float = K_SV_CRIT, temperature: float = TEMPERATURE) -> float:
    """t_PC in kcal/mol for a photocatalyst lifetime tau0 in ns.

    -inf when even a diffusion-limited pair cannot reach K_SV,crit within tau0.
    """
    if not tau0_ns > 0:
        raise ValueError(f"tau0 must be a positive lifetime in ns, not {tau0_ns!r}")
    ratio = K_DIFF * tau0_ns * 1e-9 / k_sv_crit - 1.0
    if ratio <= 0:
        return -math.inf
    escape_over_nu = escape_over_frequency(distance, temperature)
    return R_KCAL * temperature * math.log(ratio / escape_over_nu)


def quenches(dG: float, barrier: float, t_pc: float) -> bool:
    """The two-gate verdict: dG <= 0 and barrier <= t_PC, in the same unit."""
    return bool(dG <= 0 and barrier <= t_pc)
