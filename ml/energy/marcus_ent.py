import math

from typing import Tuple


def _compute_marcus_core(
    pc_adiabatic_E_T: float,
    pc_lambda_S_T: float,
    pc_lambda_T_S: float,
    quencher_E_T: float,
    quencher_lambda_S_T: float,
    quencher_lambda_T_S: float,
    symmetric_approx: bool = False,
) -> Tuple[float, float, float, float, float, bool, str]:
    """Shared core logic for computing driving force and activation barriers.

    Returns
    -------
    dG : float
        Thermodynamic driving force.
    dG_activation_forward : float
        Forward activation barrier.
    dG_activation_backward : float
        Backward activation barrier.
    lambda_forward : float
        Forward reorganization energy.
    lambda_reverse : float
        Reverse reorganization energy.
    symmetric_flag : bool
        Whether the symmetric approximation has been applied.
    reason : str
        Short reason why the symmetric approximation was used (if any).
    """

    dG = quencher_E_T - pc_adiabatic_E_T
    lambda_forward = pc_lambda_T_S + quencher_lambda_S_T
    lambda_reverse = pc_lambda_S_T + quencher_lambda_T_S

    reason = ""
    root_arg = lambda_forward * lambda_reverse + ((lambda_forward - lambda_reverse) * dG)
    auto_symmetric = root_arg < 0 or abs(lambda_forward - lambda_reverse) < 1e-10

    if auto_symmetric:
        symmetric_approx = True
        reason = "parabolas_do_not_intersect_or_symmetric"
    if not symmetric_approx:
        root_term = math.sqrt(root_arg)
        dG_activation = lambda_forward * (
            ((root_term - lambda_reverse) / (lambda_forward - lambda_reverse)) ** 2
        )
    else:
        lambda_rxn = math.sqrt(max(lambda_forward * lambda_reverse, 0.0))
        if lambda_rxn <= 0:
            # Fallback in case of numerical issues
            dG_activation = 0.0
            reason = reason or "invalid_lambda_product"
        else:
            dG_activation = (lambda_rxn + dG) ** 2 / (4 * lambda_rxn)

    # Backward barrier from the product minimum at dG
    dG_activation_backward = dG_activation - dG
    return (
        dG,
        dG_activation,
        dG_activation_backward,
        lambda_forward,
        lambda_reverse,
        symmetric_approx,
        reason,
    )


def calculate_driving_force_and_barrier(
    pc_adiabatic_E_T: float,
    pc_lambda_S_T: float,
    pc_lambda_T_S: float,
    quencher_E_T: float,
    quencher_lambda_S_T: float,
    quencher_lambda_T_S: float,
    symmetric_approx: bool = False,
) -> Tuple[float, float, float]:
    """Driving force dG and forward/backward Marcus barriers for EnT.

    ``symmetric_approx``: lambda = sqrt(lambda_forward * lambda_reverse); applied
    automatically when the parabolas do not intersect (negative root argument) or
    lambda_forward ~= lambda_reverse.
    """

    dG, dG_activation, dG_activation_backward, _fallback = calculate_driving_force_and_barrier_flagged(
        pc_adiabatic_E_T,
        pc_lambda_S_T,
        pc_lambda_T_S,
        quencher_E_T,
        quencher_lambda_S_T,
        quencher_lambda_T_S,
        symmetric_approx,
    )
    return dG, dG_activation, dG_activation_backward


def calculate_driving_force_and_barrier_flagged(
    pc_adiabatic_E_T: float,
    pc_lambda_S_T: float,
    pc_lambda_T_S: float,
    quencher_E_T: float,
    quencher_lambda_S_T: float,
    quencher_lambda_T_S: float,
    symmetric_approx: bool = False,
) -> Tuple[float, float, float, bool]:
    """As `calculate_driving_force_and_barrier`, plus ``symmetric_fallback``: True when
    the symmetric approximation was applied automatically (not requested).
    """

    (
        dG,
        dG_activation,
        dG_activation_backward,
        _lambda_forward,
        _lambda_reverse,
        symmetric_flag,
        reason,
    ) = _compute_marcus_core(
        pc_adiabatic_E_T,
        pc_lambda_S_T,
        pc_lambda_T_S,
        quencher_E_T,
        quencher_lambda_S_T,
        quencher_lambda_T_S,
        symmetric_approx,
    )

    symmetric_fallback = bool(symmetric_flag and not symmetric_approx)
    if symmetric_fallback:
        # warn only for the automatic fallback
        print(
            "Warning: Product and reactant parabolas do not intersect properly. Using symmetric approximation.",
            f"Reason: {reason}",
        )

    return dG, dG_activation, dG_activation_backward, symmetric_fallback
