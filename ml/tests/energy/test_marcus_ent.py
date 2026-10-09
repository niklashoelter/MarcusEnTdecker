import numpy as np
import pytest

from ml.energy.marcus_ent import calculate_driving_force_and_barrier


def _pair(pc_E_T, q_E_T):
    # lambda_forward = pc_T_S + q_S_T = 20, lambda_reverse = pc_S_T + q_T_S = 15
    return calculate_driving_force_and_barrier(
        pc_adiabatic_E_T=pc_E_T,
        pc_lambda_S_T=5.0,
        pc_lambda_T_S=10.0,
        quencher_E_T=q_E_T,
        quencher_lambda_S_T=10.0,
        quencher_lambda_T_S=10.0,
    )


@pytest.mark.parametrize("dG", [-30.0, -10.0, 5.0, 25.0])
def test_backward_barrier_is_forward_barrier_minus_driving_force(dG):
    # Detailed balance: the backward barrier is measured from the product minimum at dG.
    dG_out, forward, backward = _pair(60.0, 60.0 + dG)
    assert np.isclose(dG_out, dG)
    assert np.isclose(backward, forward - dG)


def test_backward_barrier_equals_reverse_reaction_forward_barrier():
    # Running the endergonic pair backwards (partners swapped) gives the same crossing point.
    _, forward, backward = _pair(60.0, 65.0)
    _, forward_of_reverse, _ = calculate_driving_force_and_barrier(
        pc_adiabatic_E_T=65.0,
        pc_lambda_S_T=10.0,
        pc_lambda_T_S=10.0,
        quencher_E_T=60.0,
        quencher_lambda_S_T=5.0,
        quencher_lambda_T_S=10.0,
    )
    assert np.isclose(backward, forward_of_reverse)
    assert backward < forward
