import numpy as np
import pytest

import theory


def test_pair_lambdas_take_the_surface_each_partner_ends_on():
    forward, reverse = theory.pair_lambdas(pc_s0_t1=1.0, pc_t1_s0=2.0, q_s0_t1=10.0, q_t1_s0=20.0)
    assert forward == 21.0
    assert reverse == 12.0


def test_symmetric_barrier_is_quarter_lambda_at_zero_driving_force():
    assert theory.marcus_barrier_symmetric(0.0, 20.0) == pytest.approx(5.0)


def test_symmetric_barrier_vanishes_when_driving_force_equals_lambda():
    assert theory.marcus_barrier_symmetric(-20.0, 20.0) == pytest.approx(0.0)


def test_asymmetric_barrier_equals_symmetric_for_equal_lambdas():
    dG = np.array([-15.0, -3.0, 0.0, 4.0])
    assert theory.marcus_barrier_asymmetric(dG, 20.0, 20.0) == pytest.approx(
        theory.marcus_barrier_symmetric(dG, 20.0))


def test_asymmetric_barrier_vanishes_when_driving_force_equals_forward_lambda():
    assert theory.marcus_barrier_asymmetric(-10.0, 10.0, 30.0) == pytest.approx(0.0, abs=1e-12)


def test_asymmetric_barrier_lies_where_the_two_parabolas_cross():
    dG, lf, lr = -4.0, 12.0, 30.0
    barrier = theory.marcus_barrier_asymmetric(dG, lf, lr)
    x = np.sqrt(barrier / lr)                     # reactant surface lr * x**2
    assert lr * x ** 2 == pytest.approx(dG + lf * (x - 1.0) ** 2)
    assert 0.0 < x < 1.0


def test_asymmetric_barrier_falls_back_to_symmetric_without_a_crossing():
    dG, lf, lr = 60.0, 30.0, 10.0                 # lf * lr + (lr - lf) * dG < 0
    assert theory.marcus_barrier_asymmetric(dG, lf, lr) == pytest.approx(
        theory.marcus_barrier_symmetric(dG, np.sqrt(lf * lr)))


def test_rehm_weller_barrier_limits():
    assert theory.rehm_weller_barrier(0.0, 20.0) == pytest.approx(5.0)
    assert theory.rehm_weller_barrier(-2000.0, 20.0) == pytest.approx(0.0, abs=0.02)
    assert theory.rehm_weller_barrier(2000.0, 20.0) == pytest.approx(2000.0, abs=0.02)


def test_barrier_dispatches_by_model_name():
    dG, lf, lr = -5.0, 12.0, 30.0
    lam = np.sqrt(lf * lr)
    assert theory.barrier("marcus_symmetric", dG, lf, lr) == pytest.approx(
        theory.marcus_barrier_symmetric(dG, lam))
    assert theory.barrier("marcus_asymmetric", dG, lf, lr) == pytest.approx(
        theory.marcus_barrier_asymmetric(dG, lf, lr))
    assert theory.barrier("rehm_weller", dG, lf, lr) == pytest.approx(
        theory.rehm_weller_barrier(dG, lam))
    with pytest.raises(ValueError):
        theory.barrier("unknown", dG, lf, lr)


def test_rate_of_the_full_scheme():
    expected = theory.K_DIFF / (1.0 + 1.0 + theory.K_ESCAPE_OVER_NU * 1.0)
    assert theory.rate(0.0, 0.0) == pytest.approx(expected)


def test_rate_is_diffusion_limited_without_barrier_at_large_driving_force():
    assert theory.rate(-100.0, 0.0) == pytest.approx(theory.K_DIFF / (1.0 + theory.K_ESCAPE_OVER_NU))


def test_rate_does_not_overflow_for_huge_barriers():
    with np.errstate(all="raise"):
        k = theory.rate(np.array([50.0]), np.array([1e6]))
    assert k[0] == pytest.approx(0.0, abs=1e-100)


def test_irreversible_rate_ignores_back_transfer():
    expected = theory.K_DIFF / (1.0 + theory.K_DIFF_OVER_Z * np.exp(2.0 / theory.RT))
    assert theory.rate_irreversible(2.0) == pytest.approx(expected)


def test_sandros_rate_is_half_the_diffusion_limit_at_zero_driving_force():
    assert theory.sandros_rate(0.0) == pytest.approx(theory.K_DIFF / 2.0)
