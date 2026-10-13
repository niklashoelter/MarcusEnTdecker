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


def test_barrier_dispatches_by_model_name():
    dG, lf, lr = -5.0, 12.0, 30.0
    lam = np.sqrt(lf * lr)
    assert theory.barrier("marcus_symmetric", dG, lf, lr) == pytest.approx(
        theory.marcus_barrier_symmetric(dG, lam))
    assert theory.barrier("marcus_asymmetric", dG, lf, lr) == pytest.approx(
        theory.marcus_barrier_asymmetric(dG, lf, lr))
    with pytest.raises(ValueError):
        theory.barrier("unknown", dG, lf, lr)


def test_rate_of_the_full_scheme():
    escape_over_nu = theory.escape_over_frequency()
    expected = theory.K_DIFF / (1.0 + 1.0 + escape_over_nu * 1.0)
    assert theory.rate(0.0, 0.0, escape_over_nu) == pytest.approx(expected)


def test_rate_is_diffusion_limited_without_barrier_at_large_driving_force():
    escape_over_nu = theory.escape_over_frequency()
    assert theory.rate(-100.0, 0.0, escape_over_nu) == pytest.approx(
        theory.K_DIFF / (1.0 + escape_over_nu))


def test_rate_takes_the_escape_over_frequency_of_each_pair():
    r = np.array([5.0, 7.0, 10.0])
    k = theory.rate(np.full(3, -100.0), np.full(3, 10.0), theory.escape_over_frequency(r))
    expected = [theory.rate(-100.0, 10.0, theory.escape_over_frequency(d)) for d in r]
    assert k == pytest.approx(expected)
    assert np.all(np.diff(k) > 0)


def test_rate_at_the_lifetime_barrier_reaches_k_sv_crit_in_the_exergonic_limit():
    tau0, k_sv_crit = 1e-6, 10.0
    escape_over_nu = theory.escape_over_frequency()
    barrier = theory.lifetime_barrier(tau0, k_sv_crit, escape_over_nu)
    assert theory.rate(-100.0, barrier, escape_over_nu) * tau0 == pytest.approx(k_sv_crit)


def test_rate_does_not_overflow_for_huge_barriers():
    with np.errstate(all="raise"):
        k = theory.rate(np.array([50.0]), np.array([1e6]), theory.escape_over_frequency())
    assert k[0] == pytest.approx(0.0, abs=1e-100)


def test_irreversible_rate_ignores_back_transfer():
    expected = theory.K_DIFF / (1.0 + theory.K_DIFF_OVER_Z * np.exp(2.0 / theory.RT))
    assert theory.rate_irreversible(2.0) == pytest.approx(expected)


def test_sandros_rate_is_half_the_diffusion_limit_at_zero_driving_force():
    assert theory.sandros_rate(0.0) == pytest.approx(theory.K_DIFF / 2.0)


def test_encounter_constant_is_the_contact_sphere_in_litres_per_mole():
    assert theory.encounter_constant(7.0) == pytest.approx(0.8652, rel=1e-4)
    assert theory.encounter_constant(14.0) == pytest.approx(8.0 * theory.encounter_constant(7.0))


def test_escape_over_frequency_divides_the_dissociation_rate_by_kt_over_h():
    k_escape = theory.K_DIFF / theory.encounter_constant()
    frequency = theory.K_BOLTZMANN * theory.TEMPERATURE / theory.PLANCK
    assert theory.escape_over_frequency() == pytest.approx(k_escape / frequency)


def test_lifetime_barrier_gives_k_tau0_equal_to_k_sv_crit_in_the_exergonic_limit():
    tau0, k_sv_crit, escape_over_nu = np.array([2e-7, 1e-6, 5e-6]), 10.0, 3.7e-3
    barrier = theory.lifetime_barrier(tau0, k_sv_crit, escape_over_nu)
    k = theory.K_DIFF / (1.0 + escape_over_nu * np.exp(barrier / theory.RT))
    assert k * tau0 == pytest.approx(np.full(3, k_sv_crit))
    assert np.all(np.diff(barrier) > 0)


def test_lifetime_barrier_rises_by_3rt_ln_r_with_the_contact_distance():
    r = np.array([5.0, 7.0, 10.0])
    barrier = theory.lifetime_barrier(1e-6, 10.0, theory.escape_over_frequency(r))
    assert barrier - barrier[1] == pytest.approx(3.0 * theory.RT * np.log(r / 7.0), abs=1e-6)
