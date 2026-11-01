import math

import pytest

from ml.energy import two_gate


def test_encounter_constant_is_the_contact_sphere_in_litres_per_mole():
    assert two_gate.encounter_constant(7.0) == pytest.approx(0.8652, rel=1e-4)


def test_escape_over_frequency_divides_the_dissociation_rate_by_kt_over_h():
    k_escape = two_gate.K_DIFF / two_gate.encounter_constant(7.0)
    frequency = two_gate.K_BOLTZMANN * two_gate.TEMPERATURE / two_gate.PLANCK
    assert two_gate.escape_over_frequency() == pytest.approx(k_escape / frequency)
    assert two_gate.escape_over_frequency(k_diff=1e10) == pytest.approx(
        0.5 * two_gate.escape_over_frequency())


@pytest.mark.parametrize("tau0_ns", [200.0, 1000.0, 5000.0])
def test_lifetime_barrier_gives_k_tau0_equal_to_k_sv_crit_in_the_exergonic_limit(tau0_ns):
    t_pc = two_gate.lifetime_barrier(tau0_ns)
    escape_over_nu = (two_gate.K_DIFF / two_gate.encounter_constant()
                      / (two_gate.K_BOLTZMANN * two_gate.TEMPERATURE / two_gate.PLANCK))
    rt = two_gate.R_KCAL * two_gate.TEMPERATURE
    k = two_gate.K_DIFF / (1.0 + escape_over_nu * math.exp(t_pc / rt))
    assert k * tau0_ns * 1e-9 == pytest.approx(two_gate.K_SV_CRIT)


def test_lifetime_barrier_rises_with_the_lifetime_and_the_contact_distance():
    assert two_gate.lifetime_barrier(2000.0) > two_gate.lifetime_barrier(1000.0)
    rt = two_gate.R_KCAL * two_gate.TEMPERATURE
    shift = two_gate.lifetime_barrier(1000.0, distance=10.0) - two_gate.lifetime_barrier(1000.0)
    assert shift == pytest.approx(3.0 * rt * math.log(10.0 / 7.0))


def test_lifetime_too_short_for_the_detection_floor_admits_no_barrier():
    # k_d * tau0 = 8 M-1 < K_SV,crit even at the diffusion limit
    assert two_gate.lifetime_barrier(0.4) == -math.inf


@pytest.mark.parametrize("tau0_ns", [0.0, -5.0, float("nan")])
def test_lifetime_must_be_positive(tau0_ns):
    with pytest.raises(ValueError, match="tau0"):
        two_gate.lifetime_barrier(tau0_ns)


def test_two_gate_needs_driving_force_and_barrier():
    assert two_gate.quenches(-5.0, 3.0, 4.0)
    assert two_gate.quenches(0.0, 4.0, 4.0)
    assert not two_gate.quenches(-5.0, 5.0, 4.0)     # downhill, barrier too high
    assert not two_gate.quenches(0.5, 1.0, 4.0)      # low barrier, uphill
