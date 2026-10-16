import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import error_propagation  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402


def make_pairs(n=24, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 3}" for i in range(n)],
        pairs.Q_ID: [i % 6 + 1 for i in range(n)],
        pairs.Q_NAME: [f"Q{i % 6 + 1}" for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.K_Q: 10 ** rng.uniform(7, 10, n),
        pairs.CONTACT: rng.uniform(7, 10, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.QUENCHES: np.arange(n) % 2,
    })
    for target in error_propagation.ML_TARGETS:
        uncertainty = dict(zip(range(1, 7), rng.uniform(0.5, 2.0, 6)))
        df[pairs.q_ml_uncertainty(target)] = df[pairs.Q_ID].map(uncertainty)
    for source in pairs.SOURCES:
        df[f"q_E_T_{source} / kcal mol-1"] = 60.0 + df[pairs.Q_ID] * (1.0 if source == "dft" else 1.1)
        df[pairs.q_lambda_s0_t1(source)] = 10.0 + df[pairs.Q_ID]
        df[pairs.q_lambda_t1_s0(source)] = 12.0 + df[pairs.Q_ID]
        df[pairs.dg_ent(source)] = rng.uniform(-10, 5, n)
    return df


@pytest.mark.parametrize("source", pairs.SOURCES)
def test_unperturbed_classes_and_rates_are_those_of_the_benchmark(source):
    df = make_pairs()
    data = error_propagation.Pairs(df)
    log_k, two_gate, adiabatic = data.unperturbed[source]
    setting = classification.candidates("two_gate", "r from DFT sizes")[0]
    np.testing.assert_array_equal(two_gate, classification.predict(df, setting, source)[0])
    np.testing.assert_array_equal(adiabatic, pairs.driving_force(df, source) <= 0)
    dG = pairs.driving_force(df, source)
    barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(df, source)))
    expected = theory.rate(dG, barrier, theory.escape_over_frequency(pairs.contact_distances(df)))
    np.testing.assert_allclose(log_k, np.log10(expected))


def test_an_energy_error_is_shared_by_the_pairs_of_one_quencher():
    df = make_pairs()
    data = error_propagation.Pairs(df)
    p = error_propagation.draw(np.random.default_rng(0), data, "Q E_T", "dft", 2.0, 5)
    error = p["dG"] - data.base["dft"]["dG"]
    for q in np.unique(data.q_index):
        same = error[:, data.q_index == q]
        np.testing.assert_allclose(same, np.repeat(same[:, :1], same.shape[1], axis=1))
    assert np.ptp(error[0]) > 0


def test_perturbed_reorganization_energies_keep_the_floor():
    data = error_propagation.Pairs(make_pairs())
    p = error_propagation.draw(np.random.default_rng(1), data, "lambda multiplicative", "dft", 3.0, 20)
    for key in ("pc_s0_t1", "pc_t1_s0", "q_s0_t1", "q_t1_s0"):
        assert (p[key] >= error_propagation.LAMBDA_MIN * data.base["dft"][key] - 1e-12).all()


def test_monte_carlo_is_deterministic(monkeypatch):
    monkeypatch.setattr(error_propagation, "N_DRAWS", 20)
    data = error_propagation.Pairs(make_pairs())
    pd.testing.assert_frame_equal(error_propagation.monte_carlo(data), error_propagation.monte_carlo(data))


def test_sensitivity_to_the_driving_force_is_bounded_by_one_over_rt_ln10():
    data = error_propagation.Pairs(make_pairs())
    table = error_propagation.sensitivity(data)
    bound = 1.0 / (theory.RT * np.log(10.0))
    assert (table["max |dlog k / dE_T(Q)|"] <= bound * 1.01).all()
    assert table["kcal per decade (RT ln10)"].iloc[0] == pytest.approx(1.364, abs=1e-3)
