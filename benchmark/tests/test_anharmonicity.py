import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import anharmonicity  # noqa: E402
import classification  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402


def make_pairs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 4}" for i in range(n)],
        pairs.Q_NAME: [f"Q{i % 5 + 1}" for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.CONTACT: rng.uniform(7, 10, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.QUENCHES: rng.integers(0, 2, n),
    })
    for source in pairs.SOURCES:
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 80, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
        df[pairs.dg_ent(source)] = rng.uniform(-15, 10, n)
    return df


def make_samples(gaps_by_quencher, n_runs=10):
    rows = []
    for qid, gaps in gaps_by_quencher.items():
        for i, g in enumerate(gaps):
            rows.append({"quencher_id": qid, "md_run": i % n_runs + 1, "E_singlet_Eh": 0.0,
                         "E_triplet_Eh": g / anharmonicity.H_TO_KCAL})
    return pd.DataFrame(rows)


def test_gap_shape_flags_a_long_low_tail_and_is_deterministic(monkeypatch):
    monkeypatch.setattr(anharmonicity, "B_SKEW", 200)
    monkeypatch.setattr(anharmonicity, "N_IID", 2)
    rng = np.random.default_rng(1)
    samples = make_samples({1: 70 - rng.exponential(5, 600), 2: 70 + rng.normal(0, 5, 600)})
    a, b = anharmonicity.gap_shape(samples), anharmonicity.gap_shape(samples)
    pd.testing.assert_frame_equal(a, b)
    a = a.set_index("quencher")
    assert a.loc["Q1", "skewness_significant"] == "negative"
    assert a.loc["Q1", "tail_shift / kcal mol-1"] < 0
    assert a.loc["Q1", "mu / kcal mol-1"] == pytest.approx(samples[samples.quencher_id == 1]
                                                            .E_triplet_Eh.mean() * anharmonicity.H_TO_KCAL)


def test_crossing_point_gives_the_asymmetric_barrier():
    dG = np.array([-10.0, -3.0, 0.0, 2.0])
    lf, lr = np.array([60.0, 15.0, 50.0, 30.0]), np.array([20.0, 30.0, 25.0, 40.0])
    x = anharmonicity.crossing_point(dG, lf, lr)
    assert lf * x ** 2 == pytest.approx(theory.marcus_barrier_asymmetric(dG, lf, lr))


def test_torsional_curves_without_weight_reproduce_the_harmonic_barrier():
    dG, pls, plt_, qls, qlt = -4.0, 10.0, 8.0, 60.0, 20.0
    lf, lr = theory.pair_lambdas(pls, plt_, qls, qlt)
    b, _ = anharmonicity.barrier_torsional(dG, pls, plt_, qls, qlt, 0.0, 0.0)
    assert b == pytest.approx(float(theory.marcus_barrier_asymmetric(dG, lf, lr)), abs=1e-8)
    b_s0, _ = anharmonicity.barrier_torsional(dG, pls, plt_, qls, qlt, 1.0, 0.0)
    assert b_s0 > b                      # sin^2 lies above x^2 on (0, 1)


def test_unshifted_two_gate_is_the_benchmark_prediction():
    df = make_pairs()
    assert np.array_equal(anharmonicity.two_gate_shifted(df, 0.0),
                          classification.predict(df, anharmonicity.TWO_GATE, "dft")[0])
    # a large positive shift makes every pair endergonic
    assert not anharmonicity.two_gate_shifted(df, 100.0).any()


def test_signed_error_counts_misses_minus_false_alarms():
    df = pd.DataFrame({pairs.Q_NAME: ["Q1"] * 4, pairs.QUENCHES: [1, 1, 0, 0]})
    err = anharmonicity.signed_errors(df, np.array([False, True, True, True]))
    assert err.loc["Q1", "signed_error"] == pytest.approx((1 - 2) / 4)
