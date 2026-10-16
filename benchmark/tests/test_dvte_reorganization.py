import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import dvte_reorganization as dr  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402


def make_pairs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 5 + 1}" for i in range(n)],
        pairs.Q_NAME: [f"Q{i % 6 + 1}" for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.CONTACT: rng.uniform(7, 10, n),
        pairs.PC_E_T: rng.uniform(50, 70, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.q_dvte("0.008", "normal"): rng.uniform(45, 75, n),
        pairs.QUENCHES: rng.integers(0, 2, n),
    })
    for source in pairs.SOURCES:
        df[f"q_E_T_{source} / kcal mol-1"] = df[pairs.PC_E_T] + rng.uniform(-15, 10, n)
        df[pairs.dg_ent(source)] = df[f"q_E_T_{source} / kcal mol-1"] - df[pairs.PC_E_T]
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 80, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
    return df


def test_effective_temperature_limits():
    assert dr.kT_eff(None) == pytest.approx(theory.RT)
    assert dr.kT_eff(1) == pytest.approx(theory.RT, rel=1e-5)          # classical limit
    assert dr.kT_eff(1000) / theory.RT == pytest.approx(2.452, abs=1e-3)
    hw = 5000 * dr.CM_TO_KCAL
    assert dr.kT_eff(5000) == pytest.approx(hw / 2, rel=1e-3)           # zero-point limit


def test_gaussian_estimate_quantile():
    mu, sd, q = dr.gaussian_estimate(60.0, 20.0, None)
    assert mu == 80.0
    assert sd == pytest.approx(np.sqrt(2 * 20.0 * theory.RT))
    assert q == pytest.approx(mu + stats.norm.ppf(0.008) * sd)


def test_youden_threshold_separates_and_takes_the_middle_tie():
    dG = np.array([-6.0, -4.0, -2.0, 1.0, 3.0])
    y = np.array([1, 1, 1, 0, 0], bool)
    assert dr.youden(dG, y) == pytest.approx(-0.5)
    # all thresholds between -4 and 1 are equally good here; the middle one is taken
    y2 = np.array([1, 1, 0, 1, 0], bool)
    assert dr.youden(dG, y2) in (-3.0, -0.5)


def test_two_gate_matches_the_benchmark_and_the_gate_free_variant_admits_more():
    df = make_pairs()
    for source in pairs.SOURCES:
        p, s, _, _ = dr.two_gate(df, source)
        p_ref, s_ref = classification.predict(df, dr.TWO_GATE, source)
        assert np.array_equal(p, p_ref) and np.allclose(s, s_ref)
        free = dr.two_gate(df, source, energy_gate=False)[0]
        assert (free | ~p).all()
