import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import kdet_threshold  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402


def make_pairs(n, seed=0):
    """Synthetic pairs with the inputs of the two-gate criterion."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({pairs.TAU0: rng.uniform(100, 5000, n), pairs.CONTACT: rng.uniform(6, 11, n),
                       pairs.PC_E_T: rng.uniform(55, 70, n), pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
                       pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n)})
    for source in pairs.SOURCES:
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 40, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
        df[pairs.dg_ent(source)] = rng.uniform(-15, 10, n)
    return df


def test_global_floor_reproduces_the_benchmark_two_gate():
    df = make_pairs(40)
    setting = classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                     k_sv_crit=classification.K_SV_CRIT, contact_distance=None)
    for source in pairs.SOURCES:
        predicted, score, _ = kdet_threshold.two_gate(df, np.full(len(df), classification.K_SV_CRIT), source)
        reference, reference_score = classification.predict(df, setting, source)
        assert (predicted == reference).all()
        assert score == pytest.approx(reference_score)


def test_higher_floor_lowers_the_threshold_by_rt_ln_ratio():
    df = make_pairs(10)
    _, _, t10 = kdet_threshold.two_gate(df, np.full(len(df), 10.0), "dft")
    _, _, t40 = kdet_threshold.two_gate(df, np.full(len(df), 40.0), "dft")
    # exact form: RT ln((k_d tau0 / K - 1)); close to RT ln 4 when k_d tau0 >> K
    expected = theory.RT * np.log((theory.K_DIFF * pairs.tau0_seconds(df) / 10.0 - 1)
                                  / (theory.K_DIFF * pairs.tau0_seconds(df) / 40.0 - 1))
    assert t10 - t40 == pytest.approx(expected)


def test_floor_above_the_diffusion_limit_rejects_every_pair():
    df = make_pairs(10)
    huge = theory.K_DIFF * pairs.tau0_seconds(df) * 2
    predicted, score, t_pc = kdet_threshold.two_gate(df, huge, "dft")
    assert not predicted.any()
    assert np.isneginf(t_pc).all()
    assert np.isfinite(score).all()
