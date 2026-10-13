import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import gate_offset  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402


def make_pairs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 3}" for i in range(n)],
        pairs.Q_NAME: [f"Q{i % 5}" for i in range(n)],
        pairs.K_SV: rng.uniform(-5, 200, n),
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.CONTACT: rng.uniform(7, 10, n),
        pairs.PC_E_T: rng.uniform(55, 70, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.QUENCHES: rng.integers(0, 2, n),
        pairs.q_dvte(*gate_offset.DVTE): rng.uniform(55, 75, n),
    })
    for source in pairs.SOURCES:
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 40, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
        df[pairs.dg_ent(source)] = rng.uniform(-6, 4, n)
    return df


def benchmark_setting(criterion):
    if criterion == "adiabatic":
        return classification.Setting("adiabatic")
    if criterion == "dvte":
        p, kind = gate_offset.DVTE
        return classification.Setting("dvte", p=p, dvte_kind=kind)
    return classification.candidates("two_gate", "r from DFT sizes")[0]


@pytest.mark.parametrize("key", gate_offset.CRITERIA)
def test_c_zero_is_the_benchmark_criterion(key):
    df = make_pairs()
    o = gate_offset.Offsets(df)
    predicted, score = o.predict(*key, 0.0)
    expected, expected_score = classification.predict(df, benchmark_setting(key[0]), key[1])
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_allclose(score, expected_score)
    np.testing.assert_array_equal(o.grid[key][gate_offset.index_of(0.0)], expected)


def test_a_stricter_gate_only_removes_predicted_pairs():
    o = gate_offset.Offsets(make_pairs())
    strict, _ = o.predict("two_gate", "dft", -1.5)
    loose, _ = o.predict("two_gate", "dft", 0.0)
    assert not (strict & ~loose).any()


def test_two_gate_moves_the_gate_but_not_the_barrier():
    df = make_pairs()
    o = gate_offset.Offsets(df)
    predicted, _ = o.predict("two_gate", "dft", -1.0)
    margin = o.t_pc - o.barrier["dft"]
    np.testing.assert_array_equal(predicted, (o.dG["dft"] <= -1.0) & (margin >= 0))


def test_weighted_mcc_with_zero_weights_equals_the_mcc_of_the_rest():
    o = gate_offset.Offsets(make_pairs())
    matrix = o.grid[("adiabatic", "dft")]
    keep = o.df[pairs.PC_NAME].to_numpy() != "PC1"
    weighted = gate_offset.mcc_weighted(o.y, matrix, keep.astype(float))
    np.testing.assert_allclose(weighted, scoring.mcc_rows(o.y[keep], matrix[:, keep].astype(bool)))


def test_held_out_chooses_c_without_the_held_out_group():
    o = gate_offset.Offsets(make_pairs())
    key = ("adiabatic", "dft")
    groups = o.df[pairs.PC_NAME].to_numpy()
    predicted, _, chosen = gate_offset.held_out_predictions(o, key, groups)
    for group, c in zip(pd.unique(groups), chosen):
        rest = groups != group
        best = scoring.best_setting(o.y[rest], o.grid[key][:, rest].astype(bool))
        assert c == gate_offset.C_GRID[best]
        np.testing.assert_array_equal(predicted[~rest], o.predict(*key, c)[0][~rest])
