import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import ml_vs_dft  # noqa: E402
import pairs  # noqa: E402


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
        pairs.QUENCHES: rng.integers(0, 2, n),
    })
    for source in pairs.SOURCES:
        df[pairs.dg_ent(source)] = rng.uniform(-15, 10, n)
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 80, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
    return df


def test_with_dft_swaps_only_the_named_parameters_and_rows():
    df = make_pairs()
    rows = (df[pairs.Q_NAME] == "Q2").to_numpy()
    d = ml_vs_dft.with_dft(df, ("E_T",), rows)
    assert np.array_equal(d.loc[rows, pairs.dg_ent("ml")], df.loc[rows, pairs.dg_ent("dft")])
    assert np.array_equal(d.loc[~rows, pairs.dg_ent("ml")], df.loc[~rows, pairs.dg_ent("ml")])
    assert np.array_equal(d[pairs.q_lambda_s0_t1("ml")], df[pairs.q_lambda_s0_t1("ml")])


def test_all_parameters_from_dft_give_the_dft_prediction():
    df = make_pairs()
    p, s = ml_vs_dft.two_gate(ml_vs_dft.with_dft(df, ml_vs_dft.PARAMETERS))
    p_ref, s_ref = classification.predict(df, ml_vs_dft.TWO_GATE, "dft")
    assert np.array_equal(p, p_ref) and np.allclose(s, s_ref)
