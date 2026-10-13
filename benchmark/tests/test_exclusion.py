import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import exclusion  # noqa: E402
import pairs  # noqa: E402


def make_pairs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 5}" for i in range(n)],
        pairs.Q_NAME: [["Q1", "Q2", "Q31", "Q38"][i % 4] for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.CONTACT: rng.uniform(7, 10, n),
        pairs.PC_E_T: rng.uniform(50, 70, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.q_dvte("0.008", "normal"): rng.uniform(45, 75, n),
        pairs.QUENCHES: rng.integers(0, 2, n),
    })
    for source in pairs.SOURCES:
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 80, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
        df[pairs.dg_ent(source)] = rng.uniform(-15, 10, n)
    return df


def test_left_out_quenchers_are_restored_after_loading():
    saved = dict(pairs.LEFT_OUT_QUENCHERS)
    with pytest.raises(RuntimeError):
        with exclusion.left_out_quenchers_included():
            assert pairs.LEFT_OUT_QUENCHERS == {}
            raise RuntimeError
    assert pairs.LEFT_OUT_QUENCHERS == saved


def test_emission_ratio_uses_the_highest_concentration():
    t = pd.DataFrame({"pc_old_id": 1, "quencher_id": 2, "quencher_conc_M": [0, 0, 0.002, 0.004, 0.004],
                      "intensity_au": [100.0, 110.0, 90.0, 50.0, 60.0]})
    r = exclusion.emission_ratios(t).iloc[0]
    assert r["conc_max / mM"] == pytest.approx(4.0)
    assert r["I0_over_I_at_max"] == pytest.approx(105.0 / 55.0)


def test_scenarios_drop_the_named_quenchers_only():
    df = make_pairs()
    s = exclusion.scenarios(df).set_index(["scenario", "criterion"])
    n = df[pairs.Q_NAME].value_counts()
    assert s.loc[("all pairs", "dvte"), "n"] == len(df)
    assert s.loc[("without Q38", "dvte"), "n"] == len(df) - n["Q38"]
    assert s.loc[("without Q38 and Q31", "two_gate, dft"), "n"] == len(df) - n["Q38"] - n["Q31"]
    assert s.loc[("without Q31", "adiabatic, ml"), "n_quench"] == int(df.loc[df[pairs.Q_NAME] != "Q31",
                                                                              pairs.QUENCHES].sum())
