import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import mlj  # noqa: E402
import theory  # noqa: E402


def test_without_quantum_mode_the_symmetric_line_shape_is_the_marcus_barrier():
    dG, ls = np.array([-12.0, -3.0, 0.0, 4.0]), np.full(4, 20.0)
    ln_fc = mlj.log_fc(dG, ls, ls, np.zeros(4), np.full(4, 4.0), "symmetric")
    assert ln_fc == pytest.approx(-theory.marcus_barrier_symmetric(dG, ls) / theory.RT)


def test_without_quantum_mode_the_asymmetric_line_shape_is_the_two_gate_barrier():
    dG = np.array([-12.0, -3.0, 0.0, 4.0])
    lf, lr = np.full(4, 15.0), np.full(4, 40.0)
    ln_fc = mlj.log_fc(dG, lf, lr, np.zeros(4), np.full(4, 4.0), "asymmetric")
    assert ln_fc == pytest.approx(-theory.marcus_barrier_asymmetric(dG, lf, lr) / theory.RT)


def test_quantum_mode_raises_the_franck_condon_factor_in_the_inverted_region():
    dG, ls = np.array([-30.0]), np.array([5.0])
    classical = mlj.log_fc(dG, ls, ls, np.zeros(1), np.full(1, 4.0), "symmetric")
    quantum = mlj.log_fc(dG, ls, ls, np.full(1, 20.0), np.full(1, 4.0), "symmetric")
    assert quantum > classical


def test_unknown_line_shape_raises():
    with pytest.raises(ValueError):
        mlj.log_fc(np.zeros(1), np.ones(1), np.ones(1), np.zeros(1), np.ones(1), "parabolic")


def test_rate_with_the_classical_factor_is_the_rate_law_of_theory():
    dG, barrier = np.array([-6.0, 1.0]), np.array([4.0, 9.0])
    ratio = theory.escape_over_frequency(np.array([7.0, 9.0]))
    expected = np.log10(theory.rate(dG, barrier, ratio))
    assert mlj.log10_rate(dG, -barrier / theory.RT, ratio) == pytest.approx(expected)


def modes_table(rows):
    return pd.DataFrame(rows, columns=["kind", "id", "surface", "displacement", "lambda / kcal mol-1",
                                       "classical_fraction", "quantum_frequency / cm-1"])


def test_split_takes_internal_then_cartesian_then_the_median_of_the_reliable():
    modes = modes_table([
        ("quencher", 1, "T1", "internal", 10.0, 0.2, 1400.0),   # internal reliable
        ("quencher", 1, "T1", "cartesian", 10.0, 0.9, 900.0),
        ("quencher", 2, "T1", "internal", 30.0, 0.5, 1000.0),   # only cartesian reliable
        ("quencher", 2, "T1", "cartesian", 11.0, 0.4, 1600.0),
        ("quencher", 3, "T1", "internal", 30.0, 0.7, 500.0),    # neither
        ("quencher", 3, "T1", "cartesian", 1.0, 0.7, 500.0),
    ])
    four_point = pd.DataFrame({"kind": "quencher", "id": [1, 2, 3], "surface": "T1",
                               "lambda_four_point": 10.0})
    split, medians = mlj.molecule_split(modes, four_point)
    assert split.loc[("quencher", 1, "T1"), "classical_fraction"] == pytest.approx(0.2)
    assert split.loc[("quencher", 2, "T1"), "quantum_frequency"] == pytest.approx(1600.0)
    assert not split.loc[("quencher", 3, "T1"), "reliable"]
    assert split.loc[("quencher", 3, "T1"), "classical_fraction"] == pytest.approx(0.3)
    assert medians.loc[("quencher", "T1"), "quantum_frequency"] == pytest.approx(1500.0)


def test_bootstrap_difference_is_zero_against_itself():
    df = pd.DataFrame({"pc_name": ["PC1", "PC1", "PC2", "PC2"], "quencher_name": ["Q1", "Q2", "Q1", "Q2"],
                       "quenches": [True, False, True, False]})
    p = np.array([True, False, False, False])
    predicted = {("two_gate", "Eyring", "dft"): p, ("mlj_symmetric", "x", "dft"): p.copy()}
    out = mlj.bootstrap_dmcc(df, predicted)
    assert out[("mlj_symmetric", "x", "dft")] == {"dMCC": 0.0, "dMCC lo95": 0.0, "dMCC hi95": 0.0}
