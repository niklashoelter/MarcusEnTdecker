import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import metrics_extended as mx  # noqa: E402
import scoring  # noqa: E402


def criterion(matrix, score=None):
    m = np.asarray(matrix, bool)
    return mx.Criterion("k", "dft", mx.ROLE_SHOWN, m, m, np.zeros(m.shape[1]) if score is None else score,
                        tuple(str(i) for i in range(len(m))))


def test_average_precision_by_hand():
    y = np.array([1, 0, 1, 0], bool)
    # thresholds 0.9 (P 1, R 1/2), 0.8 (P 1/2), 0.7 (P 2/3, R 1), 0.1
    assert mx.average_precision(y, [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_average_precision_groups_tied_scores():
    y = np.array([1, 0, 1, 0], bool)
    assert mx.average_precision(y, [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)


def test_average_precision_equals_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.4
    score = np.round(rng.normal(size=200), 1)               # with ties
    assert mx.average_precision(y, score) == pytest.approx(metrics.average_precision_score(y, score))


def test_weighted_mcc_with_unit_weights_is_the_mcc():
    rng = np.random.default_rng(1)
    y = rng.random(50) < 0.5
    matrix = rng.random((4, 50)) < 0.5
    assert np.allclose(mx.weighted_mcc(matrix, y, np.ones(50)), scoring.mcc_rows(y, matrix))


def test_integer_weights_count_copies():
    y = np.array([1, 0, 1, 0], bool)
    p = np.array([1, 1, 0, 0], bool)
    w = np.array([2.0, 1.0, 1.0, 3.0])
    copies = np.repeat(np.arange(4), w.astype(int))
    assert mx.weighted_mcc(p, y, w)[0] == pytest.approx(scoring.metrics(y[copies], p[copies])["MCC"])


def test_choose_takes_the_middle_of_tied_settings():
    y = np.array([1, 0], bool)
    c = criterion([[0, 1], [1, 0], [1, 0], [1, 0], [0, 0]])
    assert c.choose(y, np.ones(2)) == 2


def test_held_out_chooses_without_the_group():
    # setting 0 fits group a, setting 1 group b: each group gets the setting of the other
    y = np.array([1, 1, 0, 1, 0], bool)
    groups = np.array(["a", "a", "a", "b", "b"])
    c = criterion([[1, 1, 0, 0, 1], [0, 0, 1, 1, 0]])
    predicted = mx.held_out(c, y, groups)
    assert list(predicted) == [False, False, True, False, True]


def test_pc_cuts_lie_between_and_outside_the_values():
    cuts = mx.pc_cuts(np.array([60.0, 62.0, 60.0, 65.0]))
    assert list(cuts) == [59.0, 61.0, 63.5, 66.0]


def test_within_pc_mcc_skips_groups_with_one_outcome():
    y = np.array([1, 0, 1, 1], bool)
    p = np.array([1, 0, 0, 1], bool)
    assert mx.within_pc_mcc(y, p, np.array(["a", "a", "b", "b"])) == pytest.approx(1.0)
