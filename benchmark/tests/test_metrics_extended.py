import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import metrics_extended as mx  # noqa: E402


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


def test_within_pc_mcc_skips_groups_with_one_outcome():
    y = np.array([1, 0, 1, 1], bool)
    p = np.array([1, 0, 0, 1], bool)
    assert mx.within_pc_mcc(y, p, np.array(["a", "a", "b", "b"])) == pytest.approx(1.0)
