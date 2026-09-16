import numpy as np
import pytest

import scoring


def test_metrics_from_confusion_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0], bool)
    p = np.array([1, 1, 0, 1, 0, 0, 0], bool)       # TP 2, FN 1, FP 1, TN 3
    m = scoring.metrics(y, p)
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (2, 1, 3, 1)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["F1"] == pytest.approx(4 / 6)
    assert m["MCC"] == pytest.approx((2 * 3 - 1 * 1) / np.sqrt(3 * 3 * 4 * 4))


def test_mcc_is_zero_when_one_prediction_class_is_empty():
    y = np.array([1, 0, 1, 0], bool)
    assert scoring.metrics(y, np.ones(4, bool))["MCC"] == 0.0


def test_auc_perfect_reversed_and_tied():
    y = np.array([0, 0, 1, 1], bool)
    assert scoring.auc(y, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert scoring.auc(y, [0.4, 0.3, 0.2, 0.1]) == pytest.approx(0.0)
    assert scoring.auc(y, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_auc_counts_ties_as_half():
    y = np.array([0, 1], bool)
    assert scoring.auc(y, [2.0, 2.0]) == pytest.approx(0.5)
    y = np.array([0, 0, 1], bool)
    assert scoring.auc(y, [1.0, 2.0, 2.0]) == pytest.approx(0.75)


def test_mcc_rows_scores_each_setting():
    y = np.array([1, 0], bool)
    matrix = np.array([[1, 0], [0, 1], [1, 1]], bool)
    assert scoring.mcc_rows(y, matrix) == pytest.approx([1.0, -1.0, 0.0])


def test_best_setting_takes_the_middle_of_tied_settings():
    y = np.array([1, 1, 0, 0], bool)
    perfect, wrong = [1, 1, 0, 0], [0, 0, 1, 1]
    matrix = np.array([wrong, perfect, perfect, perfect, wrong], bool)
    assert scoring.best_setting(y, matrix) == 2


def test_best_setting_takes_the_lower_middle_of_an_even_tie():
    y = np.array([1, 0], bool)
    matrix = np.array([[1, 0], [1, 0], [0, 1]], bool)
    assert scoring.best_setting(y, matrix) == 0


def test_best_setting_from_scores_takes_the_middle_of_tied_maxima():
    assert scoring.best_setting_from_scores([0.1, 0.5, 0.5, 0.5, 0.2]) == 2
    assert scoring.best_setting_from_scores([0.5, 0.5, 0.1]) == 0
