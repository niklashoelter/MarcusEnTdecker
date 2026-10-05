"""
Classification metrics and the choice of the best setting in a sweep.

Not a script; imported by the benchmark and further_analysis scripts.
"""

import numpy as np
from scipy.stats import rankdata


def rounded(table):
    """A result table as written: every float to three decimals, from full precision.

    The metrics are unitless; the settings and energies in these tables carry no more
    than three decimals to begin with, so rounding leaves them as they are.
    """
    return table.round(3)


def metrics(y, predicted) -> dict:
    """Confusion counts and the metrics built from them. Recall is sensitivity."""
    y = np.asarray(y, bool)
    p = np.asarray(predicted, bool)
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    tn, fn = int((~p & ~y).sum()), int((~p & y).sum())
    n = tp + fp + tn + fn
    return {
        "n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": _ratio(tp + tn, n),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "F1": _ratio(2 * tp, 2 * tp + fp + fn),
        "MCC": float(_mcc(tp, fp, tn, fn)),
    }


def auc(y, score) -> float:
    """ROC AUC: the probability that a random positive scores above a random negative.

    Ties count one half. Higher score means quenching is more likely.
    """
    y = np.asarray(y, bool)
    ranks = rankdata(np.asarray(score, float))
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mcc_rows(y, matrix):
    """MCC of every row of a (settings x pairs) prediction matrix."""
    y, matrix = np.asarray(y, bool), np.asarray(matrix, bool)
    counts = ((matrix & y).sum(axis=1), (matrix & ~y).sum(axis=1),
              (~matrix & ~y).sum(axis=1), (~matrix & y).sum(axis=1))
    return _mcc(*counts)


def best_setting(y, matrix) -> int:
    """Row with the highest MCC; among tied rows, the middle one in row order.

    With an even number of tied rows the lower middle is taken.
    """
    return best_setting_from_scores(mcc_rows(y, matrix))


def best_setting_from_scores(scores) -> int:
    """Index of the highest score; among tied indices, the (lower) middle one."""
    scores = np.asarray(scores, float)
    tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    return int(tied[(len(tied) - 1) // 2])


def _ratio(a, b):
    return a / b if b else float("nan")


def _mcc(tp, fp, tn, fn):
    """MCC for scalars or arrays of counts; 0 where undefined."""
    tp, fp, tn, fn = (np.asarray(x, float) for x in (tp, fp, tn, fn))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (tp * tn - fp * fn) / denominator, 0.0)
