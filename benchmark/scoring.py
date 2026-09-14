"""
Classification metrics, leave-one-photocatalyst-out selection, and rate residuals.

Not a script; imported by classification.py and rates.py.
"""

import numpy as np


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


def _ratio(a, b):
    return a / b if b else float("nan")


def _mcc(tp, fp, tn, fn):
    """MCC for scalars or arrays of counts; 0 where undefined."""
    tp, fp, tn, fn = (np.asarray(x, float) for x in (tp, fp, tn, fn))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (tp * tn - fp * fn) / denominator, 0.0)


def _counts(y, matrix):
    """Confusion counts per row of a (settings x pairs) prediction matrix."""
    return ((matrix & y).sum(axis=1), (matrix & ~y).sum(axis=1),
            (~matrix & ~y).sum(axis=1), (~matrix & y).sum(axis=1))


def best_setting(y, matrix) -> int:
    """Row index of the setting with the highest MCC; the first one on a tie."""
    y, matrix = np.asarray(y, bool), np.asarray(matrix, bool)
    return int(np.argmax(_mcc(*_counts(y, matrix))))


def leave_one_out(y, matrix, groups):
    """Held-out predictions: each group gets the setting that scores best on the others.

    Returns (predictions, number of distinct settings chosen). The held-out predictions
    are pooled into one confusion matrix by the caller.
    """
    y, matrix, groups = np.asarray(y, bool), np.asarray(matrix, bool), np.asarray(groups)
    total = np.array(_counts(y, matrix))
    held_out = np.zeros(len(y), bool)
    chosen = set()
    for group in np.unique(groups):
        inside = groups == group
        rest = total - np.array(_counts(y[inside], matrix[:, inside]))
        pick = int(np.argmax(_mcc(*rest)))
        chosen.add(pick)
        held_out[inside] = matrix[pick, inside]
    return held_out, len(chosen)


def log_residuals(k_predicted, k_measured) -> dict:
    """RMSE and median bias of log10(k_predicted / k_measured), in decades."""
    residual = np.log10(np.asarray(k_predicted, float)) - np.log10(np.asarray(k_measured, float))
    residual = residual[np.isfinite(residual)]
    return {
        "n": int(residual.size),
        "rmse / dex": float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan"),
        "median_bias / dex": float(np.median(residual)) if residual.size else float("nan"),
    }
