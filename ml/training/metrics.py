"""Metrics for spin-population predictions: top-k overlap and ranking."""

import numpy as np


def top_k_jaccard_similarity(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """
    Jaccard index of the top-k atoms (highest values) of y_true and y_pred, order ignored.
    """
    if len(y_true) < k or len(y_pred) < k:
        k = min(len(y_true), len(y_pred))

    # Indices of the top-k true values (highest values first)
    # argsort sorts ascending, so take the last k
    top_k_true_indices = set(np.argsort(y_true)[-k:])

    # Indices of the top-k predicted values
    top_k_pred_indices = set(np.argsort(y_pred)[-k:])

    intersection = len(top_k_true_indices.intersection(top_k_pred_indices))
    union = len(top_k_true_indices.union(top_k_pred_indices))

    if union == 0:
        return 0.0

    return intersection / union


def highest_dens(a: np.ndarray, num_highest: int) -> np.ndarray:
    """
    Indices of the `num_highest` largest values, largest first; on ties the later index wins.
    """

    idx_dens = np.array([np.argsort(a)[-(i + 1)] for i in range(num_highest)])

    return idx_dens


def ranked_score(y_true: np.ndarray, y_pred: np.ndarray, num_highest=10) -> float:
    if len(y_true) < num_highest:
        num_highest = len(y_true)

    if len(y_pred) < num_highest:
        num_highest = len(y_pred)

    ha = highest_dens(y_true, num_highest=num_highest)
    hb = highest_dens(y_pred, num_highest=num_highest)

    take_into_account = int(num_highest / 2)

    error = 0

    for i, posi in enumerate(ha[:take_into_account]):
        if posi in hb:
            error += np.abs(list(hb).index(posi) - i)
        else:
            error += num_highest

    for i, posi in enumerate(hb[:take_into_account]):
        if posi in ha:
            error += np.abs(list(ha).index(posi) - i)
        else:
            error += num_highest

    return error / (2 * take_into_account)
