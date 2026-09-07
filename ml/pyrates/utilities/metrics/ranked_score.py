import numpy as np


def highest_dens(a: np.ndarray, num_highest: int) -> np.ndarray:
    """
    Problem: If same values are given the latest one will be taken
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
