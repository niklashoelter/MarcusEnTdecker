import numpy as np

def top_k_jaccard_similarity(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """
    Calculates the Jaccard coefficient for the top-k atoms with the highest density.
    A value of 1.0 means perfect agreement of the top-k atoms (regardless of order).
    A value of 0.0 means no overlap.
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
