import numpy as np
import pytest
from ml.utilities.metrics.digitize import digitize_spin_populations
from ml.utilities.metrics.jaccard import top_k_jaccard_similarity
from ml.utilities.metrics.ranked_score import ranked_score, highest_dens

def test_top_k_jaccard_similarity_perfect():
    y_true = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    y_pred = np.array([0.15, 0.25, 0.95, 0.85, 0.35])
    # Top 2 true: indices 2 (0.9) and 3 (0.8)
    # Top 2 pred: indices 2 (0.95) and 3 (0.85)

    # Intersection = {2, 3}, size 2
    # Union = {2, 3}, size 2
    # Jaccard = 1.0

    score = top_k_jaccard_similarity(y_true, y_pred, k=2)
    assert score == 1.0

def test_top_k_jaccard_similarity_disjoint():
    y_true = np.array([0.1, 0.1, 0.9, 0.9])
    # Top 2 indices: 2, 3

    y_pred = np.array([0.9, 0.9, 0.1, 0.1])
    # Top 2 indices: 0, 1

    score = top_k_jaccard_similarity(y_true, y_pred, k=2)
    assert score == 0.0

def test_top_k_jaccard_similarity_partial():
    y_true = np.array([0.1, 0.1, 0.9, 0.9])
    # Top 2 indices: 2, 3

    y_pred = np.array([0.1, 0.9, 0.1, 0.9])
    # Top 2 indices: 1, 3 (values 0.9, 0.9)
    # Note: argsort stability might matter for ties.
    # y_pred argsort: [0, 2, 1, 3] or similar. Last 2: 1, 3.

    # Intersection: {3} -> size 1
    # Union: {1, 2, 3} -> size 3
    # Jaccard: 1/3

    score = top_k_jaccard_similarity(y_true, y_pred, k=2)
    assert abs(score - 1.0/3.0) < 1e-6

def test_highest_dens():
    a = np.array([1, 5, 2, 4, 3])
    # Sorted: 1, 2, 3, 4, 5
    # Indices: 0, 2, 4, 3, 1
    # Highest 3: 5 (idx 1), 4 (idx 3), 3 (idx 4)
    # HighestDens returns strictly indices of top-k values, descending order.
    # i=0: max (5) -> index 1
    # i=1: 2nd (4) -> index 3
    # i=2: 3rd (3) -> index 4

    res = highest_dens(a, num_highest=3)
    assert np.array_equal(res, np.array([1, 3, 4]))

def test_ranked_score_perfect():
    # If perfectly matching top-k order and indices
    y_true = np.array([1, 5, 2, 4, 3])
    y_pred = np.array([1, 5, 2, 4, 3])

    # num_highest=4. take_into_account=2
    # ha = [1, 3, 4, 2] (indices of 5, 4, 3, 2)
    # hb = [1, 3, 4, 2]

    # loop ha[:2] -> [1, 3]
    # i=0, posi=1. 1 in hb. index in hb is 0. abs(0 - 0) = 0.
    # i=1, posi=3. 3 in hb. index in hb is 1. abs(1 - 1) = 0.
    # error = 0

    # Same for loop hb[:2]
    # error = 0

    # score = 0 / 4 = 0

    score = ranked_score(y_true, y_pred, num_highest=4)
    assert score == 0.0

def test_ranked_score_bad():
    y_true = np.array([10, 9, 1, 2])
    # Top 2 (take 1): [0, 1]

    y_pred = np.array([1, 2, 10, 9])
    # Top 2 (take 1): [2, 3]

    # num_highest = 2. take_into_account = 1.
    # ha = [0, 1] (indices of 10, 9)
    # hb = [2, 3] (indices of 10, 9)

    # loop ha[:1] -> [0]
    # i=0, posi=0. 0 not in hb.
    # error += num_highest (2)

    # loop hb[:1] -> [2]
    # i=0, posi=2. 2 not in ha.
    # error += num_highest (2)

    # error = 4
    # score = 4 / (2 * 1) = 2.0

    score = ranked_score(y_true, y_pred, num_highest=2)
    assert score == 2.0

