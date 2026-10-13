import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import bootstrap  # noqa: E402
import scoring  # noqa: E402


def data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.5
    score = np.round(rng.normal(size=n) + y, 1)
    groups = rng.integers(0, 5, n)
    return y, score, groups


def test_weighted_auc_with_unit_weights_is_the_auc():
    y, score, groups = data()
    w = np.ones(len(y))
    assert bootstrap.WeightedAUC(y, score)(w) == pytest.approx(scoring.auc(y, score))
    pooled = scoring.auc_within(y, score, groups)["pooled"]
    assert bootstrap.WeightedAUC(y, score, groups)(w) == pytest.approx(pooled)


def test_weighted_auc_with_integer_weights_counts_copies():
    y, score, groups = data(seed=1)
    w = np.random.default_rng(2).integers(0, 4, len(y)).astype(float)
    copies = np.repeat(np.arange(len(y)), w.astype(int))
    assert bootstrap.WeightedAUC(y, score)(w) == pytest.approx(scoring.auc(y[copies], score[copies]))
    expected = scoring.auc_within(y[copies], score[copies], groups[copies])["pooled"]
    assert bootstrap.WeightedAUC(y, score, groups)(w) == pytest.approx(expected)


def test_references():
    assert bootstrap.references(("adiabatic", "dft")) == [("dvte_2024", "dvte")]
    assert bootstrap.references(("twogate_lifetime", "dft")) == [
        ("dvte_2024", "dvte"), ("adiabatic", "dft"), ("twogate_lifetime", "ml")]
    assert bootstrap.references(("dvte_2024", "dvte")) == [("adiabatic", "dft"), ("adiabatic", "ml")]
    assert bootstrap.references(("pc_only", "-")) == [
        ("dvte_2024", "dvte"), ("adiabatic", "dft"), ("adiabatic", "ml")]
