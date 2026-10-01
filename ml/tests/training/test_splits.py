import numpy as np
import pandas as pd
import pytest

from ml.training import splits


@pytest.mark.parametrize("seed", splits.RANDOM_SEEDS)
def test_committed_random_split_is_the_procedure(seed):
    n = len(pd.read_csv(splits.DATASET, usecols=["idx"]))
    expected = splits.labels(n, splits.random_split(n, seed))
    committed = pd.read_csv(splits.SPLITS / f"random_{seed}.csv")["split"].to_numpy()
    assert (committed == expected).all()


def test_random_split_is_80_10_10_and_disjoint():
    train, val, test = splits.random_split(1000, 42)
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    assert len(set(train) | set(val) | set(test)) == 1000


def test_oos_split_holds_out_every_styryl_molecule():
    smiles = ["C=Cc1ccccc1", "CC", "CCC", "c1ccccc1/C=C/C", "CCO", "CCN", "CCCl", "O"] * 5
    train, val, test = splits.oos_split(smiles)
    styryl = {i for i, s in enumerate(smiles) if "C=C" in s and "c1ccccc1" in s}
    assert set(test) == styryl
    assert len(train) == int(0.8 * (len(smiles) - len(styryl)))
    assert sorted(train + val + test) == list(range(len(smiles)))
    assert splits.oos_split(smiles) == (train, val, test)


def test_read_split_returns_positional_indices():
    train, val, test = splits.read_split("random_42")
    assert len(np.intersect1d(train, test)) == 0
    assert (np.diff(train) > 0).all()
