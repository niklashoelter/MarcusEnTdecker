"""
Train/validation/test splits of the training dataset.

Reads   ml/data/training_dataset_full.csv
Writes  ml/data/splits/random_<seed>.csv   80:10:10 random split, one per seed
        ml/data/splits/oos_styryl.csv      out-of-sample split, styryl molecules as test set

    uv sync --extra ml
    uv run --extra ml python -m ml.training.splits

Each split file has one row per molecule, in the order of the dataset, with the columns
`idx` (the dataset's own id) and `split` (train, val or test). Row i of a split file is
row i of the dataset, so `read_split` returns positional indices that can be passed
straight to `create_dataloaders`.

Both procedures are Leon Schlosser's, taken from his training scripts:

- random: `train_test_split` takes 20 % out of the dataset with the seed, and a second
  `train_test_split` with the same seed halves that into validation and test
  (playground/model_setup/save_model_and_inspect/new_data/graph_gps/og_data_opt_param/train.py).
- out-of-sample: every molecule that contains a styryl motif (SMARTS `c1ccc(cc1)C=C`) is
  the test set; the others, in dataset order, are shuffled with Python's `random` seeded
  with 42, and the first 80 % train, the rest validate
  (playground/thesis/model/oos/train.py).
"""

import importlib.resources as ir
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PACKAGE_DATA = Path(str(ir.files("ml.data")))
DATASET = PACKAGE_DATA / "training_dataset_full.csv"
SPLITS = PACKAGE_DATA / "splits"

RANDOM_SEEDS = (42, 101, 143, 404, 666)

STYRYL_SMARTS = "c1ccc(cc1)C=C"
OOS_SEED = 42
OOS_TRAIN_FRACTION = 0.8


def random_split(n, seed):
    """Train, validation and test indices of n molecules, as Leon's training script draws them."""
    train_idx, temp_idx = train_test_split(range(n), test_size=0.2, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


def has_styryl(smiles):
    """True for each SMILES that contains the styryl motif."""
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(STYRYL_SMARTS)
    matches = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"unparsable SMILES: {s}")
        matches.append(mol.HasSubstructMatch(pattern))
    return np.array(matches)


def oos_split(smiles, seed=OOS_SEED, train_fraction=OOS_TRAIN_FRACTION):
    """Styryl molecules -> test; the rest shuffled and cut train/val, as in Leon's OOS script."""
    styryl = has_styryl(smiles)
    tmp_idx = np.flatnonzero(~styryl).tolist()
    test_idx = np.flatnonzero(styryl).tolist()
    random.Random(seed).shuffle(tmp_idx)
    n_train = int(train_fraction * len(tmp_idx))
    return tmp_idx[:n_train], tmp_idx[n_train:], test_idx


def labels(n, parts):
    """One train/val/test label per molecule from the three index lists."""
    out = np.empty(n, dtype=object)
    for name, idx in zip(("train", "val", "test"), parts):
        out[list(idx)] = name
    return out


def read_split(name):
    """Positional train, val and test indices of a split, e.g. `random_42` or `oos_styryl`."""
    split = pd.read_csv(SPLITS / f"{name}.csv")["split"].to_numpy()
    return tuple(np.flatnonzero(split == part) for part in ("train", "val", "test"))


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[1]).parse_args()
    dataset = pd.read_csv(DATASET, usecols=["idx", "smiles"])
    SPLITS.mkdir(exist_ok=True)

    n = len(dataset)
    splits = {f"random_{seed}": labels(n, random_split(n, seed)) for seed in RANDOM_SEEDS}
    splits["oos_styryl"] = labels(n, oos_split(dataset["smiles"].tolist()))

    for name, split in splits.items():
        out = SPLITS / f"{name}.csv"
        pd.DataFrame({"idx": dataset["idx"], "split": split}).to_csv(
            out, index=False, lineterminator="\n")
        counts = pd.Series(split).value_counts()
        print(f"{out.name}  "
              + "  ".join(f"{part} {counts.get(part, 0)}" for part in ("train", "val", "test")))


if __name__ == "__main__":
    main()
