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

The out-of-sample split follows Project 1 of Leon Schlosser's thesis: every molecule that
contains a styryl motif (SMARTS `c1ccc(cc1)C=C`) is held out as the test set, and the
remaining molecules are split 90:10 into training and validation sets.
"""

import importlib.resources as ir
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DATA = Path(str(ir.files("ml.data")))
DATASET = PACKAGE_DATA / "training_dataset_full.csv"
SPLITS = PACKAGE_DATA / "splits"

RANDOM_SEEDS = (42, 67, 69, 404, 1337)
RANDOM_FRACTIONS = (0.8, 0.1, 0.1)

STYRYL_SMARTS = "c1ccc(cc1)C=C"
OOS_SEED = 42
OOS_TRAIN_FRACTION = 0.9


def random_split(n, seed, fractions=RANDOM_FRACTIONS):
    """Label n molecules train/val/test in the given proportions from a seeded permutation."""
    order = np.random.default_rng(seed).permutation(n)
    n_train = round(fractions[0] * n)
    n_val = round(fractions[1] * n)
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    return labels


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
    """Styryl molecules -> test; the rest split train/val from a seeded permutation."""
    styryl = has_styryl(smiles)
    rest = np.flatnonzero(~styryl)
    order = rest[np.random.default_rng(seed).permutation(len(rest))]
    n_train = round(train_fraction * len(rest))
    labels = np.full(len(smiles), "test", dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:]] = "val"
    return labels


def read_split(name):
    """Positional train, val and test indices of a split, e.g. `random_42` or `oos_styryl`."""
    labels = pd.read_csv(SPLITS / f"{name}.csv")["split"].to_numpy()
    return tuple(np.flatnonzero(labels == part) for part in ("train", "val", "test"))


def main():
    dataset = pd.read_csv(DATASET, usecols=["idx", "smiles"])
    SPLITS.mkdir(exist_ok=True)

    splits = {f"random_{seed}": random_split(len(dataset), seed) for seed in RANDOM_SEEDS}
    splits["oos_styryl"] = oos_split(dataset["smiles"].tolist())

    for name, labels in splits.items():
        out = SPLITS / f"{name}.csv"
        pd.DataFrame({"idx": dataset["idx"], "split": labels}).to_csv(
            out, index=False, lineterminator="\n")
        counts = pd.Series(labels).value_counts()
        print(f"{out.name}  "
              + "  ".join(f"{part} {counts.get(part, 0)}" for part in ("train", "val", "test")))


if __name__ == "__main__":
    main()
