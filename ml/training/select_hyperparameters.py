"""
Choose between the hyperparameter sets by validation loss.

Reads   ml/results/<set>/random_<seed>/model.pt          every set, all five random splits
Writes  ml/results/hyperparameter_selection_runs.csv     per set and split: best epoch and
                                                         the validation loss of that epoch
        ml/results/hyperparameter_selection.csv          per set: mean and SD of that loss
                                                         over the splits, median best epoch,
                                                         and which set is selected

    uv run --extra ml python -m ml.training.select_hyperparameters

The selected set has the lower mean validation loss, the objective the hyperparameter
search minimised (the weighted negative log-likelihood of the validation molecules, in
scaled units). The choice is made on validation data so that the test metrics in each set's
metrics_random_splits.csv stay an unbiased report for both. Written for this repository.
"""

import pandas as pd

from ml.training.hyperparameters import HYPERPARAMETERS
from ml.training.splits import RANDOM_SEEDS
from ml.training.test_predictions import RESULTS, check_set, run_dir

RUNS = RESULTS / "hyperparameter_selection_runs.csv"
SELECTION = RESULTS / "hyperparameter_selection.csv"


def runs():
    rows = []
    for name in HYPERPARAMETERS:
        for seed in RANDOM_SEEDS:
            split = f"random_{seed}"
            path = run_dir(name, split) / "model.pt"
            if not path.exists():
                raise SystemExit(f"{path} is missing; every set needs all five random splits")
            checkpoint = check_set(name, path)
            rows.append({"hyperparameters": name, "split": split,
                         "best_epoch": int(checkpoint["epoch"]),
                         "val_loss": float(checkpoint["val_loss"])})
    return pd.DataFrame(rows)


def selection(table):
    summary = table.groupby("hyperparameters", sort=False).agg(
        n_splits=("split", "size"),
        val_loss_mean=("val_loss", "mean"),
        val_loss_sd=("val_loss", lambda v: v.to_numpy().std()),
        best_epoch_median=("best_epoch", "median")).reset_index()
    summary["selected"] = summary["val_loss_mean"] == summary["val_loss_mean"].min()
    if summary["selected"].sum() != 1:
        raise SystemExit("the sets tie on mean validation loss; no set is selected")
    return summary


def selected_set():
    """The name of the selected set, from hyperparameter_selection.csv."""
    if not SELECTION.exists():
        raise FileNotFoundError(f"{SELECTION} does not exist; run select_hyperparameters first")
    table = pd.read_csv(SELECTION)
    return str(table.loc[table["selected"], "hyperparameters"].item())


def main():
    table = runs()
    summary = selection(table)
    table.round({"val_loss": 4}).to_csv(RUNS, index=False, lineterminator="\n")
    summary.round({"val_loss_mean": 4, "val_loss_sd": 4}).to_csv(
        SELECTION, index=False, lineterminator="\n")
    print(summary.to_string(index=False))
    print(f"selected: {selected_set()}")


if __name__ == "__main__":
    main()
