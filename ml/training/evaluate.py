"""
Metrics of the validation runs, from their test predictions.

Reads   ml/results/<split>/test_predictions_{graph,node}.csv   for every split that has them
Writes  ml/results/metrics.csv                  MAE, RMSE, R² per split and target
        ml/results/metrics_random_splits.csv    mean and population SD over the random splits

    uv run --extra ml python -m ml.training.evaluate

Spin population: pooled over all test atoms. r2_abs_error_vs_uncertainty: R² of |error|
against the predicted uncertainty. Rounding: kcal/mol 2 decimals, unitless 3.

Adapted from Leon Schlosser's evaluation scripts, without their plotting:
playground/model_setup/save_model_and_inspect/new_data/graph_gps/og_data_opt_param/eval.py
and playground/thesis/model/gps/eval_fivefold.py.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ml.training.splits import RANDOM_SEEDS
from ml.training.test_predictions import ALL_TARGETS, NODE_TARGETS, RESULTS, run_dir

RANDOM_SPLITS = [f"random_{seed}" for seed in RANDOM_SEEDS]
SPLITS = RANDOM_SPLITS + ["oos_styryl"]


def decimals(target):
    return 3 if target in NODE_TARGETS else 2


def predictions(split, target):
    """Targets, predictions and uncertainties of one target on the test set of a split."""
    kind = "node" if target in NODE_TARGETS else "graph"
    path = run_dir(split) / f"test_predictions_{kind}.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    return (data[f"{target}_target"].to_numpy(), data[f"{target}_pred"].to_numpy(),
            data[f"{target}_uncertainty"].to_numpy())


def split_metrics(splits):
    rows = []
    for split in splits:
        for target in ALL_TARGETS:
            found = predictions(split, target)
            if found is None:
                continue
            y, yhat, unc = found
            rows.append({"split": split, "target": target, "n": len(y),
                         "mae": mean_absolute_error(y, yhat),
                         "rmse": root_mean_squared_error(y, yhat),
                         "r2": r2_score(y, yhat),
                         "r2_abs_error_vs_uncertainty": r2_score(np.abs(y - yhat), unc),
                         "median_uncertainty": np.median(unc)})
    return pd.DataFrame(rows)


def summary(metrics):
    rows = []
    for target, group in metrics[metrics["split"].isin(RANDOM_SPLITS)].groupby("target", sort=False):
        row = {"target": target, "n_splits": len(group)}
        for col in ("mae", "rmse", "r2"):
            row[f"{col}_mean"] = group[col].to_numpy().mean()
            row[f"{col}_sd"] = group[col].to_numpy().std()
        rows.append(row)
    return pd.DataFrame(rows)


def rounded(table):
    """Two decimals for kcal/mol, three for unitless values; counts untouched."""
    table = table.copy()
    per_target = [c for c in table.columns if c.startswith(("mae", "rmse"))
                  or c == "median_uncertainty"]
    for target in table["target"].unique():
        rows = table["target"] == target
        table.loc[rows, per_target] = table.loc[rows, per_target].round(decimals(target))
    unitless = [c for c in table.columns if c.startswith("r2")]
    table[unitless] = table[unitless].round(3)
    return table


def evaluate():
    """Write the two metric tables; needs all random splits."""
    metrics = split_metrics(SPLITS)
    found = set(metrics["split"]) if not metrics.empty else set()
    missing_random = [s for s in RANDOM_SPLITS if s not in found]
    if missing_random:
        raise SystemExit("no test predictions for " + ", ".join(missing_random))
    missing = [s for s in SPLITS if s not in found]
    if missing:
        print("no test predictions for " + ", ".join(missing))

    outputs = {"metrics.csv": metrics,
               "metrics_random_splits.csv": summary(metrics)}
    for file, table in outputs.items():
        rounded(table).to_csv(RESULTS / file, index=False, lineterminator="\n")
        print(f"wrote {file} ({len(table)} rows)")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[1]).parse_args()
    evaluate()


if __name__ == "__main__":
    main()
