"""
Metrics of the validation runs, from their test predictions.

Reads   ml/results/<split>/test_predictions_{graph,node}.csv   for every split that has them
        ml/results/random_<seed>/model.pt                     for their md5 sums
Writes  ml/results/metrics.csv                  MAE, RMSE, R² per split and target
        ml/results/metrics_random_splits.csv    mean and population SD over the random splits
        ml/results/uncertainty_filtering.csv    metrics on the test predictions with
                                                uncertainty <= threshold, mean and SD over
                                                the random splits
        ml/results/ensemble_members.csv         random-split checkpoints with md5 sums
                                                (members of ml.ensemble)

    uv run --extra ml python -m ml.training.evaluate

Spin population: pooled over all test atoms. r2_abs_error_vs_uncertainty: R² of |error|
against the predicted uncertainty. Thresholds: 20 evenly spaced from the 60 % quantile to the
maximum of the uncertainties pooled over the random splits; < 2 kept predictions give NaN.
Rounding: kcal/mol 2 decimals, unitless 3.

Adapted from Leon Schlosser's evaluation scripts, without their plotting:
playground/model_setup/save_model_and_inspect/new_data/graph_gps/og_data_opt_param/eval.py,
playground/thesis/model/gps/eval_fivefold.py and
playground/thesis/model/gps/uncertainty_curve.py.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ml.training.splits import RANDOM_SEEDS
from ml.training.test_predictions import ALL_TARGETS, NODE_TARGETS, RESULTS, run_dir

RANDOM_SPLITS = [f"random_{seed}" for seed in RANDOM_SEEDS]
SPLITS = RANDOM_SPLITS + ["oos_styryl"]

N_THRESHOLD_STEPS = 20
THRESHOLD_LOW_QUANTILE = 0.6


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


def metrics_at_threshold(y_true, y_pred, unc, threshold):
    mask = unc <= threshold
    n_keep = int(mask.sum())
    if n_keep < 2:
        return np.nan, np.nan, np.nan, n_keep
    yt, yp = y_true[mask], y_pred[mask]
    return (mean_absolute_error(yt, yp), root_mean_squared_error(yt, yp), r2_score(yt, yp),
            n_keep)


def uncertainty_filtering(splits):
    rows = []
    for target in ALL_TARGETS:
        runs = [found for split in splits
                if (found := predictions(split, target)) is not None]
        if not runs:
            continue
        pooled = np.concatenate([unc for _, _, unc in runs])
        thresholds = np.linspace(np.quantile(pooled, THRESHOLD_LOW_QUANTILE), pooled.max(),
                                 N_THRESHOLD_STEPS)
        for threshold in thresholds:
            arr = np.array([metrics_at_threshold(y, yhat, unc, threshold) + (len(unc),)
                            for y, yhat, unc in runs])
            mean, sd = arr.mean(axis=0), arr.std(axis=0)
            rows.append({"target": target, "n_splits": len(runs), "threshold": threshold,
                         "n_kept": int(round(mean[3])), "data_used_percent": 100 * mean[3] / mean[4],
                         "mae_mean": mean[0], "mae_sd": sd[0],
                         "rmse_mean": mean[1], "rmse_sd": sd[1],
                         "r2_mean": mean[2], "r2_sd": sd[2]})
    return pd.DataFrame(rows)


def rounded(table):
    """Two decimals for kcal/mol, three for unitless values; counts untouched."""
    table = table.copy()
    per_target = [c for c in table.columns if c.startswith(("mae", "rmse"))
                  or c in ("threshold", "median_uncertainty")]
    for target in table["target"].unique():
        rows = table["target"] == target
        table.loc[rows, per_target] = table.loc[rows, per_target].round(decimals(target))
    unitless = [c for c in table.columns if c.startswith("r2")]
    table[unitless] = table[unitless].round(3)
    if "data_used_percent" in table:
        table["data_used_percent"] = table["data_used_percent"].round(1)
    return table


def evaluate():
    """Write the three metric tables and the ensemble members; needs all random splits."""
    metrics = split_metrics(SPLITS)
    found = set(metrics["split"]) if not metrics.empty else set()
    missing_random = [s for s in RANDOM_SPLITS if s not in found]
    if missing_random:
        raise SystemExit("no test predictions for " + ", ".join(missing_random))
    missing = [s for s in SPLITS if s not in found]
    if missing:
        print("no test predictions for " + ", ".join(missing))

    outputs = {"metrics.csv": metrics,
               "metrics_random_splits.csv": summary(metrics),
               "uncertainty_filtering.csv": uncertainty_filtering(RANDOM_SPLITS)}
    for file, table in outputs.items():
        rounded(table).to_csv(RESULTS / file, index=False, lineterminator="\n")
        print(f"wrote {file} ({len(table)} rows)")
    write_ensemble_members()


def write_ensemble_members():
    """Record the five random-split checkpoints as the members of ml.ensemble."""
    from ml.ensemble import MEMBERS, md5

    root = RESULTS.parent.parent
    rows = [{"split": split,
             "checkpoint": (run_dir(split) / "model.pt").relative_to(root).as_posix(),
             "md5": md5(run_dir(split) / "model.pt")} for split in RANDOM_SPLITS]
    pd.DataFrame(rows).to_csv(MEMBERS, index=False, lineterminator="\n")
    print(f"wrote {MEMBERS.name} ({len(rows)} rows)")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[1]).parse_args()
    evaluate()


if __name__ == "__main__":
    main()
