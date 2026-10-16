"""
How well the predicted uncertainty describes the error, from the test predictions of a split.

Reads   ml/results/<split>/test_predictions_graph.csv   for each split asked for (default random_42)
Writes  ml/results/uncertainty_calibration.csv          per split and molecular target:
            within_1_sigma, within_2_sigma        fraction of test molecules with
                                                  |error| < 1 and 2 sigma_pred
            median_abs_error_most_certain_decile, median_abs_error_least_certain_decile
                                                  median |error| in the decile of smallest
                                                  and largest sigma_pred

    uv run --extra ml python -m ml.training.uncertainty_calibration [split ...]

Written for this repository. Deciles: cut at the quantiles of sigma_pred, closed on both
sides. Rounding: kcal/mol 2 decimals, fractions 3.
"""

import argparse

import numpy as np
import pandas as pd

from ml.training.evaluate import decimals, predictions
from ml.training.test_predictions import GRAPH_TARGETS, RESULTS

DEFAULT_SPLITS = ["random_42"]
OUTPUT = RESULTS / "uncertainty_calibration.csv"


def decile_medians(abs_error, unc):
    """Median absolute error in the lowest and the highest decile of the uncertainty."""
    edges = np.quantile(unc, np.linspace(0, 1, 11))
    lowest = abs_error[(unc >= edges[0]) & (unc <= edges[1])]
    highest = abs_error[(unc >= edges[-2]) & (unc <= edges[-1])]
    return np.median(lowest), np.median(highest)


def calibration(splits):
    rows = []
    for split in splits:
        for target in GRAPH_TARGETS:
            found = predictions(split, target)
            if found is None:
                raise SystemExit(f"no test predictions for {split}")
            y, yhat, unc = found
            ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(unc)
            abs_error, unc = np.abs(yhat[ok] - y[ok]), unc[ok]
            most, least = decile_medians(abs_error, unc)
            rows.append({"split": split, "target": target, "n": len(unc),
                         "within_1_sigma": np.mean(abs_error < unc),
                         "within_2_sigma": np.mean(abs_error < 2 * unc),
                         "median_abs_error_most_certain_decile": most,
                         "median_abs_error_least_certain_decile": least})
    return pd.DataFrame(rows)


def rounded(table):
    """Two decimals for kcal/mol, three for the fractions."""
    table = table.copy()
    energies = [c for c in table.columns if c.startswith("median_abs_error")]
    for target in table["target"].unique():
        rows = table["target"] == target
        table.loc[rows, energies] = table.loc[rows, energies].round(decimals(target))
    fractions = [c for c in table.columns if c.startswith("within")]
    table[fractions] = table[fractions].round(3)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("splits", nargs="*", default=DEFAULT_SPLITS)
    table = calibration(parser.parse_args().splits)
    rounded(table).to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"wrote {OUTPUT.name} ({len(table)} rows)")


if __name__ == "__main__":
    main()
