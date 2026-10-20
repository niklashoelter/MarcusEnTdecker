import pandas as pd

from ml.training import evaluate


def test_summary_uses_the_population_sd_over_random_splits():
    metrics = pd.DataFrame({"split": ["random_42", "random_101", "oos_styryl"],
                            "target": ["adiabatic_energy"] * 3,
                            "mae": [1.0, 3.0, 99.0], "rmse": [1.0, 1.0, 1.0],
                            "r2": [0.5, 0.5, 0.5]})
    row = evaluate.summary(metrics).iloc[0]
    assert row["n_splits"] == 2 and row["mae_mean"] == 2.0 and row["mae_sd"] == 1.0


def test_rounding_is_two_decimals_for_kcal_and_three_for_unitless():
    table = pd.DataFrame({"target": ["adiabatic_energy", "spin_population"],
                          "mae": [1.23456, 0.0123456], "r2": [0.987654, 0.987654]})
    out = evaluate.rounded(table)
    assert out["mae"].tolist() == [1.23, 0.012] and out["r2"].tolist() == [0.988, 0.988]
