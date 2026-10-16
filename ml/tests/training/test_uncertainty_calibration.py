import numpy as np
import pandas as pd

from ml.training import uncertainty_calibration


def test_decile_medians_take_the_lowest_and_highest_tenth_of_the_uncertainty():
    unc = np.arange(1.0, 101.0)
    abs_error = unc * 10
    most, least = uncertainty_calibration.decile_medians(abs_error, unc)
    assert most == np.median(abs_error[:10]) and least == np.median(abs_error[-10:])


def test_rounding_is_two_decimals_for_kcal_and_three_for_fractions():
    table = pd.DataFrame({"target": ["adiabatic_energy"], "within_1_sigma": [0.73345],
                          "median_abs_error_most_certain_decile": [0.38765]})
    out = uncertainty_calibration.rounded(table)
    assert out["within_1_sigma"].tolist() == [0.733]
    assert out["median_abs_error_most_certain_decile"].tolist() == [0.39]
