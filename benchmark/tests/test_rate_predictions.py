import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import pairs  # noqa: E402
import rate_predictions  # noqa: E402
import theory  # noqa: E402


def test_full_rate_at_a_fixed_distance_uses_that_distance_for_every_pair():
    df = pd.DataFrame({pairs.CONTACT: [8.0, 9.5]})
    assert rate_predictions.escape_over_nu(df, 7.0) == pytest.approx(
        np.full(2, theory.escape_over_frequency(7.0)))
    assert rate_predictions.escape_over_nu(df) == pytest.approx(theory.escape_over_frequency(np.array([8.0, 9.5])))


def test_constant_predictor_has_no_rank_correlation_and_no_spread():
    measured = np.array([8.0, 9.0, 10.0])
    row = rate_predictions.parity_row(np.full(3, measured.mean()), measured)
    assert np.isnan(row["Spearman rho"])
    assert row["SD predicted / decades"] == 0.0
    assert row["RMSE / decades"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert row["within 1 decade"] == pytest.approx(1.0)
