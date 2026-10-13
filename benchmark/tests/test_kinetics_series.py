import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import kinetics_series  # noqa: E402
import pairs  # noqa: E402


def frame(pcs, quenchers, k_q):
    return pd.DataFrame({pairs.PC_NAME: pcs, pairs.Q_NAME: quenchers, pairs.K_Q: k_q})


def test_series_needs_enough_pairs_with_a_measured_rate():
    df = frame(["PC1"] * 6 + ["PC2"] * 6, [f"Q{i}" for i in range(6)] * 2,
               [1e8, 2e8, 3e8, 4e8, 5e8, 6e8, 1e8, 2e8, 0.0, 0.0, 0.0, 5e8])
    predicted = {s: np.log10(df[pairs.K_Q].to_numpy() + 1.0) for s in pairs.SOURCES}
    table = kinetics_series.series(df, predicted, kinetics_series.measured(df), min_size=5)
    assert set(table.series) == {"PC1"}
    assert table["Spearman rho"].to_numpy() == pytest.approx(1.0)


def test_summary_counts_positive_series():
    table = pd.DataFrame({"kind": "quencher", "series": ["Q1", "Q2", "Q3"], "parameters": "dft",
                          "n": 5, "Spearman rho": [0.5, -0.2, 0.9]})
    summary = kinetics_series.summarize(table).iloc[0]
    assert (summary.n_series, summary.rho_positive) == (3, 2)
    assert summary["median"] == pytest.approx(0.5)


def test_rule_quencher_has_the_most_measured_pairs_and_refuses_a_tie():
    df = frame(["PC1", "PC2", "PC3", "PC1"], ["Q1", "Q1", "Q2", "Q2"], [1e8, 1e8, 1e8, 0.0])
    assert kinetics_series.rule_quencher(df, kinetics_series.measured(df)) == "Q1"
    tie = frame(["PC1", "PC2"], ["Q1", "Q2"], [1e8, 1e8])
    with pytest.raises(ValueError):
        kinetics_series.rule_quencher(tie, kinetics_series.measured(tie))
