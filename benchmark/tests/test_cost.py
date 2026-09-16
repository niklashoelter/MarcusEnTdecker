import numpy as np
import pandas as pd
import pytest

import cost


def timing_rows(quencher, hours):
    return [{"quencher_name": quencher, "calculation": c, "cpu / h": h} for c, h in hours.items()]


def full_hours(scale=1.0):
    hours = {c: 1.0 * scale for group in cost.CALCULATIONS.values() for c in group}
    hours["dft ox opt_freq"] = 100.0                      # counts toward no criterion
    return hours


def test_cost_sums_the_calculations_of_each_group():
    timings = pd.DataFrame(timing_rows("Q1", full_hours()))
    table = cost.per_quencher(timings).set_index("quencher_name")
    for group, calculations in cost.CALCULATIONS.items():
        assert table.loc["Q1", group] == pytest.approx(len(calculations))


def test_cost_adds_rows_of_the_same_calculation():
    rows = timing_rows("Q1", full_hours()) + [{"quencher_name": "Q1", "calculation": "dvte md", "cpu / h": 4.0}]
    table = cost.per_quencher(pd.DataFrame(rows)).set_index("quencher_name")
    assert table.loc["Q1", "DvTE"] == pytest.approx(len(cost.CALCULATIONS["DvTE"]) + 4.0)


def test_cost_is_blank_when_a_calculation_is_missing():
    hours = full_hours()
    del hours["dft T1 sp_vert_spin_change"]
    table = cost.per_quencher(pd.DataFrame(timing_rows("Q1", hours))).set_index("quencher_name")
    assert np.isnan(table.loc["Q1", "four-point DFT"])
    assert table.loc["Q1", "adiabatic DFT"] == pytest.approx(4.0)


def test_four_point_contains_the_adiabatic_calculations():
    assert set(cost.CALCULATIONS["adiabatic DFT"]) < set(cost.CALCULATIONS["four-point DFT"])


def wide_table():
    return pd.DataFrame({"quencher_name": ["Q1", "Q2", "Q3"],
                         "adiabatic DFT": [10.0, 20.0, 60.0],
                         "DvTE": [100.0, np.nan, 300.0]})


def test_normalized_divides_by_the_atom_count():
    long = cost.normalized(wide_table(), {"Q1": 10, "Q2": 20, "Q3": 30})
    row = long[(long["quencher_name"] == "Q3") & (long["group"] == "adiabatic DFT")].iloc[0]
    assert row["n_atoms"] == 30
    assert row["cpu / h"] == pytest.approx(60.0)
    assert row["cpu_per_atom / h"] == pytest.approx(2.0)


def test_statistics_over_quenchers_per_molecule_and_per_atom():
    long = cost.normalized(wide_table(), {"Q1": 10, "Q2": 20, "Q3": 30})
    stats = cost.statistics(long).set_index(["group", "normalization"])
    per_molecule = stats.loc[("adiabatic DFT", "per molecule")]
    assert per_molecule["quenchers"] == 3
    assert (per_molecule["min / h"], per_molecule["max / h"]) == (10.0, 60.0)
    assert per_molecule["mean / h"] == pytest.approx(30.0)
    assert per_molecule["std / h"] == pytest.approx(np.std([10.0, 20.0, 60.0], ddof=1))
    assert stats.loc[("adiabatic DFT", "per atom"), "mean / h"] == pytest.approx(4.0 / 3.0)
    assert stats.loc[("DvTE", "per molecule"), "quenchers"] == 2


def test_dataset_totals_sum_over_quenchers_and_their_atoms():
    long = cost.normalized(wide_table(), {"Q1": 10, "Q2": 20, "Q3": 30})
    totals = cost.dataset_totals(long).set_index("group")
    assert totals.loc["adiabatic DFT", "total / h"] == pytest.approx(90.0)
    assert totals.loc["adiabatic DFT", "atoms"] == 60
    assert totals.loc["adiabatic DFT", "total_per_atom / h"] == pytest.approx(1.5)
    assert totals.loc["DvTE", "quenchers"] == 2
    assert totals.loc["DvTE", "atoms"] == 40
