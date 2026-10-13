import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import figure_tables as ft  # noqa: E402
import theory  # noqa: E402


def test_complex_class_from_the_name():
    assert ft.complex_class("[Ru(bpy)3][PF6]2") == "Ru(II) tris(diimine)"
    assert ft.complex_class("Ir(pmb)3") == "Ir(III) homoleptic tris(NHC), neutral"
    assert ft.complex_class("Ir(ppy)3") == "Ir(III) homoleptic Ir(C^N)3, neutral"
    assert ft.complex_class("[Ir(ppy)2(dtbbpy)][PF6]") == "Ir(III) heteroleptic [Ir(C^N)2(N^N)]+, cationic"
    assert ft.complex_class(np.nan) == ""


def test_numbered_orders_by_the_number():
    t = pd.DataFrame({"q": ["Q10", "Q2", "Q1"]})
    assert list(ft.numbered(t, "q", "Q")["q"]) == ["Q1", "Q2", "Q10"]


def test_quartiles():
    s = ft.quartiles(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), "x")
    assert (s["x_median"], s["x_q25"], s["x_q75"], s["x_IQR"], s["x_mean"]) == (3.0, 2.0, 4.0, 2.0, 3.0)


def test_lifetime_curve_models_and_values():
    t = ft.lifetime_curve(np.array([7.5, 8.0, 10.0]))
    assert len(t) == 5 * len(ft.TAU_GRID_NS)
    assert list(t.drop_duplicates("model")["role"]) == ["main", "range", "range", "secondary", "SI"]
    main = t[t.role == "main"]
    expected = theory.lifetime_barrier(1e-6, classification.K_SV_CRIT, theory.escape_over_frequency(8.0))
    at_1us = main.iloc[np.argmin(np.abs(main["tau0 / ns"] - 1000.0))]
    assert at_1us["t_PC / kcal mol-1"] == pytest.approx(expected, abs=0.005)


def test_cost_table_pairs_methods_with_the_criteria_they_enable():
    methods = list(ft.COST_METHODS)
    cpu = pd.DataFrame({m: [1.0, 3.0] for m in methods}, index=["Q1", "Q2"])
    metrics = pd.DataFrame([{"criterion": "adiabatic", "source": "ml", "role": "shown",
                             **{m: 0.5 for m in ft.COST_METRICS}}])
    t = ft.cost_table(cpu, cpu, metrics)
    assert sorted(t["method"]) == ["ML batch share per molecule", "ML per call"]
    assert set(t["cpu_s_median"]) == {2.0}
    assert set(t["ratio_of_cpu_medians_to_DvTE"]) == {1.0}
