import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import figure_tables as ft  # noqa: E402


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


def test_cost_table_pairs_methods_with_the_criteria_they_enable():
    methods = list(ft.COST_METHODS)
    cpu = pd.DataFrame({m: [1.0, 3.0] for m in methods}, index=["Q1", "Q2"])
    metrics = pd.DataFrame([{"criterion": "adiabatic", "source": "ml", "role": "shown",
                             **{m: 0.5 for m in ft.COST_METRICS}}])
    t = ft.cost_table(cpu, cpu, metrics)
    assert sorted(t["method"]) == ["ML batch share per molecule", "ML per call"]
    assert set(t["cpu_s_median"]) == {2.0}
    assert set(t["ratio_of_cpu_medians_to_DvTE"]) == {1.0}
