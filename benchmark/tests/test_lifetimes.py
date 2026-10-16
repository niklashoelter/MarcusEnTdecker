import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import lifetimes  # noqa: E402


def test_gate_names_the_gate_that_rejects():
    dG = np.array([1.0, -1.0, -1.0, 0.0])
    predicted = np.array([False, False, True, True])
    assert list(lifetimes.gate(dG, predicted)) == ["energy", "barrier", "pass", "pass"]


def test_barrier_bins_count_exergonic_pairs_per_group():
    y = np.array([1, 0, 1, 1, 0], bool)
    short = np.array([True, True, False, False, False])
    exergonic = np.array([True, True, True, False, True])
    barrier = np.array([4.0, 5.0, 2.0, 4.0, 13.0])
    t = lifetimes.barrier_bins(y, short, exergonic, barrier).set_index(["tau0_group", "barrier_from / kcal mol-1"])
    assert (t.loc[("short", 3.0), "n_pairs"], t.loc[("short", 3.0), "n_quench"]) == (2, 1)
    assert t.loc[("long", 0.0), "fraction_quench"] == pytest.approx(1.0)
    assert t.loc[("long", 3.0), "n_pairs"] == 0                  # endergonic pair left out
    assert t.loc[("long", 12.0), "n_pairs"] == 1                 # last bin is open


def test_dvte_table_splits_rejected_false_positives_by_gate():
    y = np.array([0, 0, 1, 0], bool)
    classes = np.array(["rigid", "flexible", "rigid", "rigid"])
    dG = np.array([0.5, -1.0, 0.2, -2.0])
    dvte = np.array([True, True, True, False])
    two_gate = np.array([False, False, False, True])
    gates = lifetimes.gate(dG, two_gate)
    t = lifetimes.dvte_table(y, classes, dG, dvte, two_gate, gates).set_index("quencher_class")
    assert t.loc["rigid", "dvte_FP_rejected_energy_gate"] == 1
    assert t.loc["flexible", "dvte_FP_rejected_barrier_gate"] == 1
    assert t.loc["rigid", "dvte_TP_rejected_energy_gate"] == 1
    assert t.loc["rigid", "two_gate_FP_not_dvte"] == 1
    assert t.loc["all", "dvte_FP_rejected"] == 2
    assert t.loc["rigid", "dG_EnT_median_energy_gate / kcal mol-1"] == pytest.approx(0.5)
