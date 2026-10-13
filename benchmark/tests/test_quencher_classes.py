import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import pairs  # noqa: E402
import quencher_classes as qc  # noqa: E402


def parameters(lambda_t1_s0):
    return pd.Series({"q_E_T_dft / kcal mol-1": 60.0, pairs.q_lambda_s0_t1("dft"): 14.0,
                      pairs.q_lambda_t1_s0("dft"): lambda_t1_s0, pairs.q_dvte("0.008", "normal"): 58.5,
                      "q_E_T_ml / kcal mol-1": 60.25})


GEOMETRY = pd.Series({"max_stretch_pct": 7.6, "stretch_bond": "C1-C2 1.37->1.47", "rmsd_heavy": 0.044,
                      "dtors_double_deg": 0.0, "dtors_any_heavy_deg": 12.5})


def test_every_quencher_has_one_of_the_three_classes():
    assert sorted(qc.CLASSES) == list(range(1, 49))
    assert {cls for _, cls, _ in qc.CLASSES.values()} == set(qc.CLASS_NAMES)


def test_class_counts():
    counts = pd.Series([cls for _, cls, _ in qc.CLASSES.values()]).value_counts()
    assert (counts["rigid"], counts["flexible"], counts["dissociative"]) == (32, 12, 4)


def test_evidence_reports_the_parameters_and_the_lambda_cut():
    text = qc.evidence("planar T1", parameters(15.81), GEOMETRY)
    assert text.startswith("planar T1. DFT: E_T 60.00, lS0T1 14.00, lT1S0 15.81 (cut 20 -> rigid)")
    assert "DvTE(0.8%)-E_T -1.50" in text and "E_T,ML-E_T,DFT 0.25 kcal/mol" in text
    assert text.endswith("7.6 % (C1-C2 1.37->1.47); heavy-atom RMSD 0.04 A")
    assert "(cut 20 -> flexible/diss)" in qc.evidence("x", parameters(20.0), GEOMETRY)


def test_classify_flags_ambiguous_notes_and_takes_the_larger_torsion():
    quenchers = {3: {"name": "a", "smiles": "C"}, 4: {"name": "b", "smiles": "C"}}
    params = pd.DataFrame({3: parameters(15.0), 4: parameters(15.0)}).T
    geometry = pd.DataFrame({"Q3": GEOMETRY, "Q4": GEOMETRY}).T
    table = qc.classify(quenchers, params, geometry).set_index("id")
    assert list(table["ambiguous"]) == [1, 0]                    # Q3's note starts with AMBIGUOUS
    assert list(table["dtheta / deg"]) == [12.5, 12.5]
    assert list(table["proposed_class"]) == ["rigid", "rigid"]


def test_cut_disagreements():
    table = pd.DataFrame({"id": ["Q1", "Q2", "Q3", "Q4"],
                          "proposed_class": ["rigid", "rigid", "flexible", "flexible"]})
    params = pd.DataFrame({pairs.q_lambda_t1_s0("dft"): [25.0, 10.0, 25.0, 10.0]}, index=[1, 2, 3, 4])
    assert qc.cut_disagreements(table, params) == (["Q1"], ["Q4"])
