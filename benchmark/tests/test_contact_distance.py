import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import classification  # noqa: E402
import contact_distance  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402


def make_pairs(n=12, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 3}" for i in range(n)],
        "pc_old_id": [i % 3 + 1 for i in range(n)],
        pairs.Q_ID: [i % 4 + 1 for i in range(n)],
        pairs.Q_NAME: [f"Q{i % 4 + 1}" for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.PC_RADIUS: rng.uniform(4.5, 6.0, n),
        pairs.Q_RADIUS: rng.uniform(2.5, 4.5, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.QUENCHES: rng.integers(0, 2, n),
    })
    df[pairs.CONTACT] = pairs.contact_distance(df)
    for source in pairs.SOURCES:
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 40, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
        df[pairs.dg_ent(source)] = rng.uniform(-15, 10, n)
    return df


def test_threshold_shift_is_zero_at_seven_angstrom_and_has_slope_3rt_over_r():
    assert contact_distance.threshold_shift(7.0) == pytest.approx(0.0)
    r, step = 8.0, 1e-4
    slope = (contact_distance.threshold_shift(r + step) - contact_distance.threshold_shift(r)) / step
    assert slope == pytest.approx(3.0 * theory.RT / r, rel=1e-3)


def test_threshold_shift_is_the_difference_of_lifetime_thresholds():
    tau0 = 1e-6
    t = {r: theory.lifetime_barrier(tau0, 10.0, theory.escape_over_frequency(r)) for r in (7.0, 9.0)}
    assert t[9.0] - t[7.0] == pytest.approx(contact_distance.threshold_shift(9.0), abs=1e-3)


@pytest.mark.parametrize("source", pairs.SOURCES)
def test_two_gate_reproduces_the_benchmark_criterion(source):
    df = make_pairs()
    for variant, distance in classification.TWO_GATE_VARIANTS.items():
        expected = classification.predict(df, classification.candidates("two_gate", variant)[0], source)
        r = pairs.contact_distances(df) if distance is None else distance
        predicted, score = contact_distance.two_gate(df, source, r)
        np.testing.assert_array_equal(predicted, expected[0])
        np.testing.assert_allclose(score, expected[1])


def test_a_larger_contact_distance_never_removes_a_predicted_pair():
    df = make_pairs(40, seed=1)
    near, _ = contact_distance.two_gate(df, "dft", 5.0)
    far, _ = contact_distance.two_gate(df, "dft", 10.0)
    assert not (near & ~far).any()


def test_extent_radii_are_mapped_by_old_photocatalyst_id_and_quencher_id(tmp_path, monkeypatch):
    table = tmp_path / "molecular_extent.csv"
    table.write_text("kind,id,n_atoms,r_extent / A\n"
                     "photocatalyst,1,10,8.00\nphotocatalyst,2,10,7.00\nphotocatalyst,3,10,6.00\n"
                     "quencher,1,5,4.00\nquencher,2,5,5.00\nquencher,3,5,6.00\nquencher,4,5,7.00\n")
    monkeypatch.setattr(contact_distance, "EXTENT", table)
    df = make_pairs(4)
    r_pc, r_q = contact_distance.extent_radii(df)
    np.testing.assert_allclose(r_pc, [8.0, 7.0, 6.0, 8.0])
    np.testing.assert_allclose(r_q, [4.0, 5.0, 6.0, 7.0])
