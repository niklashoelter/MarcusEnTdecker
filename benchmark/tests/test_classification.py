import numpy as np
import pandas as pd
import pytest

import classification
import pairs
import theory


def make_pairs(n=12, seed=0):
    """Synthetic scored pairs with every input column of the benchmark."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 3}" for i in range(n)],
        pairs.Q_ID: np.arange(n),
        pairs.Q_NAME: [f"Q{i}" for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
        pairs.CONTACT: rng.uniform(6, 11, n),
        pairs.PC_E_T: rng.uniform(55, 70, n),
        pairs.PC_LAMBDA_S0_T1: rng.uniform(5, 15, n),
        pairs.PC_LAMBDA_T1_S0: rng.uniform(5, 15, n),
        pairs.QUENCHES: rng.integers(0, 2, n),
    })
    for p in pairs.DVTE_P:
        for kind in pairs.DVTE_KINDS:
            df[pairs.q_dvte(p, kind)] = rng.uniform(55, 75, n)
    for source in pairs.SOURCES:
        df[pairs.q_lambda_s0_t1(source)] = rng.uniform(5, 40, n)
        df[pairs.q_lambda_t1_s0(source)] = rng.uniform(5, 40, n)
        df[pairs.dg_ent(source)] = rng.uniform(-15, 10, n)
    return df


def test_adiabatic_counts_zero_driving_force_as_quenching():
    df = make_pairs(3)
    df[pairs.dg_ent("dft")] = [-1.0, 0.0, 1.0]
    predicted, _ = classification.predict(df, classification.Setting("adiabatic"), "dft")
    assert list(predicted) == [True, True, False]


def test_dvte_counts_equal_energies_as_quenching():
    df = make_pairs(3)
    df[pairs.PC_E_T] = 60.0
    df[pairs.q_dvte("0.008", "normal")] = [59.0, 60.0, 61.0]
    setting = classification.Setting("dvte", p="0.008", dvte_kind="normal")
    predicted, _ = classification.predict(df, setting, "dft")
    assert list(predicted) == [True, True, False]


def two_gate(variant):
    return classification.candidates("two_gate", variant)[0]


@pytest.mark.parametrize("variant", classification.TWO_GATE_VARIANTS)
def test_two_gate_needs_driving_force_and_barrier(variant):
    df = make_pairs(3)
    df[pairs.TAU0] = 1000.0
    df[pairs.PC_LAMBDA_S0_T1] = df[pairs.PC_LAMBDA_T1_S0] = 5.0
    # downhill and no barrier; downhill but a high barrier; a low barrier but uphill
    df[pairs.q_lambda_s0_t1("dft")] = df[pairs.q_lambda_t1_s0("dft")] = [5.0, 95.0, 1.0]
    df[pairs.dg_ent("dft")] = [-10.0, -1.0, 0.5]
    predicted, score = classification.predict(df, two_gate(variant), "dft")
    assert list(predicted) == [True, False, False]
    assert score[2] < score[1] < score[0]


@pytest.mark.parametrize("variant", classification.TWO_GATE_VARIANTS)
def test_two_gate_compares_the_asymmetric_barrier_with_the_lifetime_barrier(variant):
    df = make_pairs(40, seed=1)
    predicted, score = classification.predict(df, two_gate(variant), "dft")
    dG = pairs.driving_force(df, "dft")
    barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(df, "dft")))
    r = classification.TWO_GATE_VARIANTS[variant]
    r = pairs.contact_distances(df) if r is None else r
    t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), classification.K_SV_CRIT,
                                   theory.escape_over_frequency(r))
    assert list(predicted) == list((dG <= 0) & (barrier <= t_pc))
    assert score[dG > 0].max() < score[dG <= 0].min()


def test_two_gate_from_dft_sizes_takes_the_contact_distance_of_each_pair():
    df = make_pairs(2)
    df[pairs.TAU0] = 1000.0
    df[pairs.dg_ent("dft")] = -10.0
    df[pairs.CONTACT] = [5.0, 12.0]
    _, score = classification.predict(df, two_gate("r from DFT sizes"), "dft")
    _, fixed = classification.predict(df, two_gate("r = 7 A"), "dft")
    # t_PC rises by 3 RT ln(r / 7 A); the barrier is the same in both variants
    assert score - fixed == pytest.approx(3.0 * theory.RT * np.log(df[pairs.CONTACT] / 7.0))


def test_contact_distance_adds_the_radii_of_both_partners():
    df = pd.DataFrame({pairs.PC_RADIUS: [5.0, 5.0, 5.0], pairs.Q_RADIUS: [3.0, 3.5, np.nan]})
    r = pairs.contact_distance(df)
    assert r[:2] == pytest.approx([8.0, 8.5])
    assert np.isnan(r[2])


@pytest.mark.parametrize("variant", classification.TWO_GATE_VARIANTS)
def test_candidates_of_two_gate_are_a_single_untuned_setting(variant):
    candidates = classification.candidates("two_gate", variant)
    assert len(candidates) == 1
    assert candidates[0].k_sv_crit == classification.K_SV_CRIT
    assert candidates[0].contact_distance == classification.TWO_GATE_VARIANTS[variant]


def test_candidates_of_barrier_cover_the_threshold_grid():
    candidates = classification.candidates("marcus_barrier", "")
    assert len(candidates) == len(classification.BARRIER_THRESHOLDS)


def test_n_free_counts_parameters_that_vary_across_candidates():
    assert classification.n_free(classification.candidates("marcus_barrier", "")) == 1
    for variant in classification.TWO_GATE_VARIANTS:
        assert classification.n_free(classification.candidates("two_gate", variant)) == 0


def test_chosen_settings_match_the_best_table():
    df = make_pairs(40, seed=5)
    _, best = classification.score(df)
    for criterion, variant, source, setting in classification.chosen_settings(df):
        row = best[(best["criterion"] == criterion) & (best["variant"] == variant)
                   & (best["parameters"] == source)].iloc[0]
        predicted, _ = classification.predict(df, setting, source)
        assert int(predicted.sum()) == row["TP"] + row["FP"]


def test_ml_rows_reuse_the_settings_chosen_on_dft():
    _, best = classification.score(make_pairs(60, seed=3))
    tuned = best[best["criterion"] == "marcus_barrier"]
    for (criterion, variant), rows in tuned.groupby(["criterion", "variant"]):
        dft = rows[rows["parameters"] == "dft"].iloc[0]
        ml = rows[rows["parameters"] == "ml"].iloc[0]
        assert ml["barrier_threshold / kcal mol-1"] == pytest.approx(
            dft["barrier_threshold / kcal mol-1"], nan_ok=True)
        assert ml["n_settings"] == 0


def test_best_table_has_every_criterion_variant_and_source():
    _, best = classification.score(make_pairs(30))
    rows = set(zip(best["criterion"], best["parameters"], best["variant"]))
    assert ("adiabatic", "dft", "") in rows and ("adiabatic", "ml", "") in rows
    assert sum(1 for c, _, _ in rows if c == "dvte") == 8
    for source in pairs.SOURCES:
        assert ("marcus_barrier", source, "") in rows
        for variant in classification.TWO_GATE_VARIANTS:
            assert ("two_gate", source, variant) in rows
    assert {"AUC", "AUC_within_PC_median", "AUC_within_PC_pooled", "specificity",
            "MCC"} <= set(best.columns)
    assert "alpha" not in best.columns
    assert not {"marcus_rate", "rehm_weller_rate"} & set(best["criterion"])
