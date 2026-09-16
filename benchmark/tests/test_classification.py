import numpy as np
import pandas as pd
import pytest

import classification
import pairs


def make_pairs(n=12, seed=0):
    """Synthetic scored pairs with every input column of the benchmark."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        pairs.PC_NAME: [f"PC{i % 3}" for i in range(n)],
        pairs.Q_ID: np.arange(n),
        pairs.Q_NAME: [f"Q{i}" for i in range(n)],
        pairs.TAU0: rng.uniform(100, 5000, n),
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


def test_rate_criterion_compares_k_tau0_with_k_sv_crit():
    df = make_pairs(4)
    setting = classification.Setting("rate", barrier_model="marcus_symmetric", k_sv_crit=10.0)
    predicted, score = classification.predict(df, setting, "dft")
    assert list(predicted) == list(score >= 1.0)


def test_candidates_of_raw_rate_are_a_single_untuned_setting():
    candidates = classification.candidates("marcus_rate", "lambda raw")
    assert len(candidates) == 1
    assert candidates[0].alpha == 1.0
    assert candidates[0].k_sv_crit == classification.K_SV_CRIT


def test_candidates_of_scaled_barrier_cover_alpha_and_threshold_grids():
    candidates = classification.candidates("marcus_barrier", "lambda scaled")
    assert len(candidates) == len(classification.ALPHAS) * len(classification.BARRIER_THRESHOLDS)


def test_n_free_counts_parameters_that_vary_across_candidates():
    assert classification.n_free(classification.candidates("marcus_barrier", "lambda raw")) == 1
    assert classification.n_free(classification.candidates("marcus_barrier", "lambda scaled")) == 2
    assert classification.n_free(classification.candidates("marcus_rate", "lambda raw")) == 0
    assert classification.n_free(classification.candidates("marcus_rate", "lambda scaled")) == 1


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
    tuned = best[best["criterion"].isin(["marcus_barrier", "marcus_rate", "rehm_weller_rate"])]
    for (criterion, variant), rows in tuned.groupby(["criterion", "variant"]):
        dft = rows[rows["parameters"] == "dft"].iloc[0]
        ml = rows[rows["parameters"] == "ml"].iloc[0]
        assert ml["alpha"] == dft["alpha"]
        assert ml["barrier_threshold / kcal mol-1"] == pytest.approx(
            dft["barrier_threshold / kcal mol-1"], nan_ok=True)
        assert ml["n_settings"] == 0


def test_best_table_has_every_criterion_variant_and_source():
    _, best = classification.score(make_pairs(30))
    rows = set(zip(best["criterion"], best["parameters"], best["variant"]))
    assert ("adiabatic", "dft", "") in rows and ("adiabatic", "ml", "") in rows
    assert sum(1 for c, _, _ in rows if c == "dvte") == 8
    for criterion in ("marcus_barrier", "marcus_rate", "rehm_weller_rate"):
        for source in pairs.SOURCES:
            for variant in ("lambda raw", "lambda scaled"):
                assert (criterion, source, variant) in rows
    assert {"AUC", "specificity", "MCC"} <= set(best.columns)
