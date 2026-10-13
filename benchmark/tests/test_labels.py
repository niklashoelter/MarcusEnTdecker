import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis" / "labels"))

import linear_shapes  # noqa: E402
import robust_fits  # noqa: E402
import robust_labels  # noqa: E402
import s2_posteriors  # noqa: E402
import s4_downstream  # noqa: E402
import scoring  # noqa: E402

X = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]) * 1e-3          # M, the 4 mM design


def test_ln_fit_recovers_stern_volmer_constant():
    K = 120.0
    lnI = np.log(2e5) - np.log1p(K * X)
    p, rss = linear_shapes.fit(X, lnI)
    assert p[1] == pytest.approx(K, rel=1e-6)
    assert rss == pytest.approx(0.0, abs=1e-12)
    assert linear_shapes.r2_y(X, lnI, p) == pytest.approx(1.0)


def test_ladder_label_needs_a_straighter_line_for_a_smaller_constant():
    assert linear_shapes.rung_label(60.0, 0.76) == 1
    assert linear_shapes.rung_label(15.0, 0.85) == 0
    assert linear_shapes.rung_label(15.0, 0.95) == 1
    assert linear_shapes.rung_label(5.0, 0.995) == 1
    assert linear_shapes.rung_label(500.0, 0.75) == 0
    assert linear_shapes.rung_label(np.nan, 0.99) == 0


def test_estimators_agree_on_a_clean_line():
    y = 1 + 300.0 * X
    assert robust_fits.ols_fit(X, y)[0] == pytest.approx(300.0)
    assert robust_fits.theil_sen(X, y)[0] == pytest.approx(300.0)
    assert robust_fits.irls(X, y, "tukey", 1e-3)[0] == pytest.approx(300.0)
    assert robust_fits.irls(X, y, "huber", 1e-3)[0] == pytest.approx(300.0)


def test_theil_sen_and_tukey_ignore_one_gross_outlier():
    y = 1 + 300.0 * X
    y[3] += 5.0
    assert robust_fits.theil_sen(X, y)[0] == pytest.approx(300.0, rel=0.05)
    assert robust_fits.irls(X, y, "tukey", 1e-3)[0] == pytest.approx(300.0, rel=1e-6)
    assert abs(robust_fits.ols_fit(X, y)[0] - 300.0) > 50


def test_cohen_kappa():
    a = np.array([1, 1, 0, 0])
    assert robust_labels.cohen_kappa(a, a) == pytest.approx(1.0)
    assert robust_labels.cohen_kappa(a, 1 - a) == pytest.approx(-1.0)
    b = np.array([1, 0, 0, 0])                      # po 0.75, pe 0.5
    assert robust_labels.cohen_kappa(a, b) == pytest.approx(0.5)


def test_label_classes_use_the_95_percent_bounds():
    p = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
    assert list(s4_downstream.classes(p)) == ["confident no quench", "confident no quench", "uncertain",
                                              "confident quench", "confident quench"]


def test_metrics_draws_match_the_benchmark_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40).astype(bool)
    pred = rng.integers(0, 2, 40).astype(bool)
    score = rng.normal(size=40)
    groups = np.array(["a", "b"] * 20)
    m = s4_downstream.metrics_draws(y[None], pred, score, groups)
    assert m["MCC"][0] == pytest.approx(scoring.metrics(y, pred)["MCC"])
    assert m["ROC_AUC"][0] == pytest.approx(scoring.auc(y, score))
    assert m["AUC_within_PC_pooled"][0] == pytest.approx(scoring.auc_within(y, score, groups)["pooled"])


def test_posterior_of_a_noiseless_series_peaks_at_its_constant():
    nm = pd.DataFrame({"b_additive / counts": [0.0], "p_outlier_well": [0.05], "k_outlier": [3.0]})
    post = s2_posteriors.Posterior(nm, np.random.default_rng(0))
    K_true = 200.0
    I = 2e5 / (1 + K_true * X)
    s = np.full(len(X), 0.005)
    K, dK, ll = post.loglik(X, np.log(I), s, X.max(), 0.0, 0.0)
    d = post.summarise(K, dK, ll, 30.0, 1000.0, 10.0)
    assert d["K_med"] == pytest.approx(K_true, rel=0.02)
    assert d["P_K>Kdet"] > 0.99 and d["P_K>0"] > 0.99
    assert d["log10kq_med"] == pytest.approx(np.log10(K_true / 1e-6), abs=0.02)


def test_mixture_likelihood_prefers_the_null_width_of_the_data():
    rng = np.random.default_rng(1)
    kt = rng.normal(0, 0.1, 200)
    st = np.full(200, 0.01)
    blk = np.repeat([1, 2], 100)

    def nll(tau):
        return s2_posteriors.mix_nll([0.0, 0.0, np.log(tau), 5.0, 0.0, 0.0], kt, st, blk)
    assert nll(0.1) < nll(0.03) and nll(0.1) < nll(0.3)
