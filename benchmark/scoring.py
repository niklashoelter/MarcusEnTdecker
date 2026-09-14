# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
Metrics, and the two things that make them mean less than they look like.

Reads   nothing
Writes  nothing

WHY THESE METRICS
    MCC is the headline. Accuracy is not usable on its own here: 735 of the 1248 scored
    pairs quench, so answering "quenches" for every pair already scores 0.59. MCC uses
    all four cells of the confusion matrix and does not reward that, which is why the
    original benchmark selected on it and why this one reports it first.
    The four cells are written out beside it anyway, because a method with sensitivity
    0.99 and specificity 0.05 should not be able to hide inside a summary statistic.

    No sklearn. Every metric here is four counts and one square root, and the repository
    keeps its dependency surface to numpy and pandas.

THE FIRST THING THAT MAKES THEM MEAN LESS: TUNED THRESHOLDS
    A method whose barrier threshold is chosen by maximizing MCC, and whose MCC is then
    reported at that threshold, has been fitted to the data it is scored on. That is not
    a mistake as long as it is labeled, but it does mean its number is not comparable
    with the adiabatic rule's, which has nothing to fit.

    `leave_one_photocatalyst_out` is the honest counterpart. For each photocatalyst it
    picks the best setting using every OTHER photocatalyst, applies it to the held-out
    one, and pools the held-out predictions into a single confusion matrix. Pooling
    rather than averaging per-fold MCC matters: the original's leave-one-out study
    averaged per-fold MCC and got 0.438 against an in-sample 0.617, then attributed the
    gap to the variable positive fraction of small folds. Pooling removes that particular
    artifact, so whatever gap remains is the real cost of choosing a threshold.

    Photocatalyst is the right grouping unit, not the row. Every reorganization energy
    and every triplet energy on the photocatalyst side is constant across that
    photocatalyst's 48 pairs, so a random split leaves the same photocatalyst on both
    sides and a model can key on its fingerprint instead of on the chemistry. The
    original's own audit measured this for its machine-learning baselines: a random
    5-fold random forest scored 0.805 and the same model grouped by photocatalyst scored
    0.640.

THE SECOND: HALF THE NEGATIVES CARRY NO INFORMATION
    Handled in pairs.ground_truth, which sets the label on K_SV rather than on k_q, not
    here. This module scores whatever y it is given.
"""

import numpy as np
import pandas as pd


def confusion(y_true, y_pred):
    """(TP, FP, TN, FN) as plain ints."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, fp, tn, fn


def metrics(y_true, y_pred) -> dict:
    """MCC, accuracy, F1, sensitivity, specificity and the four counts behind them."""
    tp, fp, tn, fn = confusion(y_true, y_pred)
    n = tp + fp + tn + fn
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0.0
    return {
        "n": n,
        "n_positive": tp + fn,
        "MCC": float(mcc),
        "Accuracy": (tp + tn) / n if n else float("nan"),
        "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "Sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def _mcc_from_counts(tp, fp, tn, fn):
    """Vectorized MCC over arrays of confusion counts. Zero where undefined."""
    tp, fp, tn, fn = (np.asarray(x, dtype=float) for x in (tp, fp, tn, fn))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    out = np.zeros_like(denom)
    ok = denom > 0
    out[ok] = (tp[ok] * tn[ok] - fp[ok] * fn[ok]) / np.sqrt(denom[ok])
    return out


def best_by_mcc(y_true, pred_matrix):
    """Index of the row of `pred_matrix` (n_configs x n_rows) with the highest MCC."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(pred_matrix, dtype=bool)
    tp = (p & y).sum(axis=1)
    fp = (p & ~y).sum(axis=1)
    fn = (~p & y).sum(axis=1)
    tn = (~p & ~y).sum(axis=1)
    return int(np.argmax(_mcc_from_counts(tp, fp, tn, fn)))


def leave_one_photocatalyst_out(y_true, pred_matrix, groups) -> dict:
    """Choose the setting on the other photocatalysts, score on the held-out one.

    `pred_matrix` is (n_configs, n_rows) boolean -- every setting of one method family
    evaluated on every pair. `groups` is one label per row, the photocatalyst.

    Returns the pooled metrics over all held-out predictions, plus how many distinct
    settings the folds chose between. A family whose folds all pick the same setting and
    whose pooled score matches its in-sample score was not really being fitted; one where
    the two diverge was.
    """
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(pred_matrix, dtype=bool)
    groups = np.asarray(groups)
    if p.shape[0] == 1:
        # Nothing to select between; held-out and in-sample are the same prediction.
        return {**metrics(y, p[0]), "n_settings_chosen": 1, "folds": len(set(groups))}

    # Totals once, then subtract each fold to get its complement, so the cost is one
    # pass over the matrix per fold rather than a full rescore.
    tp_all = (p & y).sum(axis=1)
    fp_all = (p & ~y).sum(axis=1)
    fn_all = (~p & y).sum(axis=1)
    tn_all = (~p & ~y).sum(axis=1)

    held = np.zeros(len(y), dtype=bool)
    chosen = []
    for g in np.unique(groups):
        in_fold = groups == g
        if in_fold.all():
            raise ValueError("only one photocatalyst present; leave-one-out is vacuous")
        pf = p[:, in_fold]
        yf = y[in_fold]
        tp = tp_all - (pf & yf).sum(axis=1)
        fp = fp_all - (pf & ~yf).sum(axis=1)
        fn = fn_all - (~pf & yf).sum(axis=1)
        tn = tn_all - (~pf & ~yf).sum(axis=1)
        pick = int(np.argmax(_mcc_from_counts(tp, fp, tn, fn)))
        chosen.append(pick)
        held[in_fold] = p[pick, in_fold]

    out = metrics(y, held)
    out["n_settings_chosen"] = len(set(chosen))
    out["folds"] = len(np.unique(groups))
    return out


def rate_agreement(k_q, k_q_floor, censored, k_pred) -> dict:
    """How close a predicted rate constant is to the measured one, censoring respected.

    A naive RMSE over all 1296 pairs would be wrong here, and wrong in a flattering
    direction for a theory that under-predicts: for the 662 censored pairs the recorded
    k_q is an upper bound, so a prediction sitting anywhere below the floor is consistent
    with the observation and a residual computed against the bound is meaningless.

    So the two halves are reported separately and neither is combined into a single
    number:

      on detected pairs      RMSE and median bias in log10 units ("dex"), the ordinary
                             regression question
      on censored pairs      the fraction whose predicted rate falls at or below that
                             pair's own detection floor, i.e. the fraction of
                             non-detections the theory is consistent with

    This is descriptive. It is not a censored-likelihood fit, and it takes no position on
    whether the benchmark should become a censored regression -- that question is open.
    """
    k_q = np.asarray(k_q, dtype=float)
    floor = np.asarray(k_q_floor, dtype=float)
    censored = np.asarray(censored, dtype=float) == 1
    k_pred = np.asarray(k_pred, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_pred = np.log10(np.maximum(k_pred, 1e-30))
        log_obs = np.log10(np.maximum(k_q, 1e-30))

    detected = ~censored & np.isfinite(log_obs) & (k_q > 0) & np.isfinite(log_pred)
    residual = log_pred[detected] - log_obs[detected]

    usable_censored = censored & np.isfinite(log_pred) & np.isfinite(floor)
    consistent = k_pred[usable_censored] <= floor[usable_censored]

    return {
        "n_detected": int(detected.sum()),
        "rmse_dex_detected": float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan"),
        "median_bias_dex_detected": float(np.median(residual)) if residual.size else float("nan"),
        "n_censored": int(usable_censored.sum()),
        "fraction_consistent_with_floor": float(np.mean(consistent)) if consistent.size else float("nan"),
    }


def as_frame(rows) -> pd.DataFrame:
    """Rows of dicts to a DataFrame, MCC first, so the CSVs read left to right."""
    df = pd.DataFrame(rows)
    lead = [c for c in ("mechanism_rule", "set_source", "bounds_policy",
                        "theory", "method_type", "family", "method", "n_free",
                        "n", "n_positive", "MCC") if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]
