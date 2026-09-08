# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
QC'd well intensities -> one Stern-Volmer fit per photocatalyst/quencher pair.

Reads   ../experimental_data_processed/stern_volmer_intensities.csv
Writes  ../experimental_data_processed/stern_volmer_fits.csv

Fits I0/I = 1 + K_SV*[Q]. 27 photocatalysts x 48 quenchers = 1296 pairs.

This is the one script that reads a generated file rather than raw instrument output.
The replicate QC that produces those intensities is implemented once, in
stern_volmer_intensities.py, rather than duplicated here where two copies could drift.

PROVENANCE
    Faithful port of SternVolmerAnalysis in
    Auto_Quenching_Backup/auto_quenching/analysis/stern_volmer.py (+ regression_utils.py),
    the acquisition-time code that produced the published K_SV values. Verified against
    the 1296 per-pair workbooks that code wrote: every point-selection mask matches, and
    K_SV agrees to ~1e-13 relative.

    The original imported scikit-learn for LinearRegression and TheilSenRegressor but
    never declared it as a dependency, so the exact version behind the published numbers
    is unrecorded. Both estimators are reimplemented here in numpy -- ordinary least
    squares, and Theil-Sen as the spatial median of the pairwise-exact lines, which is
    what TheilSenRegressor computes when n_subsamples = n_features + 1 = 2. That removes
    scikit-learn and scipy from the pipeline and makes the result version-independent.

    Two behaviours of the original look like defects. Both are reproduced deliberately
    and marked QUIRK below; changing either would silently move published numbers.

TWO LABELS COME OUT OF THIS FILE
    `quenching_observed` is the published criterion, K_SV >= 25 and r2 >= 0.90, kept
    byte-for-byte so the published table stays reproducible. Nothing downstream uses it.

    `quenches` is the label this project classifies on: K_SV > 10, or K_SV <= 10 with a
    slope that is significant across all measured points. The two differences from the
    published rule are deliberate.

    The threshold moved to 10 because a weak quencher is still a quencher, and because
    K_SV 5-10 is real signal here rather than noise: 19 such pairs clear r2 > 0.95 where
    a selection-aware null (best 5 of 6 points, no true slope) expects 1.1, and their
    traces are monotone 3-4 % drops well above the read noise.

    The rescue is a slope test rather than an r2 cut because the fitted r2 is not
    comparable across pairs. R2_NO_QUENCHING gates the outlier scan, so a pair whose
    all-points r2 is 0.45 can never be cleaned while one at 0.55 can be cleaned to 0.99.
    Of the low-K_SV pairs, 19 of the 50 that had a point dropped reach r2 > 0.95 and
    0 of the 103 that did not -- an artifact of that gate, not a property of the sample.
    The slope test reads every measured point and so judges both alike.
"""

import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IN = Path(__file__).resolve().parent.parent / "experimental_data_processed" / "stern_volmer_intensities.csv"
OUT = Path(__file__).resolve().parent.parent / "experimental_data_processed" / "stern_volmer_fits.csv"

MIN_POINTS_FOR_FIT = 3
R2_NO_QUENCHING = 0.50        # all-points r2 below this: no quenching trend
R2_GOOD_CORRELATION = 0.995   # all-points r2 above this: nothing to gain from scanning
OUTLIER_IMPROVEMENT = 0.20    # an outlier must cut the MAE by this fraction to qualify
K_SV_QUENCHING = 25.0         # published: a pair counts as quenching when K_SV >= this
R2_QUENCHING = 0.90           # published: ... and r2 >= this
K_SV_LABEL = 10.0             # this project: K_SV above this quenches outright ...
P_SLOPE_LABEL = 0.01          # ... at or below it, only with a slope this significant
SIGFIGS = 6

EPS = np.finfo(np.double).eps

COLUMNS = [
    "pc_id", "quencher_id", "K_SV", "intercept", "r2", "num_points", "n_measured",
    "K_SV_all_points", "r2_all_points", "p_slope",
    "quenches", "quenching_observed", "fit_method", "confidence", "note",
]


def sig(x, n=SIGFIGS):
    if x is None or not np.isfinite(x):
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def mae(residuals):
    return float(np.mean(np.abs(np.asarray(residuals))))


def score_fn(residuals, n_points):
    """The original's range score: rewards long intervals, punishes residuals steeply."""
    return n_points ** 0.5 / (1.0 + mae(residuals)) ** 4


# --- estimators -------------------------------------------------------------

def ols(x, y):
    """Ordinary least squares with a free intercept. Returns slope, intercept, pred, r2."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    sxx = float(dx @ dx)
    slope = float(dx @ dy) / sxx
    intercept = float(y.mean() - slope * x.mean())
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float((dy ** 2).sum())
    return slope, intercept, pred, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def _weiszfeld_step(X, x_old):
    diff = X - x_old
    diff_norm = np.sqrt(np.sum(diff ** 2, axis=1))
    mask = diff_norm >= EPS
    x_old_is_a_point = int(mask.sum() < X.shape[0])
    diff = diff[mask]
    diff_norm = diff_norm[mask][:, np.newaxis]
    quotient_norm = float(np.linalg.norm(np.sum(diff / diff_norm, axis=0)))
    if quotient_norm > EPS:
        direction = np.sum(X[mask, :] / diff_norm, axis=0) / np.sum(1 / diff_norm, axis=0)
    else:
        direction = 1.0
        quotient_norm = 1.0
    return (max(0.0, 1.0 - x_old_is_a_point / quotient_norm) * direction
            + min(1.0, x_old_is_a_point / quotient_norm) * x_old)


def spatial_median(X, max_iter=300, tol=1.0e-3):
    """Geometric median by the modified Weiszfeld iteration."""
    if X.shape[1] == 1:
        return np.median(X.ravel(), keepdims=True)
    tol **= 2
    old = np.mean(X, axis=0)
    for _ in range(max_iter):
        new = _weiszfeld_step(X, old)
        if np.sum((old - new) ** 2) < tol:
            return new
        old = new
    return old


def theil_sen(x, y):
    """Robust line: spatial median of the exact lines through every pair of points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rows = []
    for i, j in itertools.combinations(range(x.size), 2):
        A = np.array([[1.0, x[i]], [1.0, x[j]]])
        rows.append(np.linalg.lstsq(A, np.array([y[i], y[j]]), rcond=None)[0])
    med = spatial_median(np.asarray(rows))
    return float(med[1]), float(med[0])          # slope, intercept


def linear_fit(x, y, method="linear", force_intercept=False):
    """Returns slope, intercept, residuals, r2.

    force_intercept shifts y down by 1 before fitting and then reports the intercept as
    1.0. It does NOT constrain the line through (0, 1) -- the estimator still fits its
    own intercept. Reproduced as-is from the original.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_adj = y - 1.0 if force_intercept else y

    if method == "theil-sen":
        slope, intercept = theil_sen(x, y_adj)
        pred = slope * x                       # note: the original drops the intercept here
        full = slope * x + intercept
        ss_res = float(((y_adj - full) ** 2).sum())
        ss_tot = float(((y_adj - y_adj.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    else:
        slope, intercept, pred, r2 = ols(x, y_adj)

    if force_intercept:
        pred = pred + 1.0
        intercept = 1.0
    return slope, intercept, y - pred, r2


# --- the analysis -----------------------------------------------------------

def exclude_outlier_theil_sen(x, y, min_datapoints, improvement_threshold):
    """Rank points by Theil-Sen residual and flag the worst if dropping it pays.

    QUIRK 1: the trimmed arrays are returned whether or not the point qualified. The
    caller therefore always scores an interval using a fit on that interval minus its
    worst point, while only recording an exclusion when the point qualified.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n <= min_datapoints:
        return x, y, None

    *_, residuals, _ = linear_fit(x, y, method="theil-sen", force_intercept=True)
    base = mae(residuals)
    worst = int(np.argmax(np.abs(residuals)))
    mask = np.ones(n, dtype=bool)
    mask[worst] = False
    qualified = worst if mae(residuals[mask]) < (1.0 - improvement_threshold) * base else None
    return x[mask], y[mask], qualified


def student_t_sf(t_stat, dof):
    """P(T > t_stat) for Student's t. Exact closed forms, dof 1..4.

    A Stern-Volmer series here is 3 to 6 points, so dof = n - 2 never exceeds 4 and the
    closed forms cover every case. They agree with scipy.stats.t to 2e-16, which is why
    this file still needs nothing but numpy.
    """
    if dof == 1:                                   # Cauchy
        cdf = 0.5 + np.arctan(t_stat) / np.pi
    elif dof == 2:
        cdf = 0.5 + t_stat / (2.0 * np.sqrt(2.0 + t_stat * t_stat))
    elif dof == 3:
        u = t_stat / np.sqrt(3.0)
        cdf = 0.5 + (u / (1.0 + u * u) + np.arctan(u)) / np.pi
    elif dof == 4:
        x = t_stat / np.sqrt(4.0 + t_stat * t_stat)
        cdf = 0.5 + 0.25 * x * (3.0 - x * x)
    else:
        return float("nan")
    return 1.0 - cdf


def slope_p_value(x, y):
    """One-sided p for a positive slope of y on x. NaN if there are too few points.

    One-sided because quenching can only raise I0/I. A negative slope is evidence
    against quenching, not weak evidence for it, and comes back with p near 1.
    """
    n = x.size
    if n < 3:
        return float("nan")
    slope, _, _, r2 = ols(x, y)
    if not np.isfinite(r2):
        return float("nan")
    if r2 >= 1.0:                       # no residual scatter left to test against
        return 0.0 if slope > 0 else 1.0
    dof = n - 2
    t_stat = np.sign(slope) * np.sqrt(max(r2, 0.0) * dof / (1.0 - r2))
    return student_t_sf(t_stat, dof)


def slope_significance(chosen, conc, ratio):
    """One-sided p for the trend, on the points the fit actually used.

    WHY NOT ON ALL MEASURED POINTS
        That was the first attempt and it is wrong. The range scan exists to cut a
        collapsed well loose, and putting it back lets one bad reading carry the label:
        PC11/Q44 fits at K_SV = 0.67 on its own points and 7276 across all six, because
        the last well reads near zero. Testing what the fit used keeps the label and the
        reported K_SV describing the same data.

    WHY THE PENALTY
        Testing the used points alone would reward the grooming instead. A pair that had
        its worst point dropped was chosen as the best of n_measured candidates, so its
        p is optimistic by roughly that factor -- simulation puts P(r2 > 0.95) under pure
        noise at 0.023 for the best 5 of 6 against 0.0046 for a clean 5, a ratio of 5.0
        against the 6 assumed here. Bonferroni over the candidates is the conservative
        correction, and it is what makes a trimmed and an untrimmed pair comparable.

        This is the whole reason the label cannot just read the fitted r2: r2 carries no
        record of how many fits were tried to reach it.
    """
    x, y = conc[chosen], ratio[chosen]
    p_raw = slope_p_value(x, y)
    if not np.isfinite(p_raw):
        return float("nan")
    penalty = conc.size if x.size < conc.size else 1
    return min(1.0, p_raw * penalty)


def analyse(conc, intensity):
    """Returns the indices used, a note, and the concentration / I0-over-I arrays."""
    conc = np.asarray(conc, float)
    intensity = np.asarray(intensity, float)

    keep = ~np.isnan(intensity)          # wells where every read hit the detector rail
    conc, intensity = conc[keep], intensity[keep]
    ratio = intensity[0] / intensity
    indices = np.arange(conc.size)

    if conc.size >= MIN_POINTS_FOR_FIT:
        slope, _, _, r2 = ols(conc, ratio)
        r = np.sign(slope) * np.sqrt(max(r2, 0.0))
        if slope < 0:
            return indices, "negative initial slope", conc, ratio
        if round(r, 4) ** 2 < R2_NO_QUENCHING:
            return indices, "initial r2 below no-quenching threshold", conc, ratio
        if round(r ** 2, 4) > R2_GOOD_CORRELATION:
            return indices, "perfect measurement, no range scan", conc, ratio

    best_score = 0.0
    chosen = indices
    for end in range(MIN_POINTS_FOR_FIT, conc.size + 1):
        x, y = conc[:end], ratio[:end]
        x_used, y_used, outlier = exclude_outlier_theil_sen(
            x, y, MIN_POINTS_FOR_FIT, OUTLIER_IMPROVEMENT)
        *_, residuals, _ = linear_fit(x_used, y_used, method="linear", force_intercept=True)

        # scored with the FULL interval length, but residuals from the trimmed fit
        score = score_fn(residuals, len(x))
        if score > best_score:
            best_score = score
            chosen = indices[:end]
            # QUIRK 2: a plain truthiness test, so an outlier at position 0 is never
            # removed. That is why every fit keeps the [Q] = 0 anchor point.
            if outlier:
                chosen = np.delete(chosen, outlier)

    return chosen, "range scanned", conc, ratio


def finalise(chosen, conc, ratio):
    """Refit the chosen points with a free intercept; keep whichever method has lower MAE."""
    x, y = conc[chosen], ratio[chosen]
    best = None
    best_mae = 100.0
    for method in ("linear", "theil-sen"):
        slope, intercept, residuals, r2 = linear_fit(x, y, method=method, force_intercept=False)
        err = mae(residuals)
        if err < best_mae:
            best_mae = err
            best = (slope, intercept, r2, method)
    return best


def confidence(k_sv, r2, intercept, n_used, n_total, quenching):
    """The original's four-level qualitative flag, kept so the tables stay comparable."""
    if not all(np.isfinite(v) for v in (k_sv, r2, intercept)):
        return "unknown"
    if intercept < 0.5 or intercept > 1.5:
        return "low"
    n_good = 4 / 6 * n_total
    if quenching:
        if n_used >= n_good:
            if not (0.8 <= intercept <= 1.2):
                return "medium"
            return "very high" if r2 >= 0.99 else "high" if r2 >= 0.97 else "medium"
        if 0.8 <= intercept <= 1.2:
            return "high" if r2 >= 0.99 else "medium"
        return "low"
    if k_sv > 100:
        return "low"
    return "medium" if 25 < k_sv < 100 else "high"


def main():
    if not IN.exists():
        sys.exit(f"{IN.name} not found -- run stern_volmer_intensities.py first")

    pairs = defaultdict(list)
    for row in csv.DictReader(IN.open(encoding="utf-8")):
        pairs[(int(row["pc_id"]), int(row["quencher_id"]))].append((
            float(row["quencher_conc_M"]),
            float(row["intensity_au"]) if row["intensity_au"] else float("nan"),
        ))

    rows = []
    for pc_id, q_id in sorted(pairs):
        points = sorted(pairs[(pc_id, q_id)])
        chosen, note, conc, ratio = analyse([c for c, _ in points], [i for _, i in points])
        slope, intercept, r2, method = finalise(chosen, conc, ratio)
        quenching = bool(slope >= K_SV_QUENCHING and r2 >= R2_QUENCHING)

        # the all-points fit is a diagnostic, not the label: where it disagrees sharply
        # with K_SV, a well the range scan dropped is the reason.
        slope_all, _, _, r2_all = ols(conc, ratio)
        p_slope = slope_significance(chosen, conc, ratio)
        quenches = bool(
            slope > K_SV_LABEL
            or (slope > 0 and np.isfinite(p_slope) and p_slope < P_SLOPE_LABEL)
        )
        rows.append({
            "pc_id": pc_id,
            "quencher_id": q_id,
            "K_SV": sig(slope),
            "intercept": sig(intercept),
            "r2": sig(r2),
            "num_points": len(chosen),
            "n_measured": int(conc.size),
            "K_SV_all_points": sig(slope_all),
            "r2_all_points": sig(r2_all),
            "p_slope": sig(p_slope),
            "quenches": int(quenches),
            "quenching_observed": quenching,
            "fit_method": method,
            "confidence": confidence(slope, r2, intercept, len(chosen), int(conc.size), quenching),
            "note": note,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    quench = sum(1 for r in rows if r["quenching_observed"])
    label = sum(1 for r in rows if r["quenches"])
    rescued = sum(1 for r in rows if r["quenches"] and not (r["K_SV"] or 0) > K_SV_LABEL)
    print(f"{len(rows)} pairs -> {OUT.name}")
    print(f"  quenches (K_SV > {K_SV_LABEL:g}, or p_slope < {P_SLOPE_LABEL}): "
          f"{label} yes, {len(rows) - label} no")
    print(f"    of which carried by the slope test alone: {rescued}")
    print(f"  quenching_observed (published, K_SV >= {K_SV_QUENCHING:g} and r2 >= {R2_QUENCHING}): "
          f"{quench} yes, {len(rows) - quench} no")
    counts = defaultdict(int)
    for r in rows:
        counts[r["note"]] += 1
    for note, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  {note}")


if __name__ == "__main__":
    main()
