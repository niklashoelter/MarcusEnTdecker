# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
QC'd well intensities -> one Stern-Volmer fit per photocatalyst/quencher pair.

Reads   ../experimental_data_processed/stern_volmer_intensities.csv
Writes  ../experimental_data_processed/stern_volmer_fits.csv

I0/I against [Q], 27 x 48 = 1296 pairs. The replicate QC behind those intensities lives
once, in stern_volmer_intensities.py.

    uv run data/experimental_data_scripts/stern_volmer_fits.py [--fix-intercept]

THE INTERCEPT FLOATS
    I0/I = 1 + K_SV [Q] has no free intercept, but the reported fit lets it float because
    the published K_SV values do. 327 pairs land more than 0.05 from 1 and twelve more
    than 0.5; there the intercept is absorbing curvature that belongs in the slope.
    `--fix-intercept` anchors at (0, 1) instead. It moves every K_SV, so it is off.

PROVENANCE
    Port of SternVolmerAnalysis in Auto_Quenching_Backup (stern_volmer.py +
    regression_utils.py), the acquisition-time code behind the published values, with the
    one change below. Otherwise verified against the 1296 workbooks that code wrote: same
    point masks, K_SV to ~1e-13. Its scikit-learn estimators are reimplemented in numpy,
    since the version behind the published numbers was never recorded. Two of its
    behaviours look like defects and are reproduced anyway, marked QUIRK.

THE ONE CHANGE: THE RANGE SCAN RUNS FOR EVERY PAIR
    The original returns before the scan when the all-points r2 is below 0.5 and fits all
    points, so a pair that is linear and then saturates took its K_SV from the whole
    series: old PC23/Q39 gave K_SV 2254 on an intercept of 6.3, where its linear first
    three points give 21774 at r2 0.97, intercept 1.38. That exit is gone. 58 pairs choose
    different points and say so in `note`; the other 1238 are unchanged.

TWO LABELS
    `quenching_observed`, the published criterion K_SV >= 25 and r2 >= 0.90, unchanged.
    It reaches the dataset as `quenches_strict`.

    `quenches`, what this project classifies on: a ladder where weaker quenching demands a
    straighter line -- K_SV > 50 at r2 > 0.75, > 20 at 0.80, > 10 at 0.90, > 0 at 0.99.
    Any one rung is enough and `rung` says which. So no fit worse than r2 0.75 is ever
    called quenching however large its K_SV, and a K_SV of a few M-1 needs an almost
    perfect line. The floor stops a collapsed well from carrying the label; the sliding
    threshold keeps the genuine weak quenchers, monotone 3-4 % drops well above read noise.
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

# This project's label: (K_SV above, r2 above) -- any one rung is enough. Highest K_SV
# first, so the rung a pair is reported under is the most permissive one it clears.
LABEL_RUNGS = [(50.0, 0.75), (20.0, 0.80), (10.0, 0.90), (0.0, 0.99)]

# Anchor the reported fit at (0, 1) instead of letting the intercept float. OFF by
# default: the published K_SV values are free-intercept fits, and turning this on moves
# every number in the table. `--fix-intercept` sets it for one run. See ols_anchored.
FIX_INTERCEPT = False
SIGFIGS = 6

EPS = np.finfo(np.double).eps

COLUMNS = [
    "pc_old_id", "quencher_id", "K_SV", "intercept", "r2", "num_points", "n_measured",
    "K_SV_all_points", "r2_all_points",
    "quenches", "rung", "quenching_observed", "fit_method", "confidence", "note",
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

def ols_anchored(x, y):
    """Least squares through (0, 1): I0/I = 1 + K_SV [Q]. Returns slope, 1.0, pred, r2.

    The real one-parameter fit, which `linear_fit(force_intercept=True)` is not. r2 is
    against the mean of y, as in the free-intercept fit, so the two compare; it goes
    negative where the anchored line is worse than a flat one. Off by default.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sxx = float(x @ x)
    slope = float(x @ (y - 1.0)) / sxx if sxx else float("nan")
    pred = 1.0 + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return slope, 1.0, pred, (1 - ss_res / ss_tot if ss_tot else float("nan"))


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


def rung(k_sv, r2):
    """The most permissive rung of LABEL_RUNGS this fit clears, or "" for none.

    Written to the table, so a row says on what evidence it quenches.
    """
    if not (np.isfinite(k_sv) and np.isfinite(r2)):
        return ""
    for k_min, r2_min in LABEL_RUNGS:
        if k_sv > k_min and r2 > r2_min:
            return f"K_SV > {k_min:g} and r2 > {r2_min:g}"
    return ""


def analyse(conc, intensity):
    """Returns the indices used, a note, and the concentration / I0-over-I arrays."""
    conc = np.asarray(conc, float)
    intensity = np.asarray(intensity, float)

    keep = ~np.isnan(intensity)          # wells where every read hit the detector rail
    conc, intensity = conc[keep], intensity[keep]
    ratio = intensity[0] / intensity
    indices = np.arange(conc.size)
    note = "range scanned"

    if conc.size >= MIN_POINTS_FOR_FIT:
        slope, _, _, r2 = ols(conc, ratio)
        r = np.sign(slope) * np.sqrt(max(r2, 0.0))
        if slope < 0:
            return indices, "negative initial slope", conc, ratio
        if round(r ** 2, 4) > R2_GOOD_CORRELATION:
            return indices, "perfect measurement, no range scan", conc, ratio
        if round(r, 4) ** 2 < R2_NO_QUENCHING:
            # The original returns here with every point kept. See THE ONE CHANGE.
            note = "scanned despite a low all-points r2"

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

    return chosen, note, conc, ratio


def finalise(chosen, conc, ratio, fix_intercept=FIX_INTERCEPT):
    """Refit the chosen points; keep whichever method has lower MAE.

    With fix_intercept the line is anchored at (0, 1) instead and there is nothing to
    choose between -- a one-parameter fit has no robust counterpart here.
    """
    x, y = conc[chosen], ratio[chosen]
    if fix_intercept:
        slope, intercept, _, r2 = ols_anchored(x, y)
        return slope, intercept, r2, "anchored"
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


def main(fix_intercept=FIX_INTERCEPT):
    if not IN.exists():
        sys.exit(f"{IN.name} not found -- run stern_volmer_intensities.py first")
    if fix_intercept:
        print("fitting through (0, 1): K_SV will not match the published table")

    pairs = defaultdict(list)
    for row in csv.DictReader(IN.open(encoding="utf-8")):
        pairs[(int(row["pc_old_id"]), int(row["quencher_id"]))].append((
            float(row["quencher_conc_M"]),
            float(row["intensity_au"]) if row["intensity_au"] else float("nan"),
        ))

    rows = []
    for pc_id, q_id in sorted(pairs):
        points = sorted(pairs[(pc_id, q_id)])
        chosen, note, conc, ratio = analyse([c for c, _ in points], [i for _, i in points])
        slope, intercept, r2, method = finalise(chosen, conc, ratio, fix_intercept)
        quenching = bool(slope >= K_SV_QUENCHING and r2 >= R2_QUENCHING)

        # the all-points fit is a diagnostic, not the label: where it disagrees sharply
        # with K_SV, a well the range scan dropped is the reason.
        slope_all, _, _, r2_all = ols(conc, ratio)
        carried = rung(slope, r2)
        rows.append({
            "pc_old_id": pc_id,
            "quencher_id": q_id,
            "K_SV": sig(slope),
            "intercept": sig(intercept),
            "r2": sig(r2),
            "num_points": len(chosen),
            "n_measured": int(conc.size),
            "K_SV_all_points": sig(slope_all),
            "r2_all_points": sig(r2_all),
            "quenches": int(bool(carried)),
            "rung": carried,
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
    print(f"{len(rows)} pairs -> {OUT.name}")
    print(f"  quenches: {label} yes, {len(rows) - label} no")
    rungs = defaultdict(int)
    for r in rows:
        if r["rung"]:
            rungs[r["rung"]] += 1
    for k_min, r2_min in LABEL_RUNGS:
        name = f"K_SV > {k_min:g} and r2 > {r2_min:g}"
        print(f"    {rungs[name]:5d}  {name}")
    print(f"  quenching_observed (published, K_SV >= {K_SV_QUENCHING:g} and r2 >= {R2_QUENCHING}): "
          f"{quench} yes, {len(rows) - quench} no")
    counts = defaultdict(int)
    for r in rows:
        counts[r["note"]] += 1
    for note, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  {note}")


if __name__ == "__main__":
    main(fix_intercept=FIX_INTERCEPT or "--fix-intercept" in sys.argv[1:])
