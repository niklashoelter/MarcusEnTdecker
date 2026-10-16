# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.2"]
# ///
"""
Well intensities -> one Stern-Volmer fit per photocatalyst/quencher pair.

Reads   ../experimental_data_processed/stern_volmer_intensities.csv
Writes  ../experimental_data_processed/stern_volmer_fits.csv

    uv run data/experimental_data_scripts/stern_volmer_fits.py

I0/I is fitted against [Q] for 27 x 48 = 1296 pairs. Photocatalysts carry old ids.

Point selection. If the all-points fit has a negative slope or r2 > 0.995, all points
are used. Otherwise the script scans the first n points (n >= 3), scores each range as
sqrt(n) / (1 + MAE)^4, and may drop one outlier per range, found by Theil-Sen
residuals. The [Q] = 0 point is always kept. The chosen points are refitted by ordinary
least squares and by Theil-Sen, and the fit with the lower MAE is reported.

Intercept. The final fit has a free intercept.

Label. `quenches` is 1 if the fit clears any rung of a ladder in which weaker quenching
needs a straighter line: K_SV > 50 at r2 > 0.75, > 20 at 0.80, > 10 at 0.90, > 0 at 0.99.
`rung` gives the rung. No fit with r2 <= 0.75 is labelled quenching.
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

# (K_SV above, r2 above); any one rung is enough. Checked from the highest K_SV down.
LABEL_RUNGS = [(50.0, 0.75), (20.0, 0.80), (10.0, 0.90), (0.0, 0.99)]

SIGFIGS = 6

EPS = np.finfo(np.double).eps

COLUMNS = [
    "pc_old_id", "quencher_id", "K_SV", "intercept", "r2", "num_points", "n_measured",
    "K_SV_all_points", "r2_all_points",
    "quenches", "rung", "fit_method", "confidence", "note",
]


def sig(x, n=SIGFIGS):
    if x is None or not np.isfinite(x):
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def mae(residuals):
    return float(np.mean(np.abs(np.asarray(residuals))))


def score_fn(residuals, n_points):
    """Range score: favours longer ranges and penalises residuals steeply."""
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

    force_intercept shifts y down by 1 before fitting and reports the intercept as 1.0.
    The estimator still fits its own intercept, so the line is not constrained through
    (0, 1).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_adj = y - 1.0 if force_intercept else y

    if method == "theil-sen":
        slope, intercept = theil_sen(x, y_adj)
        pred = slope * x                       # residuals from the slope only, without intercept
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
    """Find the point with the largest Theil-Sen residual.

    Returns the arrays without that point, and its index if dropping it lowers the MAE
    by at least improvement_threshold (else None). The trimmed arrays are returned in
    both cases, so a range is always scored without its worst point.
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
    """The first rung of LABEL_RUNGS this fit clears, or "" for none."""
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
            note = "scanned despite a low all-points r2"

    best_score = 0.0
    chosen = indices
    for end in range(MIN_POINTS_FOR_FIT, conc.size + 1):
        x, y = conc[:end], ratio[:end]
        x_used, y_used, outlier = exclude_outlier_theil_sen(
            x, y, MIN_POINTS_FOR_FIT, OUTLIER_IMPROVEMENT)
        *_, residuals, _ = linear_fit(x_used, y_used, method="linear", force_intercept=True)

        # full range length, residuals from the trimmed fit
        score = score_fn(residuals, len(x))
        if score > best_score:
            best_score = score
            chosen = indices[:end]
            # index 0 is falsy, so the [Q] = 0 point is always kept
            if outlier:
                chosen = np.delete(chosen, outlier)

    return chosen, note, conc, ratio


def finalise(chosen, conc, ratio):
    """Refit the chosen points by OLS and Theil-Sen and keep the lower MAE."""
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
    """Qualitative fit confidence from intercept, r2, and the fraction of points used."""
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
        sys.exit(f"{IN.name} not found; run stern_volmer_intensities.py first")

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
        slope, intercept, r2, method = finalise(chosen, conc, ratio)

        # all-points fit, reported for comparison only
        slope_all, _, _, r2_all = ols(conc, ratio)
        carried = rung(slope, r2)
        quenching = bool(carried)
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
            "fit_method": method,
            "confidence": confidence(slope, r2, intercept, len(chosen), int(conc.size), quenching),
            "note": note,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

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
    counts = defaultdict(int)
    for r in rows:
        counts[r["note"]] += 1
    for note, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  {note}")


if __name__ == "__main__":
    main()
