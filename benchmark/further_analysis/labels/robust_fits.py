# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Every Stern-Volmer pair refitted with thirteen estimators, the blank-well noise of every
photocatalyst, and a per-photocatalyst noise test.

Reads   ../../../data/experimental_data_raw/stern_volmer/ (raw reads)
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
        ../../../data/experimental_data_processed/stern_volmer_fits.csv
        ../../../data/experimental_data_scripts/stern_volmer_fits.py (imported, unchanged)
Writes  results/labels/robust/pc_noise.csv       one row per photocatalyst: noise of the
                                                 48 quencher-free wells and the noise-test
                                                 threshold of every estimator
        results/labels/robust/fits_long.csv.gz   one row per pair and estimator: K_SV,
                                                 intercept, r2, ladder label, noise-test
                                                 labels, case-bootstrap interval and label
                                                 frequencies

    uv run benchmark/further_analysis/labels/robust_fits.py

Blank-well noise. Each quencher-free well is divided by the median of the quencher-free
wells of its plate; CV0 (`well_cv_robust`) is the MAD-based SD (scale 1.4826) of these 48
ratios, `well_cv_sd` their plain SD. Plate-to-plate differences in level are thereby
removed; the scatter between wells and plate positions remains.

Estimators on I0/I against [Q]: CUR, the fit of stern_volmer_fits.py; CUR_TSFIX, the same
points with the Theil-Sen alternative compared on residuals that include the intercept;
OLS; OLS0 through (0, 1); OLS_POOLI0, I0 = median quencher-free well of the plate;
WLS_READS, weights from the read-to-read variance; WLS_REL, weights 1/y^2; HUBER and TUKEY
(IRLS from Theil-Sen, scale floored at sqrt(2) CV0 median(y)); TS, Theil-Sen; RANSAC
(exhaustive, inliers within 3 sqrt(2) CV0 y, at least four); COOK, OLS without the point
of largest Cook's distance if D > F(0.5; 2, n-2); REPS, every read as a point.

Noise test. For each photocatalyst, 2,000 null series are drawn from its own blank-well
ratios at its median concentrations and fitted by every estimator; a pair passes at alpha
when its K_SV exceeds the 1 - alpha quantile (alpha = 0.01, 0.05). Bootstrap: 200
resamples of the wells with [Q] > 0, the [Q] = 0 well kept, per pair. Seeds are fixed per
photocatalyst and pair.
"""

import itertools
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

import common

ALPHAS = (0.01, 0.05)
N_NULL = 2000
N_BOOT = 200
SEED = 20261009
METHODS = ["CUR", "CUR_TSFIX", "OLS", "OLS0", "OLS_POOLI0", "WLS_READS", "WLS_REL",
           "HUBER", "TUKEY", "TS", "RANSAC", "COOK", "REPS"]

svf = common.sv_fits_module()


# ---------------------------------------------------------------- data
def raw_reads():
    """(batch, well) -> list of raw counts."""
    out = {}
    for p in sorted(common.RAW.glob("PC*_QBatch*.csv")):
        for line in common.read_plate_text(p).splitlines():
            parts = [s.strip() for s in line.split(";")]
            if re.fullmatch(r"[A-H](?:[1-9]|1[0-2])", parts[0] or ""):
                vals = [int(s) for s in parts[1:] if s]
                if vals:
                    out[(p.stem, parts[0])] = vals
    return out


def load_pairs():
    s = pd.read_csv(common.INTENSITIES)
    reads = raw_reads()
    s["reads"] = [[r for r in reads[(b, w)] if r < common.DETECTOR_CEILING] for b, w in zip(s.batch, s.well)]
    blanks = s[s.quencher_conc_M == 0].copy()
    blanks["med"] = blanks.groupby("batch").intensity_au.transform("median")
    blanks["d"] = blanks.intensity_au / blanks.med
    pairs = {}
    for (pc, q), g in s.groupby(["pc_old_id", "quencher_id"]):
        g = g.sort_values("quencher_conc_M")
        conc = g.quencher_conc_M.to_numpy(float)
        inten = g.intensity_au.to_numpy(float)
        rd = list(g.reads)
        keep = ~np.isnan(inten)
        batch = g.batch.iloc[0]
        pairs[(pc, q)] = dict(conc=conc[keep], inten=inten[keep],
                              reads=[np.asarray(r, float) for r, k in zip(rd, keep) if k],
                              i0_pool=float(blanks.loc[blanks.batch == batch, "med"].iloc[0]))
    return pairs, blanks


# ---------------------------------------------------------------- estimators
def r2_of(y, pred, w=None):
    w = np.ones_like(y) if w is None else w
    if w.sum() <= 0:
        return np.nan
    ym = np.sum(w * y) / w.sum()
    sst = np.sum(w * (y - ym) ** 2)
    return 1 - np.sum(w * (y - pred) ** 2) / sst if sst > 0 else np.nan


def wls(x, y, w):
    W = w.sum()
    xm, ym = (w * x).sum() / W, (w * y).sum() / W
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan
    b = (w * (x - xm) * (y - ym)).sum() / sxx
    return b, ym - b * xm


def ols_fit(x, y):
    b, a = wls(x, y, np.ones_like(x))
    return b, a, r2_of(y, a + b * x)


def anchored(x, y):
    b, a, _, r2 = svf.ols_anchored(x, y)
    return b, 1.0, r2


def theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    m = dx != 0
    if not m.any():
        return np.nan, np.nan
    b = float(np.median((y[j] - y[i])[m] / dx[m]))
    return b, float(np.median(y - b * x))


def ts_fit(x, y):
    b, a = theil_sen(x, y)
    return b, a, r2_of(y, a + b * x)


def irls(x, y, kind, s_floor, it=50):
    b, a = theil_sen(x, y)
    w = np.ones_like(y)
    for _ in range(it):
        r = y - (a + b * x)
        s = max(stats.median_abs_deviation(r, scale="normal"), s_floor)
        u = r / s
        if kind == "huber":
            c = 1.345
            w_new = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        else:
            c = 4.685
            w_new = np.where(np.abs(u) < c, (1 - (u / c) ** 2) ** 2, 0.0)
        if (w_new > 0).sum() < 3 or len(np.unique(x[w_new > 0])) < 2:
            break
        b_new, a_new = wls(x, y, w_new)
        w = w_new
        if abs(b_new - b) < 1e-9 * max(1, abs(b)) and abs(a_new - a) < 1e-12:
            b, a = b_new, a_new
            break
        b, a = b_new, a_new
    return b, a, r2_of(y, a + b * x, w)


def ransac(x, y, sig):
    """Exhaustive over point pairs; inliers |r| <= 3 sigma_i; most inliers, ties by SSR;
    OLS on the inliers. Needs four inliers, else all points."""
    n = len(x)
    best = None
    for i, j in itertools.combinations(range(n), 2):
        if x[i] == x[j]:
            continue
        b = (y[j] - y[i]) / (x[j] - x[i])
        a = y[i] - b * x[i]
        inl = np.abs(y - (a + b * x)) <= 3 * sig
        if inl.sum() < 4 or len(np.unique(x[inl])) < 2:
            continue
        bb, aa = wls(x[inl], y[inl], np.ones(inl.sum()))
        ssr = float(((y[inl] - aa - bb * x[inl]) ** 2).sum())
        key = (inl.sum(), -ssr)
        if best is None or key > best[0]:
            best = (key, bb, aa, inl)
    if best is None:
        b, a, r2 = ols_fit(x, y)
        return b, a, r2, n
    _, b, a, inl = best
    return b, a, r2_of(y[inl], a + b * x[inl]), int(inl.sum())


def cook_fit(x, y):
    n = len(x)
    b, a, r2 = ols_fit(x, y)
    if n < 5 or not np.isfinite(b):
        return b, a, r2, n
    X = np.column_stack([np.ones(n), x])
    h = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
    r = y - (a + b * x)
    s2 = (r ** 2).sum() / (n - 2)
    if s2 <= 0:
        return b, a, r2, n
    D = r ** 2 / (2 * s2) * h / (1 - h) ** 2
    k = int(np.argmax(D))
    if D[k] > stats.f.ppf(0.5, 2, n - 2):
        m = np.ones(n, bool)
        m[k] = False
        b, a, r2 = ols_fit(x[m], y[m])
        return b, a, r2, n - 1
    return b, a, r2, n


def current(x, y_int):
    """The fit of stern_volmer_fits.py, and the same points with the Theil-Sen alternative
    compared on residuals that include the intercept."""
    chosen, note, conc, ratio = svf.analyse(x, y_int)
    k, a, r2, meth = svf.finalise(chosen, conc, ratio, False)
    xx, yy = conc[chosen], ratio[chosen]
    kl, al, _, r2l = svf.linear_fit(xx, yy, "linear")
    kt, at = svf.theil_sen(xx, yy)
    mae_l = np.mean(np.abs(yy - (al + kl * xx)))
    mae_t = np.mean(np.abs(yy - (at + kt * xx)))
    fixed = (kt, at, r2_of(yy, at + kt * xx), "theil-sen") if mae_t < mae_l else (kl, al, r2l, "linear")
    return (k, a, r2, len(chosen), meth), fixed


def fit_all(conc, inten, reads, i0_pool, cv):
    """method -> (K, intercept, r2, n_used) for one series."""
    out = {}
    x = conc
    if len(x) < 3:
        return {m: (np.nan, np.nan, np.nan, len(x)) for m in METHODS}
    y = inten[0] / inten
    (k, a, r2, nu, _), fx = current(conc, inten)
    out["CUR"] = (k, a, r2, nu)
    out["CUR_TSFIX"] = (fx[0], fx[1], fx[2], nu)
    out["OLS"] = (*ols_fit(x, y), len(x))
    out["OLS0"] = (*anchored(x, y), len(x))
    out["OLS_POOLI0"] = (*ols_fit(x, i0_pool / inten), len(x))
    # read scatter: relative variance of the well mean, floored at the series median
    rv = np.array([(np.var(r, ddof=1) / r.mean() ** 2 / len(r)) if len(r) >= 2 else np.nan for r in reads])
    floor = np.nanmedian(rv) if np.isfinite(rv).any() else cv ** 2
    rv = np.where(np.isfinite(rv), np.maximum(rv, floor), floor)
    var = y ** 2 * (rv[0] + rv)
    var[0] = rv[0] * 2
    b, a0 = wls(x, y, 1 / var)
    out["WLS_READS"] = (b, a0, r2_of(y, a0 + b * x, 1 / var), len(x))
    wr = 1 / np.maximum(y, 1e-6) ** 2
    b, a0 = wls(x, y, wr)
    out["WLS_REL"] = (b, a0, r2_of(y, a0 + b * x, wr), len(x))
    s_floor = np.sqrt(2) * cv * np.median(y)
    out["HUBER"] = (*irls(x, y, "huber", s_floor), len(x))
    out["TUKEY"] = (*irls(x, y, "tukey", s_floor), len(x))
    out["TS"] = (*ts_fit(x, y), len(x))
    out["RANSAC"] = ransac(x, y, np.sqrt(2) * cv * np.maximum(y, 1.0))
    out["COOK"] = cook_fit(x, y)
    r0 = reads[0]
    xs, ys = [], []
    for xi, r in zip(x, reads):
        for kk in range(min(len(r0), len(r))):
            xs.append(xi)
            ys.append(r0[kk] / r[kk])
    xs, ys = np.array(xs), np.array(ys)
    out["REPS"] = (*ols_fit(xs, ys), len(xs)) if len(np.unique(xs)) >= 2 else (np.nan, np.nan, np.nan, 0)
    return out


def ladder(k, r2):
    return int(bool(svf.rung(k, r2)))


# ---------------------------------------------------------------- workers
G = {}


def init(pairs, noise, nulls):
    G["pairs"], G["noise"], G["nulls"] = pairs, noise, nulls


def null_worker(args):
    """Null distribution of every estimator's K for one photocatalyst, wells drawn from its
    blank-well ratios."""
    pc, conc, blank_d, blank_reads, cv, seed = args
    rng = np.random.default_rng(seed)
    ks = {m: [] for m in METHODS}
    for _ in range(N_NULL):
        idx = rng.integers(0, len(blank_d), len(conc))
        f = fit_all(conc, 1e5 * blank_d[idx], [1e5 * blank_reads[i] for i in idx], 1e5, cv)
        for m in METHODS:
            ks[m].append(f[m][0])
    return pc, {m: np.asarray(v) for m, v in ks.items()}


def pair_worker(key):
    pc, q = key
    p = G["pairs"][key]
    cv = G["noise"][pc]
    thr = G["nulls"][pc]
    conc, inten, reads = p["conc"], p["inten"], p["reads"]
    f0 = fit_all(conc, inten, reads, p["i0_pool"], cv)
    rng = np.random.default_rng(SEED + 1000 * pc + q)
    n = len(conc)
    bk = {m: [] for m in METHODS}
    bl = {m: [] for m in METHODS}
    for _ in range(N_BOOT):
        idx = np.concatenate([[0], np.sort(rng.integers(1, n, n - 1))])
        if len(np.unique(conc[idx])) < 3:
            continue
        f = fit_all(conc[idx], inten[idx], [reads[i] for i in idx], p["i0_pool"], cv)
        for m in METHODS:
            bk[m].append(f[m][0])
            bl[m].append(ladder(f[m][0], f[m][2]))
    rows = []
    for m in METHODS:
        k, a, r2, nu = f0[m]
        kb = np.asarray(bk[m], float)
        row = dict(pc_old_id=pc, quencher_id=q, method=m, K_SV=k, intercept=a, r2=r2, n_used=nu,
                   label_ladder=ladder(k, r2),
                   K_lo=np.nanpercentile(kb, 2.5) if kb.size else np.nan,
                   K_hi=np.nanpercentile(kb, 97.5) if kb.size else np.nan,
                   p_ladder_boot=np.mean(bl[m]) if bl[m] else np.nan, n_boot=kb.size)
        for al in ALPHAS:
            kc = thr[(m, al)]
            row[f"K_crit_{al}"] = kc
            row[f"label_stat_{al}"] = int(np.isfinite(k) and k > kc)
            row[f"p_stat_{al}_boot"] = float(np.mean(kb > kc)) if kb.size else np.nan
        rows.append(row)
    return rows


def main():
    pairs, blanks = load_pairs()
    ref = common.fits().set_index(["pc_old_id", "quencher_id"])
    mismatched = 0
    for key, p in pairs.items():
        (k, a, r2, nu, meth), _ = current(p["conc"], p["inten"])
        r = ref.loc[key]
        mismatched += not (np.isclose(svf.sig(k), r.K_SV, rtol=1e-5) and nu == r.num_points
                           and ladder(k, r2) == r.quenches)
    assert mismatched == 0, f"{mismatched} pairs differ from stern_volmer_fits.csv"

    names = common.pc_names()
    noise, null_args, blank_rows = {}, [], []
    for pc, g in blanks.groupby("pc_old_id"):
        g = g.dropna(subset=["d"])
        d = g.d.to_numpy(float)
        cv = float(stats.median_abs_deviation(d, scale="normal"))
        noise[pc] = cv
        rd = [np.asarray(r, float) / m for r, m in zip(g.reads, g.med)]
        read_cv = float(np.median([np.std(r, ddof=1) / r.mean() for r in rd if len(r) >= 2]))
        full = [v["conc"] for (pp, _), v in pairs.items() if pp == pc and len(v["conc"]) == 6]
        conc = np.median(np.vstack(full), axis=0)
        null_args.append((pc, conc, d, rd, cv, SEED + pc))
        blank_rows.append(dict(pc=names[pc], pc_old_id=pc, n_blanks=len(d), well_cv_robust=cv,
                               well_cv_sd=float(np.std(d, ddof=1)), read_cv_median=read_cv, q_max_M=conc.max()))
    with Pool() as pool:
        res = pool.map(null_worker, null_args)
    nulls = {pc: {(m, al): float(np.nanquantile(ks[m], 1 - al)) for m in METHODS for al in ALPHAS}
             for pc, ks in res}
    nb = pd.DataFrame(blank_rows)
    for m in METHODS:
        for al in ALPHAS:
            nb[f"K_crit_{al}_{m}"] = nb.pc_old_id.map(lambda pc: nulls[pc][(m, al)])
    common.write(nb, common.ROBUST / "pc_noise.csv")

    with Pool(initializer=init, initargs=(pairs, noise, nulls)) as pool:
        out = pool.map(pair_worker, sorted(pairs), chunksize=8)
    common.write(pd.DataFrame([r for rows in out for r in rows]), common.ROBUST / "fits_long.csv.gz", gzip=True)


if __name__ == "__main__":
    main()
