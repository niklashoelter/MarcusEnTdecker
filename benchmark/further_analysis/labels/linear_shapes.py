# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Shape of every Stern-Volmer series, linear-range variants of K_SV, inner filter, read
drift, plate noise and failed wells.

Reads   ../../../data/experimental_data_raw/stern_volmer/ (raw reads and plate layouts)
        ../../../data/experimental_data_raw/quencher_absorption/absorption_extinction.csv
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
        ../../../data/experimental_data_processed/stern_volmer_fits.csv
        ../../../data/experimental_data_processed/photocatalyst_properties_exp.xlsx
Writes  results/labels/linear/plate_noise.csv       one row per plate: SD of ln I of the
                                                    quencher-free wells within a column,
                                                    column 7 / column 1, read noise, drift
        results/labels/linear/pair_shapes.csv.gz    one row per series: fits, tests, shape
                                                    class, K_SV variants and their labels
        results/labels/linear/step_drop_wells.csv   wells that read implausibly low

    uv run benchmark/further_analysis/labels/linear_shapes.py

Fit: ln I_i = a - ln(1 + K x_i [+ K2 x_i^2]) by least squares, I0 free, with noise
s_series per plate: the pooled residual SD of the near-flat series of the plate (|K| x_max
< 0.1), or the SD of ln I of the 16 quencher-free wells within a column times the median
ratio of the two where a plate has fewer than eight degrees of freedom.

Shape classes (alpha = 0.01): flat (|z| < 2 for z = K / se), negative slope (z <= -2),
flat but noisy (flat with lack of fit), offset quencher-free well or outlier-dominated
(lack of fit removed by dropping the worst point), upward or downward curvature
(quadratic term), unexplained lack of fit, linear otherwise.

K_SV variants: all points; the largest prefix (at least three points) without lack of fit
or curvature; the points with I0/I <= 2; the initial slope of the quadratic fit; inner-
filter corrected (quencher absorbance at the excitation and emission wavelengths, path
0.47 cm); each read alone (drift). Labels: the ladder of the data set on each variant, and
the significance rule K - 1.96 se > 10 M-1 x (4 mM / [Q]max).

Step-drop wells: I0/I more than five times the linear extrapolation from the previous
point while the previous point shows less than twofold quenching.
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common

ALPHA = 0.01
PATH_CM = 0.47                     # 150 uL in a 96-well plate
LABEL_RUNGS = [(50.0, 0.75), (20.0, 0.80), (10.0, 0.90), (0.0, 0.99)]
ABSORPTION = common.DATA / "experimental_data_raw" / "quencher_absorption" / "absorption_extinction.csv"


def rung_label(k, r2):
    if not (np.isfinite(k) and np.isfinite(r2)):
        return 0
    return int(any(k > a and r2 > b for a, b in LABEL_RUNGS))


def read_raw(path):
    reads, times, date = {}, None, None
    for line in path.read_text(encoding="latin-1").splitlines():
        p = [s.strip() for s in line.split(";")]
        if p[0] == "Time [s]":
            times = [float(x.replace(",", ".")) for x in p[1:] if x]
        if p[0] == "Date":
            m = re.search(r"(\d+\.\d+\.\d+) Time: (\d+:\d+:\d+)", line)
            date = datetime.strptime(" ".join(m.groups()), "%d.%m.%Y %H:%M:%S")
        if re.fullmatch(r"[A-H](?:[1-9]|1[0-2])", p[0] or ""):
            reads[p[0]] = [int(x) for x in p[1:] if x]
    return reads, times, date


def wells():
    """One row per well with the three raw reads, read times and plate layout; pos 1-6 is
    the position in the series (1 = the [Q] = 0 well)."""
    iv = pd.read_csv(common.INTENSITIES)
    rows = []
    for b in iv.batch.unique():
        reads, times, date = read_raw(common.RAW / f"{b}.csv")
        layout = json.loads((common.RAW / f"{b}.well_plate.json").read_text(encoding="utf-8"))
        prep = datetime.strptime(layout["experiment_date"], "%Y-%m-%d_%H-%M-%S")
        for w, r in reads.items():
            rows.append(dict(batch=b, well=w, r1=r[0], r2=r[1], r3=r[2], t2=times[1], t3=times[2],
                             read_start=date, prep_to_read_min=(date - prep).total_seconds() / 60))
    d = iv.merge(pd.DataFrame(rows), on=["batch", "well"], how="left")
    d["row"] = d.well.str[0]
    d["col"] = d.well.str[1:].astype(int)
    d["half"] = np.where(d.col <= 6, "L", "R")
    d["pos"] = np.where(d.col <= 6, d.col, d.col - 6)
    d["I"] = d.intensity_au
    return d.sort_values(["batch", "quencher_id", "pos"])


def epsilon():
    """Quencher epsilon (mM-1 cm-1) per wavelength, minus its minimum over the scan
    (baseline offset), clipped at 0."""
    a = pd.read_csv(ABSORPTION).set_index("quencher_id")
    cols = [c for c in a.columns if c.startswith("epsilon_")]
    wl = np.array([int(c.split("_")[1]) for c in cols])
    e = a[cols].to_numpy(float)
    return wl, np.clip(e - e.min(axis=1, keepdims=True), 0, None), a.index.to_numpy()


def ln_model(p, x, quad):
    u = 1 + p[1] * x + (p[2] * x * x if quad else 0)
    return p[0] - np.log(np.clip(u, 1e-9, None))


def fit(x, lnI, quad=False, p0=None):
    """Least squares in ln I. Returns params, rss."""
    xm = x.max()
    if not quad:
        def rss_k(k):
            r = lnI + np.log1p(k * x)
            return float(((r - r.mean()) ** 2).sum())
        grid = np.concatenate([np.linspace(-0.98 / xm, 0, 20), np.geomspace(1e-2, 3e5, 120)])
        k0 = grid[np.argmin([rss_k(k) for k in grid])]
        i = np.searchsorted(grid, k0)
        a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        k = optimize.minimize_scalar(rss_k, bounds=(a_, b_), method="bounded",
                                     options=dict(xatol=1e-10 * max(1, abs(k0)))).x
        return np.array([float((lnI + np.log1p(k * x)).mean()), k]), rss_k(k)
    if len(x) < 4:
        pl, rl = fit(x, lnI)
        return np.r_[pl, 0.0], rl
    if p0 is None:
        p0 = np.r_[fit(x, lnI)[0], 0.0]

    def resid(p):
        u = 1 + p[1] * x + p[2] * x * x
        pen = np.where(u <= 1e-6, 1e3 * (1e-6 - u + 1), 0)
        return lnI - ln_model(p, x, True) + pen
    sc = np.array([1, max(abs(p0[1]), 10), max(abs(p0[1]), 10) / xm])
    r = optimize.least_squares(resid, p0, x_scale=sc, method="lm", max_nfev=5000)
    return r.x, float((r.fun ** 2).sum())


def se_k(x, p, s):
    J = np.column_stack([np.ones_like(x), -x / (1 + p[1] * x)])
    try:
        return float(np.sqrt(s ** 2 * np.linalg.inv(J.T @ J)[1, 1]))
    except np.linalg.LinAlgError:
        return np.nan


def r2_y(x, lnI, p):
    """r2 of I0/I against [Q], I0 = exp(a), against the line 1 + K x."""
    y = np.exp(p[0] - lnI)
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - (1 + p[1] * x)) ** 2).sum() / sst) if sst > 0 else np.nan


def runs_p(res):
    s = np.sign(res[res != 0])
    n1, n2 = (s > 0).sum(), (s < 0).sum()
    if n1 == 0 or n2 == 0:
        return 1.0
    runs = 1 + (np.diff(s) != 0).sum()
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    if var <= 0:
        return 1.0
    return float(stats.norm.cdf((runs - mu) / np.sqrt(var)))


def plate_noise(d):
    z = d[d.pos == 1].copy()
    z["lnI"] = np.log(z.I)
    rows = []
    for b, g in z.groupby("batch"):
        dev = g.lnI - g.groupby("col").lnI.transform("mean")
        rows.append(dict(batch=b, s_plate=np.sqrt((dev ** 2).sum() / (len(g) - 2)),
                         col7_over_col1=np.exp(g[g.col == 7].lnI.mean() - g[g.col == 1].lnI.mean()),
                         read_cv_median=float(np.median(g[["r1", "r2", "r3"]].std(axis=1) / g[["r1", "r2", "r3"]].mean(axis=1))),
                         I0_median=float(g.I.median()), I0_over_ceiling=float(g.I.median() / common.DETECTOR_CEILING),
                         drift_I0_r3_over_r1=float(np.median(g.r3 / g.r1)),
                         prep_to_read_min=float(g.prep_to_read_min.iloc[0])))
    noise = pd.DataFrame(rows)
    S = dict(zip(noise.batch, noise.s_plate))
    acc = {}
    for (b, q), g in d.groupby(["batch", "quencher_id"], sort=False):
        g = g[g.I.notna()]
        x = g.quencher_conc_M.to_numpy(float)
        p_, r_ = fit(x, np.log(g.I.to_numpy(float)))
        if abs(p_[1]) * x.max() < 0.1 and len(x) >= 4:
            a = acc.setdefault(b, [0.0, 0])
            a[0] += r_
            a[1] += len(x) - 2
    ratio = np.median([np.sqrt(v[0] / v[1]) / S[b] for b, v in acc.items() if v[1] >= 8])
    s_row = {b: (np.sqrt(acc[b][0] / acc[b][1]) if b in acc and acc[b][1] >= 8 else ratio * S[b]) for b in S}
    noise["s_series"] = noise.batch.map(s_row)
    noise["n_flat_df"] = noise.batch.map(lambda b: acc.get(b, [0, 0])[1])
    return noise, s_row


def series(d, s_row):
    f = common.fits()
    lt = common.lifetimes()
    old2new = dict(zip(lt.pc_old_id.astype(int), lt.pc_new_id.astype(int)))
    tau = dict(zip(lt.pc_old_id.astype(int), lt["tau0 / ns"]))
    wl, eps, qidx = epsilon()
    rows = []
    for (b, q), g in d.groupby(["batch", "quencher_id"], sort=False):
        g = g[g.I.notna()]
        pc = int(g.pc_old_id.iloc[0])
        x = g.quencher_conc_M.to_numpy(float)
        I = g.I.to_numpy(float)
        lnI = np.log(I)
        pos = g.pos.to_numpy()
        s = s_row[b]
        n = len(x)
        xm = x.max()
        rec = dict(batch=b, pc_old_id=pc, pc_new_id=old2new[pc], quencher_id=int(q), n=n, s_plate=s,
                   xmax_M=xm, K_crit=10.0 * 4e-3 / xm, half=g.half.iloc[0], row=g.row.iloc[0],
                   y_max_obs=float((I[0] / I).max()))
        # all points
        pL, rssL = fit(x, lnI)
        chi2L = rssL / s ** 2
        resL = (lnI - ln_model(pL, x, False)) / s
        worst = int(np.argmax(np.abs(resL)))
        rec.update(K_L=pL[1], se_L=se_k(x, pL, s), chi2_L=chi2L, p_lof_L=float(stats.chi2.sf(chi2L, n - 2)),
                   r2_L=r2_y(x, lnI, pL), I0_well_resid_sd=float(resL[0]), worst_pos=int(pos[worst]),
                   worst_resid_sd=float(resL[worst]), runs_p=runs_p(resL))
        # quadratic
        pQ, rssQ = fit(x, lnI, quad=True)
        rssQ = min(rssQ, rssL)
        dchi = (rssL - rssQ) / s ** 2
        rec.update(K1_Q=pQ[1], K2_Q=pQ[2], dchi2_quad=dchi, p_quad=float(stats.chi2.sf(dchi, 1)))
        # without the worst point
        m = np.ones(n, bool)
        m[worst] = False
        pO, rssO = fit(x[m], lnI[m])
        pOQ, rssOQ = fit(x[m], lnI[m], quad=True)
        rec.update(K_L_drop1=pO[1], p_lof_drop1=float(stats.chi2.sf(rssO / s ** 2, n - 3)),
                   p_quad_drop1=float(stats.chi2.sf(max(rssO - rssOQ, 0) / s ** 2, 1)))
        # without the last point
        m6 = pos != 6
        if m6.sum() >= 3:
            p6, rss6 = fit(x[m6], lnI[m6])
            p6Q, rss6Q = fit(x[m6], lnI[m6], quad=True)
            rec.update(p_quad_drop_last=float(stats.chi2.sf(max(rss6 - rss6Q, 0) / s ** 2, 1))
                       if m6.sum() >= 4 else np.nan)
        # largest prefix without lack of fit or curvature
        best = None
        for k in range(n, 2, -1):
            pk, rk = fit(x[:k], lnI[:k])
            plof = stats.chi2.sf(rk / s ** 2, k - 2)
            pq = 1.0
            if k >= 4:
                _, rkq = fit(x[:k], lnI[:k], quad=True)
                pq = stats.chi2.sf(max(rk - rkq, 0) / s ** 2, 1)
            if plof > ALPHA and pq > ALPHA:
                best = (k, pk, rk)
                break
        if best is None:
            pk, rk = fit(x[:3], lnI[:3])
            best = (3, pk, rk)
        k, pk, rk = best
        rec.update(n_prefix=k, K_prefix=pk[1], se_prefix=se_k(x[:k], pk, s), r2_prefix=r2_y(x[:k], lnI[:k], pk),
                   prefix_ok=1)
        # I0/I <= 2 (I0 from the all-points fit), at least three points
        ycap = np.exp(pL[0] - lnI)
        kk = max(3, int((ycap <= 2.0).sum()) if np.all(np.diff((ycap <= 2.0).astype(int)) <= 0)
                 else int(np.argmax(ycap > 2.0)) or 3)
        kk = min(kk, n)
        pc_, rc_ = fit(x[:kk], lnI[:kk])
        rec.update(n_cap=kk, K_cap=pc_[1], se_cap=se_k(x[:kk], pc_, s), r2_cap=r2_y(x[:kk], lnI[:kk], pc_))
        # inner filter
        ex, em = int(g.excitation_nm.iloc[0]), int(g.emission_nm.iloc[0])
        qi = np.where(qidx == q)[0][0]
        e_ex, e_em = float(np.interp(ex, wl, eps[qi])), float(np.interp(em, wl, eps[qi]))
        cf = 10 ** ((e_ex + e_em) * x * 1e3 * PATH_CM / 2)
        pI, rI = fit(x, lnI + np.log(cf))
        rec.update(eps_ex=e_ex, eps_em=e_em, A_ex_at_xmax=e_ex * xm * 1e3 * PATH_CM, IFE_CF_at_xmax=float(cf.max()),
                   K_L_ifc=pI[1], r2_L_ifc=r2_y(x, lnI + np.log(cf), pI), K_ifc_spurious=float((cf.max() - 1) / xm))
        # each read alone
        for j in (1, 2, 3):
            rj = g[f"r{j}"].to_numpy(float)
            okj = rj < common.DETECTOR_CEILING
            rec[f"K_read{j}"] = fit(x[okj], np.log(rj[okj]))[0][1] if okj.sum() >= 3 else np.nan
        rows.append(rec)

    P = pd.DataFrame(rows)
    P = P.merge(f.rename(columns={"K_SV": "K_cur", "r2": "r2_cur", "intercept": "b_cur"})[
        ["pc_old_id", "quencher_id", "K_cur", "r2_cur", "b_cur", "num_points", "quenches", "rung", "note"]],
        on=["pc_old_id", "quencher_id"])
    P["tau0_ns"] = P.pc_old_id.map(tau)
    return P


def shape_classes(P):
    z = P.K_L / P.se_L
    sig_lof = P.p_lof_L < ALPHA
    sig_q = P.p_quad < ALPHA
    cls = np.full(len(P), "linear", dtype=object)
    flat = z < 2
    cls[flat & (z <= -2)] = "negative slope (enhancement)"
    cls[flat & (z > -2)] = "flat (no trend)"
    cls[flat & sig_lof] = "flat, noisy/outlier"
    q = ~flat
    outl = sig_lof & (P.p_lof_drop1 > ALPHA) & (P.p_quad_drop1 > ALPHA) & (P.worst_pos.isin([1, 2, 3, 4, 5]))
    cls[q & outl & (P.worst_pos == 1)] = "intercept offset (I0 well)"
    cls[q & outl & (P.worst_pos != 1)] = "outlier-dominated"
    up = q & ~outl & sig_q & (P.K2_Q > 0)
    dn = q & ~outl & sig_q & (P.K2_Q < 0)
    cls[up] = "upward curvature"
    cls[dn] = "downward curvature"
    cls[q & ~outl & ~sig_q & sig_lof] = "lack of fit, unexplained"
    P["shape"] = cls
    P["curv_last_point_driven"] = (up | dn) & (P.p_quad_drop_last > ALPHA)
    P["z_L"] = z
    for k, r in (("L", "r2_L"), ("prefix", "r2_prefix"), ("cap", "r2_cap"), ("L_ifc", "r2_L_ifc")):
        P[f"lab_ladder_{k}"] = [rung_label(a, b) for a, b in zip(P["K_" + k], P[r])]
    P["lab_ladder_Q1"] = [rung_label(a, b) for a, b in zip(P.K1_Q, P.r2_L)]
    P["lab_sig_L"] = ((P.K_L - 1.96 * P.se_L) > P.K_crit).astype(int)
    P["lab_sig_prefix"] = ((P.K_prefix - 1.96 * P.se_prefix) > P.K_crit).astype(int)
    P["lab_point_L"] = (P.K_L > P.K_crit).astype(int)
    return P


def step_drop_wells(d, P):
    meta = P.set_index(["batch", "quencher_id"])
    out = []
    for (b, q), g in d.groupby(["batch", "quencher_id"]):
        g = g[g.I.notna()]
        I = g.I.to_numpy()
        x = g.quencher_conc_M.to_numpy()
        for i in range(2, len(I)):
            y_prev = I[0] / I[i - 1]
            k_prev = (y_prev - 1) / x[i - 1] if x[i - 1] > 0 else 0
            y_exp = 1 + max(k_prev, 0) * x[i]
            y_obs = I[0] / I[i]
            if y_obs > 5 * y_exp and y_prev < 2:
                r = meta.loc[(b, q)]
                out.append(dict(batch=b, well=g.well.iloc[i], pc=f"PC{int(r.pc_new_id)}", quencher_id=q,
                                pos=g.pos.iloc[i], y_obs=y_obs, y_expected=y_exp, K_cur=r.K_cur,
                                quenches=r.quenches, num_points_cur=r.num_points))
    return pd.DataFrame(out)


def main():
    d = wells()
    noise, s_row = plate_noise(d)
    common.write(noise, common.LINEAR / "plate_noise.csv")
    P = shape_classes(series(d, s_row))
    common.write(P, common.LINEAR / "pair_shapes.csv.gz", gzip=True)
    common.write(step_drop_wells(d, P).round(3), common.LINEAR / "step_drop_wells.csv")
    print(P["shape"].value_counts().to_string())


if __name__ == "__main__":
    main()
