# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Unquenchable background, corrected K_SV variants and their labels, and read drift per
quencher.

Reads   results/labels/linear/pair_shapes.csv.gz (linear_shapes.py)
        ../../../data/experimental_data_raw/stern_volmer/ (through linear_shapes.wells)
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
Writes  results/labels/linear/background_fits.csv        one row per strongly quenched
                                                         series (I0/I >= 3): K_SV and the
                                                         background fraction f_bg
        results/labels/linear/pc_background.csv          median f_bg per photocatalyst
        results/labels/linear/pair_variants.csv.gz       one row per series: K_SV, se and
                                                         labels of every variant
        results/labels/linear/variant_label_summary.csv  per variant and rule: quenching
                                                         series and label changes against
                                                         the data set, K_SV shift
        results/labels/linear/drift_by_quencher.csv      median relative change of K_SV
                                                         from read 1 to read 3, quenching
                                                         series per quencher

    uv run benchmark/further_analysis/labels/linear_variants.py

Background: I = A (f + (1 - f) / (1 + K [Q])), f = B / I0, fitted in ln I with the noise
of the plate. f_bg of a photocatalyst is the median over its series with I0/I > 6 and
se(f) < 0.01, clipped at 0.

Variants of K_SV beside those of linear_shapes.py: null-profile corrected (I divided by the
median I/I0 per position of the near-null series of the same photocatalyst and plate
half, leaving the series out; at least four such series), background floor (f fixed at
f_bg of the photocatalyst), and both with the inner-filter correction. The se of each
variant is inflated by sqrt(chi2 / (n - 2)) where that exceeds 1. Labels: the ladder of the
data set and the significance rule K - 1.96 se > 10 M-1 x (4 mM / [Q]max).
variant_label_summary.csv is written with three decimals, background_fits.csv with six.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common
import linear_shapes

PATH_CM = 0.47
VARIANTS = {   # name: (K column, r2 column, se column)
    "current": ("K_cur", "r2_cur", None),
    "all points, I0 free": ("K_L", "r2_L", "se_L_infl"),
    "largest linear prefix": ("K_prefix", "r2_prefix", "se_prefix"),
    "I0/I <= 2 cap": ("K_cap", "r2_cap", "se_cap"),
    "initial slope (quadratic)": ("K1_Q", "r2_L", None),
    "inner-filter corrected": ("K_L_ifc", "r2_L_ifc", "se_L_infl"),
    "null-profile corrected": ("K_null", "r2_null", "se_null"),
    "background floor (per PC)": ("K_bg", "r2_bg", "se_bg"),
    "null + IFE + background": ("K_comb", "r2_comb", "se_comb"),
}


def fit_lin(x, lnI):
    """ln I = a - ln(1 + K x), least squares. Returns (a, K), rss."""
    xm = x.max()

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


def fit_bg_fixed(x, I, s, K0, f):
    """I = A (f + (1 - f)/(1 + K x)) with f fixed. Returns K, se_K, chi2."""
    def res(p):
        A, K = p
        return (np.log(I) - np.log(np.clip(A * (f + (1 - f) / (1 + K * x)), 1, None))) / s
    r = optimize.least_squares(res, [I[0], max(K0, 1.0)], bounds=([0, -0.98 / x.max()], [np.inf, 1e7]),
                               x_scale=[I[0], max(abs(K0), 10)])
    try:
        se = float(np.sqrt(np.linalg.inv(r.jac.T @ r.jac)[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return float(r.x[1]), se, float((r.fun ** 2).sum())


def background_fits(P, d):
    rows = []
    for (b, q), g in d.groupby(["batch", "quencher_id"], sort=False):
        g = g[g.I.notna()]
        x = g.quencher_conc_M.to_numpy(float)
        I = g.I.to_numpy(float)
        rec = P[(P.batch == b) & (P.quencher_id == q)].iloc[0]
        if rec.y_max_obs < 3 or len(x) < 5:
            continue
        s = rec.s_plate

        def res(p):
            A, K, f = p
            return (np.log(I) - np.log(np.clip(A * (f + (1 - f) / (1 + K * x)), 1, None))) / s
        p0 = [I[0], max(rec.K1_Q, 50), 0.01]
        r = optimize.least_squares(res, p0, bounds=([0, 0, -0.2], [np.inf, 1e7, 0.9]), x_scale=[I[0], p0[1], 0.01])
        chi_b = float((r.fun ** 2).sum())
        try:
            cov = np.linalg.inv(r.jac.T @ r.jac)
            se_f, se_K = float(np.sqrt(cov[2, 2])), float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            se_f = se_K = np.nan
        rows.append(dict(batch=b, pc_old_id=rec.pc_old_id, pc_new_id=rec.pc_new_id, quencher_id=q,
                         shape=rec["shape"], y_max_obs=rec.y_max_obs, K_L=rec.K_L, K_cur=rec.K_cur,
                         K_bg=r.x[1], se_K_bg=se_K, f_bg=r.x[2], se_f_bg=se_f,
                         chi2_L=rec.chi2_L, chi2_bg=chi_b, n=len(x),
                         p_lof_bg=float(stats.chi2.sf(chi_b, len(x) - 3))))
    return pd.DataFrame(rows)


def variants(P, d, fpc):
    d = d[d.I.notna()].copy()
    d["rel"] = d.I / d.groupby(["batch", "quencher_id"]).I.transform("first")
    d = d.merge(P[["batch", "quencher_id", "pc_new_id", "K_L", "z_L", "xmax_M", "s_plate", "eps_ex", "eps_em"]],
                on=["batch", "quencher_id"])
    d["null"] = (d.K_L.abs() * d.xmax_M < 0.1) & (d.z_L < 3)
    rows = []
    for (b, q), g in d.groupby(["batch", "quencher_id"], sort=False):
        rec = P[(P.batch == b) & (P.quencher_id == q)].iloc[0]
        pc, half = rec.pc_new_id, rec.half
        x = g.quencher_conc_M.to_numpy(float)
        I = g.I.to_numpy(float)
        pos = g.pos.to_numpy()
        s = rec.s_plate
        ref = d[(d.pc_new_id == pc) & (d.half == half) & d.null & ~((d.batch == b) & (d.quencher_id == q))]
        prof = ref.groupby("pos").rel.median().reindex(pos).fillna(1.0).to_numpy() if len(ref) else np.ones_like(x)
        nnull = ref.groupby(["batch", "quencher_id"]).ngroups
        if nnull < 4:
            prof = np.ones_like(x)
        cf = 10 ** ((rec.eps_ex + rec.eps_em) * x * 1e3 * PATH_CM / 2)
        f = float(fpc.get(pc, 0.0))
        out = dict(batch=b, quencher_id=q, n_null_ref=nnull, null_prof_pos6=float(prof[-1]), f_bg_pc=f)
        for name, Iv, use_bg in (("null", I / prof, False), ("bg", I, True), ("comb", I / prof * cf, True)):
            lnI = np.log(Iv)
            p, rss = fit_lin(x, lnI)
            if use_bg and f > 0:
                K, se, chi = fit_bg_fixed(x, Iv, s, p[1], f)
                A = optimize.minimize_scalar(
                    lambda a: float(((np.log(Iv) - np.log(a * (f + (1 - f) / (1 + K * x)))) ** 2).sum()),
                    bounds=(0.5 * Iv[0], 2 * Iv[0]), method="bounded").x
                y = A * (1 - f) / np.clip(Iv - A * f, 1e-9, None)       # linearized for the ladder r2
                sst = ((y - y.mean()) ** 2).sum()
                r2 = float(1 - ((y - (1 + K * x)) ** 2).sum() / sst) if sst > 0 else np.nan
            else:
                K, se, r2, chi = p[1], linear_shapes.se_k(x, p, s), linear_shapes.r2_y(x, lnI, p), rss / s ** 2
            infl = np.sqrt(max(1.0, chi / max(len(x) - 2, 1)))
            out.update({f"K_{name}": K, f"se_{name}": se * infl, f"r2_{name}": r2, f"chi2_{name}": chi})
        rows.append(out)
    V = P.merge(pd.DataFrame(rows), on=["batch", "quencher_id"])
    V["se_L_infl"] = V.se_L * np.sqrt(np.maximum(1, V.chi2_L / (V.n - 2)))
    for name, (kc, rc, sc) in VARIANTS.items():
        if name == "current":
            V[f"lab|{name}|ladder"] = V.quenches
        else:
            V[f"lab|{name}|ladder"] = [linear_shapes.rung_label(a, r) for a, r in zip(V[kc], V[rc])]
        if sc is not None:
            V[f"lab|{name}|significance"] = ((V[kc] - 1.96 * V[sc]) > V.K_crit).astype(int)
    return V


def label_summary(V):
    rows = []
    cur = V.quenches.astype(bool)
    for c in [c for c in V.columns if c.startswith("lab|")]:
        lab = V[c].astype(bool)
        _, name, rule = c.split("|")
        kc = VARIANTS[name][0]
        both = cur & lab & (V.K_cur > 0) & (V[kc] > 0)
        rows.append(dict(variant=name, label_rule=rule, n_quench=int(lab.sum()),
                         flips_0_to_1=int((~cur & lab).sum()), flips_1_to_0=int((cur & ~lab).sum()),
                         median_log10_K_ratio_vs_current=float(np.median(np.log10(V.loc[both, kc] / V.loc[both, "K_cur"]))),
                         spearman_K_vs_current=float(V.loc[both, [kc, "K_cur"]].corr("spearman").iloc[0, 1])))
    return pd.DataFrame(rows)


def main():
    P = pd.read_csv(common.LINEAR / "pair_shapes.csv.gz")
    d = linear_shapes.wells()
    B = background_fits(P, d)
    common.write(B.round(6), common.LINEAR / "background_fits.csv")
    w = B[(B.y_max_obs > 6) & (B.se_f_bg < 0.01)]
    fpc = w.groupby("pc_new_id").f_bg.median().clip(lower=0)
    pcb = pd.DataFrame({"pc": [f"PC{i}" for i in fpc.index], "pc_new_id": fpc.index,
                        "f_bg_median": fpc.to_numpy(), "n_pairs": w.groupby("pc_new_id").size().to_numpy()})
    common.write(pcb, common.LINEAR / "pc_background.csv")

    V = variants(P, d, fpc)
    common.write(V, common.LINEAR / "pair_variants.csv.gz", gzip=True)
    common.write(label_summary(V).round(3), common.LINEAR / "variant_label_summary.csv")

    q = V[(V.quenches == 1) & (V.K_read1 > 0)].assign(drift=lambda x: x.K_read3 / x.K_read1 - 1)
    drift = q.groupby("quencher_id").drift.agg(["median", "count"]).sort_values("median").reset_index()
    common.write(drift, common.LINEAR / "drift_by_quencher.csv")


if __name__ == "__main__":
    main()
