# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Stage 1 of the label analysis: the intensity noise model of the Stern-Volmer screen.

Reads   ../../../data/experimental_data_raw/stern_volmer/ (raw reads, through common.py)
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
        ../../../data/experimental_data_processed/photocatalyst_properties_exp.xlsx
Writes  results/labels/noise/noise_model_pc.csv     one row per photocatalyst: read, well
                                                    and residual noise, the fitted
                                                    multiplicative noise sigma_m and the
                                                    global additive term b
        results/labels/noise/position_profile.csv   median ln(I/I0) per position in the
                                                    series and plate block, near-null series

    uv run benchmark/further_analysis/labels/s1_noise_model.py

Every series is fitted as ln I = a - ln(1 + K [Q]) by least squares. The leverage-corrected
residuals of the series with |K [Q]max| < 1 carry the noise model

    r ~ (1 - p_out) N(0, s^2) + p_out N(0, (k_out s)^2),   s^2 = sigma_m,PC^2 + (b / I)^2,

fitted by penalized maximum likelihood (ln sigma_m,PC shrunk to the pooled mean with prior
SD 0.5). The weak series of a photocatalyst all sit near I0, so b is not identified within
a photocatalyst; it is set from the excess scatter of the weakly emitting PC22 over the
median of the others, and sigma_m,PC, p_out and k_out are then fitted with b fixed. Reads
at or above the detector ceiling (260,000 counts) are left out of the read noise.
Written with five decimals.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common

WEAK = 1.0          # series with |K [Q]max| < WEAK carry the noise fit
SHRINK_SD = 0.5     # prior SD of ln sigma_m,PC around the pooled mean
NEAR_NULL = 0.03    # |K [Q]max| below this: near-null series of the position profile
LOW_SIGNAL_PC = 24  # old id of PC22, whose excess scatter sets b


def ln_fits(sv):
    """Per-series fit ln I = a - ln(1 + K [Q]): residual table R and pair table P."""
    rows, fitted = [], []
    for (pc, q), d in sv.groupby(["pc_old_id", "quencher_id"]):
        d = d.sort_values("quencher_conc_M")
        x, y, I = d.quencher_conc_M.to_numpy(), np.log(d.intensity_au.to_numpy()), d.intensity_au.to_numpy()
        kmin = -0.95 / x.max()
        k0 = max(np.polyfit(x, np.exp(y[0] - y), 1)[0], kmin * 0.5)

        def fun(t):
            return y - (t[0] - np.log1p(t[1] * x))
        sol = optimize.least_squares(fun, [y[0], k0], bounds=([-np.inf, kmin], [np.inf, np.inf]))
        a, K = sol.x
        res = fun(sol.x)
        J = np.c_[np.ones_like(x), -x / (1 + K * x)]
        H = J @ np.linalg.pinv(J.T @ J) @ J.T
        h = np.clip(np.diag(H), 0, 0.999)
        for i in range(len(x)):
            rows.append((pc, q, d.batch.iloc[i], d.block.iloc[i], d.pos.iloc[i], x[i], I[i], res[i], h[i]))
        fitted.append((pc, q, K, a, len(x)))
    R = pd.DataFrame(rows, columns=["pc_old_id", "quencher_id", "batch", "block", "pos", "x", "I", "res", "lev"])
    R["rt"] = R.res / np.sqrt(1 - R.lev)
    P = pd.DataFrame(fitted, columns=["pc_old_id", "quencher_id", "K_lnfit", "a", "n"])
    return R, P


class NoiseModel:
    """Penalized likelihood of the residuals; parameters packed as
    (ln sigma_m per PC, ln b, logit p_out, ln(k_out - 1))."""

    def __init__(self, R, pcs, ok):
        self.n_pc = len(pcs)
        self.pidx = R.pc_old_id.map({p: i for i, p in enumerate(pcs)}).to_numpy()
        self.rt, self.I, self.ok = R.rt.to_numpy(), R.I.to_numpy(), ok

    def unpack(self, t):
        n = self.n_pc
        return np.exp(t[:n]), np.exp(t[n]), 1 / (1 + np.exp(-t[n + 1])), 1 + np.exp(t[n + 2])

    def nll(self, t, form="add"):
        sm, b, pout, kout = self.unpack(t)
        if form == "add":
            s = np.sqrt(sm[self.pidx] ** 2 + (b / self.I) ** 2)
        else:                                   # "mult": s = sigma_m,PC
            s = sm[self.pidx] + 0 * self.I
        ll = np.logaddexp(np.log(1 - pout) + stats.norm.logpdf(self.rt, 0, s),
                          np.log(pout) + stats.norm.logpdf(self.rt, 0, kout * s))
        lsm = np.log(sm)
        pen = ((lsm - lsm.mean()) ** 2).sum() / (2 * SHRINK_SD ** 2)
        return -ll[self.ok].sum() + pen


def fit_noise_model(R, pcs, ok, I0_pc):
    """(sigma_m per PC, b, p_out, k_out)."""
    model = NoiseModel(R, pcs, ok)
    n = len(pcs)
    mult = optimize.minimize(model.nll, np.r_[np.full(n, np.log(0.008)), 0.0, -3, np.log(3)],
                             args=("mult",), method="L-BFGS-B")
    sm_mult = model.unpack(mult.x)[0]
    i_low = list(pcs).index(LOW_SIGNAL_PC)
    others = np.delete(sm_mult, [i_low])
    b_fixed = I0_pc[i_low] * np.sqrt(max(sm_mult[i_low] ** 2 - np.median(others) ** 2, 0))

    def nll_fixed_b(t):
        return model.nll(np.r_[t[:n], np.log(b_fixed), t[n:]], "add")

    o = optimize.minimize(nll_fixed_b, np.r_[mult.x[:n], mult.x[-2:]], method="L-BFGS-B")
    return model.unpack(np.r_[o.x[:n], np.log(b_fixed), o.x[n:]])


def per_pc_table(sv, R, pcs, ok, sm, b, pout, kout, x_max, names):
    reads = common.raw_reads()
    reads = reads[reads.counts < common.DETECTOR_CEILING]
    g = reads.groupby(["batch", "well"]).counts
    w = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.size()}).reset_index()
    w = w[w.n >= 2]
    w["cv"] = w.sd / w["mean"]
    w["pc_old_id"] = w.batch.str.extract(r"PC(\d+)_")[0].astype(int)
    p = reads.pivot_table(index=["batch", "well"], columns="read", values="counts")
    drift = np.log(p[2] / p[0]).groupby(level=0).median()

    pidx = R.pc_old_id.map({p_: i for i, p_ in enumerate(pcs)}).to_numpy()
    R = R.assign(s_model=np.sqrt(sm[pidx] ** 2 + (b / R.I) ** 2))
    R["z"] = R.rt / R.s_model
    z0 = sv[sv.pos == 0].assign(li=lambda d: np.log(d.intensity_au))
    zero_within = z0.groupby(["pc_old_id", "batch", "block"]).li.std().groupby("pc_old_id").apply(
        lambda s: np.sqrt((s ** 2).mean()))
    blockdiff = (z0.groupby(["pc_old_id", "batch", "block"]).li.mean().unstack()
                 .pipe(lambda d: d[2] - d[1]).groupby("pc_old_id").mean())
    I0 = sv[sv.pos == 0].groupby("pc_old_id").intensity_au.median().reindex(pcs).to_numpy()
    tab = pd.DataFrame({
        "pc": [names[p_] for p_ in pcs], "pc_old_id": pcs,
        "excitation_nm": sv.groupby("pc_old_id").excitation_nm.first().reindex(pcs).to_numpy(),
        "emission_nm": sv.groupby("pc_old_id").emission_nm.first().reindex(pcs).to_numpy(),
        "x_max / mM": (x_max.reindex(pcs) * 1e3).to_numpy(),
        "I0_median / counts": I0,
        "I_at_xmax_median / counts": sv[sv.pos == 5].groupby("pc_old_id").intensity_au.median().reindex(pcs).to_numpy(),
        "read_cv_median": w.groupby("pc_old_id").cv.median().reindex(pcs).to_numpy(),
        "read_drift_ln(read3/read1)": drift.groupby(
            drift.index.str.extract(r"PC(\d+)_", expand=False).astype(int)).mean().reindex(pcs).to_numpy(),
        "zero_well_sd_within_block": zero_within.reindex(pcs).to_numpy(),
        "zero_well_block2_minus_block1": blockdiff.reindex(pcs).to_numpy(),
        "resid_sd_robust": R.groupby("pc_old_id").rt.apply(
            lambda s: 1.4826 * np.median(np.abs(s))).reindex(pcs).to_numpy(),
        "sigma_m_model": sm,
        "sigma_at_I0_model": np.sqrt(sm ** 2 + (b / I0) ** 2),
        "frac_|z|>3_weak_pairs": R[ok].assign(a=lambda d: d.z.abs() > 3).groupby("pc_old_id").a.mean().reindex(pcs).to_numpy(),
        "z_madsd_weak_pairs": R[ok].groupby("pc_old_id").z.apply(
            lambda s: 1.4826 * np.median(np.abs(s))).reindex(pcs).to_numpy(),
        "n_wells_in_noise_fit": pd.Series(ok).groupby(R.pc_old_id.to_numpy()).sum().reindex(pcs).to_numpy(),
        "n_saturated_reads": sv.groupby("pc_old_id").n_saturated.sum().reindex(pcs).to_numpy(),
    })
    tab["b_additive / counts"] = b
    tab["p_outlier_well"] = pout
    tab["k_outlier"] = kout
    tab["in_benchmark"] = ~tab.pc.eq(common.LEFT_OUT_PC)
    return tab


def position_profile(R, P):
    """Median ln(I/I0) per (block, position) over the near-null series of all photocatalysts."""
    R3 = R.merge(P, on=["pc_old_id", "quencher_id"])
    R3["dl"] = np.log(R3.I) - R3.groupby(["batch", "quencher_id"]).I.transform(lambda s: np.log(s.iloc[0]))
    small = R3[(R3.K_lnfit * R3.groupby("pc_old_id").x.transform("max")).abs() < NEAR_NULL]
    return small.groupby(["block", "pos"]).dl.agg(["median", "mean", "size"]).reset_index()


def main():
    sv = common.intensities().dropna(subset=["intensity_au"])
    names = common.pc_names()
    x_max = sv.groupby("pc_old_id").quencher_conc_M.max()

    R, P = ln_fits(sv)
    pcs = np.sort(R.pc_old_id.unique())
    kx = R.merge(P, on=["pc_old_id", "quencher_id"], how="left").K_lnfit.to_numpy() * \
        R.pc_old_id.map(x_max).to_numpy()
    ok = (R.lev.to_numpy() < 0.98) & (np.abs(kx) < WEAK)
    I0_pc = sv[sv.pos == 0].groupby("pc_old_id").intensity_au.median().reindex(pcs).to_numpy()
    sm, b, pout, kout = fit_noise_model(R, pcs, ok, I0_pc)
    print(f"noise model: b = {b:.0f} counts, p_out = {pout:.3f}, k_out = {kout:.2f}, "
          f"{ok.sum()} wells")

    tab = per_pc_table(sv, R, pcs, ok, sm, b, pout, kout, x_max, names)
    common.write(tab.round(5), common.NOISE / "noise_model_pc.csv")
    common.write(position_profile(R, P), common.NOISE / "position_profile.csv")
    print(tab[["pc", "I0_median / counts", "sigma_m_model", "z_madsd_weak_pairs"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
