# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Stage 2 of the label analysis: posterior of K_SV for every pair, empirical null and the
detection limit of every photocatalyst.

Reads   results/labels/noise/noise_model_pc.csv (s1_noise_model.py)
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
        ../../../data/experimental_data_processed/stern_volmer_fits.csv
        ../../../data/experimental_data_processed/photocatalyst_properties_exp.xlsx
        ../../../data/experimental_data_scripts/stern_volmer_fits.py (imported, unchanged)
Writes  results/labels/noise/pair_posteriors.csv.gz   one row per pair (1,296): posterior
                                                      summaries of K_SV and log10 k_q
        results/labels/noise/detection_limits_pc.csv  one row per photocatalyst: empirical
                                                      null and detection limit K_det

    uv run benchmark/further_analysis/labels/s2_posteriors.py

Likelihood of the wells of one pair (ln I_i at [Q]_i = x_i, xt_i = x_i / x_max):

    ln I_i = a - ln(1 + K x_i) - beta xt_i + e_i,   e_i ~ N(0, phi^2 k_i^2 s_i^2),

with s_i from the noise model of stage 1, k_i = 1 or k_out (outlier sets with up to two
outliers enumerated, prior p_out), a pair-level scale phi on a grid (ln phi half-normal,
SD 0.7), a flat prior on a (integrated analytically) and on K in [-0.95/x_max, 3e5] M-1
(grid uniform in asinh K). beta is a pseudo-slope over the series, N(mu_PC,block, tau^2),
integrated analytically.

Empirical null. A first pass without beta gives K x_max and its statistical SD for every
pair. Per photocatalyst these estimates are fitted as a mixture of a normal null component
N(mu_block, s_i^2 + tau^2), centred separately for the two plate halves, and a log-normal
component for quenching pairs (MAP with priors centred on the pooled fit; Nelder-Mead from
a grid of 48 starts over tau, the null weight and the quencher log-mean, then a Powell
polish). Where the mixture tau exceeds 1.5 times the width of the estimates below the null
centre (at least five such pairs), that width is used; with fewer than five and a mixture
tau above twice the pooled median, the pooled median. For PC25-27 (0.2 mM) the null rests
on the prior. sigma_m of a photocatalyst whose weak series do not identify it (MAD-SD of
the standardized residuals < 0.5, PC25) is the pooled median.

Detection limit: K_det = 3 sqrt(sigma_K,stat^2 + tau^2) for a null pair at the median I0 of
the photocatalyst; log10 k_q,det = log10(K_det / tau0).

Variants of the posterior: [free] the model above; [anchored] I0 taken as exact;
[free_chosen] the model on the wells kept by the range scan of stern_volmer_fits.py
(the measured k_q); [free_stat] without beta. log10 k_q samples tau0 within its error.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common

SEED = 20261009
K_CRIT = 10.0
PHI = np.array([1, 1.5, 2, 3, 5, 8, 13, 20, 35])
G = 3000                                    # K grid points
PRIOR_ONLY_PCS = (8, 28, 29)                # old ids of PC25-27, 0.2 mM: almost no null pairs
LOW_SIGNAL_PC = 24                          # old id of PC22, not in the pooled null

LPHI = -np.log(PHI) ** 2 / (2 * 0.7 ** 2)
LPHI -= np.logaddexp.reduce(LPHI)


class Posterior:
    """Grid posterior of K for one pair under the stage-1 noise model."""

    def __init__(self, nm, rng):
        self.b = float(nm["b_additive / counts"].iloc[0])
        self.pout = float(nm.p_outlier_well.iloc[0])
        self.kout = float(nm.k_outlier.iloc[0])
        self.rng = rng
        self.grids = {}

    def kgrid(self, xmax):
        if xmax not in self.grids:
            u = np.linspace(np.arcsinh(-0.95 / xmax), np.arcsinh(3e5), G)
            self.grids[xmax] = (np.sinh(u), np.cosh(u) * (u[1] - u[0]))    # K, dK (flat prior)
        return self.grids[xmax]

    def outlier_sets(self, n):
        sets = [()] + [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2))
        lp = np.array([len(s) * np.log(self.pout) + (n - len(s)) * np.log(1 - self.pout) for s in sets])
        return sets, lp

    def loglik(self, x, lnI, s, xmax, mu, tau, anchored=False):
        """ln p(data | K) on the K grid; a, beta, phi and the outliers marginalized."""
        K, dK = self.kgrid(xmax)
        xt = x / xmax
        r = lnI[None, :] + np.log1p(np.outer(K, x)) + mu * xt[None, :]
        if anchored:
            r = (r - r[:, :1])[:, 1:]
            s, xt = s[1:], xt[1:]
        n = len(s)
        sets, lps = self.outlier_sets(n)
        out = []
        for oset, lpo in zip(sets, lps):
            k = np.ones(n)
            k[list(oset)] = self.kout
            for phi, lph in zip(PHI, LPHI):
                S = np.diag((phi * k * s) ** 2) + tau ** 2 * np.outer(xt, xt)
                Si = np.linalg.inv(S)
                _, ld = np.linalg.slogdet(S)
                if anchored:
                    q = np.einsum("gi,ij,gj->g", r, Si, r)
                    c = -0.5 * ld
                else:
                    one = np.ones(n)
                    Si1 = Si @ one
                    w = one @ Si1
                    Pm = Si - np.outer(Si1, Si1) / w
                    q = np.einsum("gi,ij,gj->g", r, Pm, r)
                    c = -0.5 * ld - 0.5 * np.log(w)
                out.append(-0.5 * q + c + lpo + lph)
        return K, dK, np.logaddexp.reduce(np.array(out), axis=0)

    def summarise(self, K, dK, ll, kdet, tau0, tau0_err):
        lp = ll + np.log(dK)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        cdf = np.cumsum(p)

        def q(a):
            return float(np.interp(a, cdf, K))
        d = {"K_med": q(0.5), "K_q025": q(0.025), "K_q975": q(0.975), "K_q05": q(0.05), "K_q95": q(0.95),
             "K_mean": float((p * K).sum()), "K_sd": float(np.sqrt((p * (K - (p * K).sum()) ** 2).sum())),
             "P_K>0": float(p[K > 0].sum()), "P_K>10": float(p[K > K_CRIT].sum()),
             "P_K>Kdet": float(p[K > kdet].sum())}
        pp = p * (K > 0)
        if pp.sum() > 1e-12:                    # log10 k_q | K > 0, tau0 error sampled
            pp = pp / pp.sum()
            ks = self.rng.choice(K, size=4000, p=pp)
            ts = self.rng.normal(tau0, tau0_err, 4000) * 1e-9
            lk = np.log10(ks) - np.log10(ts)
            d.update({"log10kq_med": np.median(lk), "log10kq_q025": np.quantile(lk, 0.025),
                      "log10kq_q975": np.quantile(lk, 0.975), "log10kq_sd": lk.std(),
                      "log10kq_q16": np.quantile(lk, 0.16), "log10kq_q84": np.quantile(lk, 0.84)})
        d["K_upper95"] = q(0.95)
        return d


def mix_nll(t, kt, st, blk, prior=None):
    """Mixture of the null N(mu_block, st^2 + tau^2) and a log-normal for quenching pairs."""
    mu1, mu2, ltau, lpi, m, lv = t
    tau = np.exp(ltau)
    pi0 = 1 / (1 + np.exp(-lpi))
    v = np.exp(lv)
    mu = np.where(blk == 1, mu1, mu2)
    ln0 = np.log(pi0) + stats.norm.logpdf(kt, mu, np.sqrt(st ** 2 + tau ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.where(kt > 0, np.log(1 - pi0) + stats.norm.logpdf(np.log(np.maximum(kt, 1e-12)), m, np.sqrt(v))
                      - np.log(np.maximum(kt, 1e-12)), -np.inf)
    nll = -np.logaddexp(ln0, la).sum()
    if prior is not None:
        g1, g2, gt = prior
        nll += (mu1 - g1) ** 2 / (2 * 0.01 ** 2) + (mu2 - g2) ** 2 / (2 * 0.01 ** 2) + (ltau - gt) ** 2 / (2 * 0.5 ** 2)
    nll += (m - 0) ** 2 / (2 * 3 ** 2) + (lv - 0) ** 2 / (2 * 2 ** 2)
    return nll


def fit_mix(d, prior=None):
    """MAP of the mixture: Nelder-Mead from a grid of starts, then a Powell polish."""
    args = (d.Kt.to_numpy(), d.st.to_numpy(), d.block.to_numpy(), prior)
    best = None
    for lt0 in np.log([0.01, 0.03, 0.1, 0.3]):
        for lp0 in (-3.0, 0.0, 3.0):
            for m0 in (-2.0, 0.0, 2.0, 4.0):
                o = optimize.minimize(mix_nll, [0.0, 0.0, lt0, lp0, m0, 0.0], args=args, method="Nelder-Mead",
                                      options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7))
                if best is None or o.fun < best.fun:
                    best = o
    o = optimize.minimize(mix_nll, best.x, args=args, method="Powell",
                          options=dict(maxiter=20000, xtol=1e-8, ftol=1e-10))
    return (o if o.fun < best.fun else best).x


def empirical_null(A, xmax_pc):
    """Per-photocatalyst null (mu per block, tau) on the K x_max scale; adds r_null to A."""
    inb = A.pc_old_id != LOW_SIGNAL_PC
    gx = fit_mix(A[inb & ~A.pc_old_id.isin(PRIOR_ONLY_PCS)])
    prior = (gx[0], gx[1], gx[2])
    print("pooled null: mu_block1 %.4f mu_block2 %.4f tau %.4f pi0 %.2f"
          % (gx[0], gx[1], np.exp(gx[2]), 1 / (1 + np.exp(-gx[3]))))
    en = []
    for pc, d in A.groupby("pc_old_id"):
        t = fit_mix(d, prior)
        xm = xmax_pc[pc]
        tau, pi0 = np.exp(t[2]), 1 / (1 + np.exp(-t[3]))
        mu = np.where(d.block == 1, t[0], t[1])
        l0 = np.log(pi0) + stats.norm.logpdf(d.Kt, mu, np.sqrt(d.st ** 2 + tau ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.where(d.Kt > 0, np.log(1 - pi0)
                          + stats.norm.logpdf(np.log(np.maximum(d.Kt, 1e-12)), t[4], np.sqrt(np.exp(t[5])))
                          - np.log(np.maximum(d.Kt, 1e-12)), -np.inf)
        A.loc[d.index, "r_null"] = np.exp(l0 - np.logaddexp(l0, la))
        en.append({"pc_old_id": pc, "mu_block1_span": t[0], "mu_block2_span": t[1], "tau_span": tau, "pi0": pi0,
                   "mu_block1 / M-1": t[0] / xm, "mu_block2 / M-1": t[1] / xm, "tau / M-1": tau / xm,
                   "stat_sd_K_median / M-1": (d.st / xm).median(), "n_pairs": len(d)})
    EN = pd.DataFrame(en).set_index("pc_old_id")
    # Negative K cannot be quenching: the spread below the null centre measures the null width
    # without weak quenchers.
    EN["tau_span_mixture"] = EN.tau_span
    EN["tau_source"] = "mixture"
    pooled_tau = EN.loc[~EN.index.isin([*PRIOR_ONLY_PCS, LOW_SIGNAL_PC]), "tau_span"].median()
    for pc, d in A.groupby("pc_old_id"):
        mu = np.where(d.block == 1, EN.loc[pc, "mu_block1_span"], EN.loc[pc, "mu_block2_span"])
        lm = (d.Kt - mu) < 0
        left = (d.Kt - mu)[lm]
        st_med = d.st[lm].median()
        if len(left) >= 5:
            t_left = np.sqrt(max((1.4826 * np.median(np.abs(left))) ** 2 - st_med ** 2, 0))
            EN.loc[pc, "tau_span_left_tail"] = t_left
            if EN.loc[pc, "tau_span"] > 1.5 * t_left:
                EN.loc[pc, ["tau_span", "tau_source"]] = [t_left, "left tail (mixture inflated)"]
        elif EN.loc[pc, "tau_span"] > 2 * pooled_tau and pc not in PRIOR_ONLY_PCS:
            EN.loc[pc, ["tau_span", "tau_source"]] = [pooled_tau, "pooled median (mixture inflated, < 5 left points)"]
        if pc in PRIOR_ONLY_PCS:
            EN.loc[pc, "tau_source"] = "mixture, essentially prior (almost no null pairs at 0.2 mM)"
    EN["tau / M-1"] = EN.tau_span / EN.index.map(xmax_pc)
    EN["n_left_of_null"] = [
        int(((d.Kt - np.where(d.block == 1, EN.loc[pc, "mu_block1_span"], EN.loc[pc, "mu_block2_span"])) < 0).sum())
        for pc, d in A.groupby("pc_old_id")]
    return EN


def detection_limits(sv, nm, b, EN, xmax_pc, lt):
    x_pc = {pc: np.sort(sv[sv.pc_old_id == pc].quencher_conc_M.unique()) for pc in xmax_pc.index}
    I0_pc = sv[sv.pos == 0].groupby("pc_old_id").intensity_au.median()
    det = []
    for pc in xmax_pc.index:
        x = x_pc[pc]
        s = np.sqrt(nm.loc[pc, "sigma_m_used"] ** 2 + (b / I0_pc[pc]) ** 2)
        X = np.c_[np.ones_like(x), -x]
        stat = np.sqrt(np.linalg.inv(X.T @ X / s ** 2)[1, 1])
        tot = np.sqrt(stat ** 2 + (EN.loc[pc, "tau_span"] / xmax_pc[pc]) ** 2)
        det.append({"pc_old_id": pc, "sd_K_null_stat / M-1": stat, "sd_K_null_total / M-1": tot,
                    "K_det_3sd / M-1": 3 * tot, "K_det_3sd_stat_only / M-1": 3 * stat,
                    "K_95power / M-1": (3 + 1.645) * tot,
                    "log10_kq_det / M-1 s-1": np.log10(3 * tot / (lt.loc[pc, "tau0 / ns"] * 1e-9))})
    return pd.DataFrame(det).set_index("pc_old_id")


def main():
    rng = np.random.default_rng(SEED)
    svf = common.sv_fits_module()
    sv = common.intensities()
    nm = pd.read_csv(common.NOISE / "noise_model_pc.csv").set_index("pc_old_id")
    pooled = nm.loc[nm.index != LOW_SIGNAL_PC, "sigma_m_model"].median()
    nm["sigma_m_used"] = np.where(nm.z_madsd_weak_pairs < 0.5, pooled, nm.sigma_m_model)
    nm["sigma_m_source"] = np.where(nm.z_madsd_weak_pairs < 0.5, "pooled median (not identified)", "per PC")
    post = Posterior(nm, rng)
    fits = common.fits().set_index(["pc_old_id", "quencher_id"])
    lt = common.lifetimes().set_index("pc_old_id")
    xmax_pc = sv.groupby("pc_old_id").quencher_conc_M.max()
    pairs = sorted(sv.groupby(["pc_old_id", "quencher_id"]).groups)
    block_of = sv[sv.pos == 0].set_index(["pc_old_id", "quencher_id"]).block

    def pair_data(pc, q, chosen_only=False):
        d = sv[(sv.pc_old_id == pc) & (sv.quencher_id == q)].sort_values("quencher_conc_M")
        d = d.dropna(subset=["intensity_au"])
        x, I = d.quencher_conc_M.to_numpy(), d.intensity_au.to_numpy()
        if chosen_only:
            idx, *_ = svf.analyse(x, I)
            x, I = x[idx], I[idx]
        s = np.sqrt(nm.loc[pc, "sigma_m_used"] ** 2 + (post.b / I) ** 2)
        return x, np.log(I), s

    # first pass without the empirical null
    rows = []
    for pc, q in pairs:
        x, lnI, s = pair_data(pc, q)
        K, dK, ll = post.loglik(x, lnI, s, xmax_pc[pc], 0.0, 0.0)
        d = post.summarise(K, dK, ll, np.inf, lt.loc[pc, "tau0 / ns"], lt.loc[pc, "tau0_err / ns"])
        rows.append({"pc_old_id": pc, "quencher_id": q, "block": block_of[(pc, q)], **d})
    A = pd.DataFrame(rows)
    A["Kt"] = A.K_med * A.pc_old_id.map(xmax_pc)
    A["st"] = (A.K_q975 - A.K_q025) / 3.92 * A.pc_old_id.map(xmax_pc)

    EN = empirical_null(A, xmax_pc)
    DET = detection_limits(sv, nm, post.b, EN, xmax_pc, lt)

    rows = []
    for pc, q in pairs:
        blk = block_of[(pc, q)]
        mu = EN.loc[pc, "mu_block1_span" if blk == 1 else "mu_block2_span"]
        tau = EN.loc[pc, "tau_span"]
        kdet = DET.loc[pc, "K_det_3sd / M-1"]
        t0, te = lt.loc[pc, "tau0 / ns"], lt.loc[pc, "tau0_err / ns"]
        rec = {"pc_old_id": pc, "pc": f"PC{int(lt.loc[pc, 'pc_new_id'])}", "quencher_id": q, "block": blk,
               "K_det / M-1": kdet}
        x, lnI, s = pair_data(pc, q)
        for var, kw in (("free", dict(mu=mu, tau=tau)), ("anchored", dict(mu=mu, tau=tau, anchored=True))):
            K, dK, ll = post.loglik(x, lnI, s, xmax_pc[pc], **kw)
            d = post.summarise(K, dK, ll, kdet, t0, te)
            rec.update({f"{k} [{var}]": v for k, v in d.items()})
        xc, lnIc, sc = pair_data(pc, q, chosen_only=True)
        K, dK, ll = post.loglik(xc, lnIc, sc, xmax_pc[pc], mu=mu, tau=tau)
        d = post.summarise(K, dK, ll, kdet, t0, te)
        rec.update({f"{k} [free_chosen]": v for k, v in d.items()})
        st = A[(A.pc_old_id == pc) & (A.quencher_id == q)].iloc[0]
        rec.update({"P_K>0 [free_stat]": st["P_K>0"], "P_K>10 [free_stat]": st["P_K>10"],
                    "K_med [free_stat]": st.K_med, "r_null_empirical": st.r_null})
        f = fits.loc[(pc, q)]
        rec.update({"K_SV_dataset": f.K_SV, "intercept_dataset": f.intercept, "r2_dataset": f.r2,
                    "num_points_dataset": f.num_points, "quenches_dataset": f.quenches, "rung_dataset": f.rung})
        rows.append(rec)
    PP = pd.DataFrame(rows)

    common.write(PP, common.NOISE / "pair_posteriors.csv.gz", gzip=True)
    EN2 = EN.join(DET).join(nm[["pc", "sigma_m_used", "sigma_m_source", "I0_median / counts", "x_max / mM"]])
    EN2.insert(0, "pc", EN2.pop("pc"))
    common.write(EN2.reset_index(), common.NOISE / "detection_limits_pc.csv")
    print(EN2[["pc", "tau / M-1", "pi0", "sd_K_null_stat / M-1", "K_det_3sd / M-1",
               "log10_kq_det / M-1 s-1"]].round(2).to_string())


if __name__ == "__main__":
    main()
