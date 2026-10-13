# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Stage 4 of the label analysis: label confidence, benchmark metrics under label
uncertainty, measurement noise in the rate parity, and a censored likelihood of the rates.

Reads   results/labels/noise/pair_posteriors.csv.gz (s2_posteriors.py)
        results/labels/noise/ladder_power_curves.csv (s3_ladder_power.py)
        ../../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/labels/noise/label_confidence_counts.csv      label classes per scope,
                                                              probability and label
        results/labels/noise/disagreement_vs_uncertainty.csv  share of uncertain labels
                                                              among the pairs a criterion
                                                              gets right and wrong
        results/labels/noise/metrics_label_uncertainty.csv    metrics per criterion and
                                                              label setting
        results/labels/noise/pair_labels_probabilistic.csv    one row per scored pair:
                                                              P(K > K_det), P(K > 10) and
                                                              label classes
        results/labels/noise/rate_parity_noise.csv            measurement noise against the
                                                              rate-parity error
        results/labels/noise/censored_likelihood.csv          censored fits of the measured
                                                              rates on the predicted ones

    uv run benchmark/further_analysis/labels/s4_downstream.py

Label classes: confident quench P >= 0.95, confident no quench P <= 0.05, uncertain
otherwise, with P = P(K > K_det,PC) of the [free] posterior (or P(K > 10 M-1), P(K > 0)).
Criteria at the benchmark settings (classification.py): adiabatic (DFT, ML), DvTE p = 0.008
normal quantile, two-gate (DFT, ML) with the per-pair contact distance. Label settings:
the labels of the data set on all scored pairs and on the confident pairs; posterior labels
P >= 0.95 on the confident pairs; the expectation over 2,000 Bernoulli draws of the labels
(with 2.5 and 97.5 % quantiles); posterior-median labels P >= 0.5.

Rates. Predicted log10 k_q from the full rate law of the two-gate criterion (asymmetric
barrier, Eyring k_-d/nu at the contact distance of the pair), as in rate_predictions.py.
Measured log10 k_q = log10(K_SV / tau0) of the quenching pairs; its SD combines the
posterior SD on the wells the range scan kept and the SD over four analyses of the same
wells (K_SV, K_SV / intercept, posterior median on all wells and on the kept wells).

Censored likelihood. The latent log10 k_q of a pair is N(a + g * pred, sigma^2). A
quenching pair contributes P(label 1 | K) N(measured; latent, sd_meas^2), a non-quenching
pair P(label 0 | K), from the simulated ladder power curve of its photocatalyst (stage 3),
integrated over the latent value by 60-point Gauss-Hermite quadrature. Fitted by
Nelder-Mead from a grid of starts over a, g and sigma, then a Powell polish.

Metric, disagreement, rate-noise and likelihood tables are written with three decimals,
pair_labels_probabilistic.csv with four.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common  # puts benchmark/ on the path
import classification
import pairs
import theory

SEED = 11
N_DRAW = 2000
P_DET, P_10 = "P_K>Kdet [free]", "P_K>10 [free]"
LO, HI = 0.05, 0.95
CRITERIA = {   # name: setting of classification.py; the source is in SOURCE
    "adiabatic dft": classification.Setting("adiabatic"),
    "adiabatic ml": classification.Setting("adiabatic"),
    "DvTE (p = 0.008, normal)": classification.Setting("dvte", p="0.008", dvte_kind="normal"),
    "two-gate dft": classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                           k_sv_crit=classification.K_SV_CRIT, contact_distance=None),
    "two-gate ml": classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                          k_sv_crit=classification.K_SV_CRIT, contact_distance=None),
}
SOURCE = {"adiabatic dft": "dft", "adiabatic ml": "ml", "DvTE (p = 0.008, normal)": "dft",
          "two-gate dft": "dft", "two-gate ml": "ml"}
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)
GH_W = GH_W / GH_W.sum()


def classes(p, lo=LO, hi=HI):
    return np.where(p >= hi, "confident quench", np.where(p <= lo, "confident no quench", "uncertain"))


def load_posteriors():
    pp = pd.read_csv(common.NOISE / "pair_posteriors.csv.gz")
    pp["in_benchmark_PC"] = pp.pc != common.LEFT_OUT_PC
    return pp


def scored_pairs(pp):
    """The scored pairs of the benchmark with their posteriors, in benchmark order."""
    S = pairs.scored(pairs.load())
    S = S.merge(pp.drop(columns=["pc"]), on=["pc_old_id", "quencher_id"], how="left", validate="one_to_one")
    assert (S[pairs.QUENCHES] == S.quenches_dataset).all()
    return S


def criteria(S):
    """name -> (predicted quenching, continuous score) at the benchmark settings."""
    return {k: tuple(np.asarray(v) for v in classification.predict(S, s, SOURCE[k]))
            for k, s in CRITERIA.items()}


def metrics_draws(Y, pred, score, groups):
    """Metrics for every row (label draw) of Y."""
    Y = np.atleast_2d(Y).astype(bool)
    pred = pred.astype(bool)
    tp = (Y & pred).sum(1)
    tn = (~Y & ~pred).sum(1)
    fp = (~Y & pred).sum(1)
    fn = (Y & ~pred).sum(1)
    acc = (tp + tn) / Y.shape[1]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn).astype(float))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    r = stats.rankdata(score)
    npos = Y.sum(1)
    nneg = Y.shape[1] - npos
    auc = ((Y * r).sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
    conc = np.zeros(Y.shape[0])
    w = np.zeros(Y.shape[0])
    for g in np.unique(groups):
        m = groups == g
        rg = stats.rankdata(score[m])
        Yg = Y[:, m]
        npg = Yg.sum(1)
        nng = m.sum() - npg
        conc += (Yg * rg).sum(1) - npg * (npg + 1) / 2
        w += npg * nng
    return {"accuracy": acc, "MCC": mcc, "ROC_AUC": auc, "AUC_within_PC_pooled": conc / w}


def confidence_counts(pp, S):
    out = []
    for scope, d in (("all 1296 pairs", pp), ("26 benchmark PCs (PC22 out)", pp[pp.in_benchmark_PC]),
                     ("scored pairs", S)):
        for pname, col in (("P(K > K_det,PC)", P_DET), ("P(K > 10 M-1)", P_10), ("P(K > 0)", "P_K>0 [free]"),
                           ("P(K > K_det,PC), anchored", "P_K>Kdet [anchored]")):
            c = classes(d[col].to_numpy())
            for lab in (0, 1):
                m = d.quenches_dataset.to_numpy() == lab
                out.append({"scope": scope, "probability": pname, "label": lab, "n": int(m.sum()),
                            "confident quench": int((c[m] == "confident quench").sum()),
                            "uncertain": int((c[m] == "uncertain").sum()),
                            "confident no quench": int((c[m] == "confident no quench").sum())})
    return pd.DataFrame(out)


def disagreement(CR, Yr, pdet):
    unc = classes(pdet) == "uncertain"
    rows = []
    for k, (a, s) in CR.items():
        dg = a != Yr
        tab = [[(unc & dg).sum(), (~unc & dg).sum()], [(unc & ~dg).sum(), (~unc & ~dg).sum()]]
        orr, pval = stats.fisher_exact(tab)
        rows.append({"criterion": k, "n_disagree": int(dg.sum()), "frac_uncertain_among_disagree": unc[dg].mean(),
                     "frac_uncertain_among_agree": unc[~dg].mean(), "odds_ratio": orr, "fisher_p": pval,
                     "disagree_rate_uncertain": dg[unc].mean(), "disagree_rate_confident": dg[~unc].mean(),
                     "n_FP_uncertain": int((a & ~Yr & unc).sum()), "n_FN_uncertain": int((~a & Yr & unc).sum())})
    return pd.DataFrame(rows)


def label_metrics(CR, Yr, grp, pdet, p10, rng):
    rows = []
    conf_det = (pdet <= LO) | (pdet >= HI)
    conf_10 = (p10 <= LO) | (p10 >= HI)
    D_det = rng.random((N_DRAW, len(Yr))) < pdet[None, :]
    D_10 = rng.random((N_DRAW, len(Yr))) < p10[None, :]
    for k, (a, s) in CR.items():
        def add(setting, n, m, ci=False):
            rec = {"criterion": k, "setting": setting, "n": n}
            for mk, v in m.items():
                rec[mk] = float(np.mean(v))
                if ci:
                    rec[mk + "_q025"] = float(np.quantile(v, 0.025))
                    rec[mk + "_q975"] = float(np.quantile(v, 0.975))
            rows.append(rec)
        add("data-set labels, all scored", len(Yr), metrics_draws(Yr[None], a, s, grp))
        add("data-set labels, confident subset (K_det)", conf_det.sum(),
            metrics_draws(Yr[conf_det][None], a[conf_det], s[conf_det], grp[conf_det]))
        add("posterior labels P>=0.95, confident subset (K_det)", conf_det.sum(),
            metrics_draws((pdet[conf_det] >= HI)[None], a[conf_det], s[conf_det], grp[conf_det]))
        add("posterior labels P>=0.95, confident subset (K=10)", conf_10.sum(),
            metrics_draws((p10[conf_10] >= HI)[None], a[conf_10], s[conf_10], grp[conf_10]))
        add("expected over label draws, truth K > K_det,PC", len(Yr), metrics_draws(D_det, a, s, grp), ci=True)
        add("expected over label draws, truth K > 10", len(Yr), metrics_draws(D_10, a, s, grp), ci=True)
        add("posterior-median labels (P_det >= 0.5)", len(Yr), metrics_draws((pdet >= 0.5)[None], a, s, grp))
        add("posterior-median labels (P_10 >= 0.5)", len(Yr), metrics_draws((p10 >= 0.5)[None], a, s, grp))
    return pd.DataFrame(rows)


def rate_frame(S):
    """Per scored pair: measured and predicted log10 k_q, and the measurement SD of the
    quenching pairs. Returns (V, Q): all scored pairs, quenching pairs."""
    kq = S[pairs.K_Q].to_numpy(float)
    V = S[[pairs.PC_NAME, pairs.Q_ID, pairs.QUENCHES, pairs.TAU0, "pc_old_id"]].copy()
    V["log10 k_q measured"] = np.log10(np.where(kq > 0, kq, np.nan))
    escape = theory.escape_over_frequency(pairs.contact_distances(S))
    for src in pairs.SOURCES:
        dG = pairs.driving_force(S, src)
        barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(S, src)))
        V[f"dG_{src} / kcal mol-1"] = dG
        V[f"log10 k_q pred {src}"] = np.log10(theory.rate(dG, barrier, escape))
        V[f"log10 k_q pred adiabatic {src}"] = np.log10(theory.sandros_rate(dG))
    for c in ("K_SV_dataset", "intercept_dataset", "K_med [free]", "K_med [free_chosen]",
              "log10kq_sd [free_chosen]"):
        V[c] = S[c].to_numpy()
    Q = V[V[pairs.QUENCHES] == 1].copy()
    lt = np.log10(Q[pairs.TAU0] * 1e-9)
    choices = np.c_[np.log10(Q.K_SV_dataset), np.log10(Q.K_SV_dataset / Q.intercept_dataset),
                    np.log10(Q["K_med [free]"].clip(lower=1e-3)),
                    np.log10(Q["K_med [free_chosen]"].clip(lower=1e-3))] - lt.to_numpy()[:, None]
    Q["sd_analysis_choice"] = choices.std(1)
    Q["sd_stat"] = Q["log10kq_sd [free_chosen]"]
    Q["sd_meas_total"] = np.sqrt(Q.sd_stat ** 2 + Q.sd_analysis_choice ** 2)
    return V, Q


def rate_parity_noise(Q):
    meas = Q["log10 k_q measured"].to_numpy()
    var_obs = meas.var(ddof=1)
    rows = []
    for name, sdcol in (("statistical (noise model)", "sd_stat"), ("statistical + analysis choice", "sd_meas_total")):
        vm = (Q[sdcol] ** 2).mean()
        rec = {"measurement_sd_definition": name, "n_quenching_pairs": len(Q),
               "median_pair_sd / decades": Q[sdcol].median(), "rms_pair_sd / decades": np.sqrt(vm),
               "SD_measured_log10kq / decades": np.sqrt(var_obs),
               "SD_true_log10kq_deconvolved / decades": np.sqrt(max(var_obs - vm, 0)),
               "reliability (1 - noise var / observed var)": 1 - vm / var_obs}
        for src in pairs.SOURCES:
            p_ = Q[f"log10 k_q pred {src}"].to_numpy()
            e = p_ - meas
            mse = np.mean(e ** 2)
            rec[f"RMSE_{src} / decades"] = np.sqrt(mse)
            rec[f"frac_MSE_from_noise_{src}"] = vm / mse
            rec[f"RMSE_{src}_noise_removed / decades"] = np.sqrt(max(mse - vm, 0))
            lo, hi = meas - 1.96 * Q[sdcol], meas + 1.96 * Q[sdcol]
            rec[f"frac_pred_inside_95pct_meas_{src}"] = np.mean((p_ >= lo) & (p_ <= hi))
            rec[f"chi2_per_pair_{src}"] = np.mean(e ** 2 / Q[sdcol] ** 2)
        rec["RMSE_constant / decades"] = np.sqrt(np.mean((meas - meas.mean()) ** 2))
        rows.append(rec)
    n, n_pc = len(Q), Q[pairs.PC_NAME].nunique()
    wpc = Q.groupby(pairs.PC_NAME)["log10 k_q measured"].transform(lambda s: s - s.mean())
    sd_within = np.sqrt(np.mean(wpc ** 2) * n / (n - n_pc))
    rows.append({"measurement_sd_definition": "within-PC (PC means removed), statistical + analysis choice",
                 "n_quenching_pairs": n, "SD_measured_log10kq / decades": sd_within,
                 "rms_pair_sd / decades": np.sqrt((Q.sd_meas_total ** 2).mean()),
                 "SD_true_log10kq_deconvolved / decades": np.sqrt(max(sd_within ** 2 - (Q.sd_meas_total ** 2).mean(), 0))})
    return pd.DataFrame(rows)


class CensoredLikelihood:
    """-ln L of mu = a + g * pred (slope_free) or a + pred over all scored pairs."""

    def __init__(self, V, Q, power_curves):
        self.curves = {pc: (d["K_true / M-1"].to_numpy(), d.P_label_1.to_numpy())
                       for pc, d in power_curves.groupby("pc_old_id")}
        self.yq = V[pairs.QUENCHES].to_numpy() == 1
        self.meas = V["log10 k_q measured"].to_numpy()
        self.tau = V[pairs.TAU0].to_numpy() * 1e-9
        self.pcs = V.pc_old_id.to_numpy().astype(int)
        self.sdm = np.zeros(len(V))
        keys = list(zip(V[pairs.PC_NAME][self.yq], V[pairs.Q_ID][self.yq]))
        self.sdm[self.yq] = Q.set_index([pairs.PC_NAME, pairs.Q_ID]).loc[keys, "sd_meas_total"].to_numpy()

    def power(self, pc, K):
        k, p = self.curves[pc]
        return np.interp(K, k, p, left=p[0], right=p[-1])

    def nll(self, theta, mu_pred, slope_free):
        if slope_free:
            a, g, ls = theta
            mu = a + g * mu_pred
        else:
            a, ls = theta
            mu = a + mu_pred
        s = np.exp(ls)
        latent = mu[:, None] + s * GH_X[None, :]
        K = 10 ** latent * self.tau[:, None]
        pw = np.vstack([self.power(pc, K[i]) for i, pc in enumerate(self.pcs)])
        pw = np.clip(pw, 1e-6, 1 - 1e-6)
        like = np.where(self.yq[:, None],
                        pw * stats.norm.pdf(self.meas[:, None], latent, np.maximum(self.sdm[:, None], 1e-3)),
                        1 - pw)
        return -np.log((like * GH_W[None, :]).sum(1) + 1e-300).sum()

    def fit(self, mu_pred, slope_free):
        best = None
        g_starts = (0.2, 0.3, 0.5, 0.7, 1.0) if slope_free else (None,)
        for a0 in (-2, 0, 2, 4, 6):
            for g0 in g_starts:
                for s0 in (np.log(0.5), np.log(0.8)):
                    t0 = [a0, g0, s0] if slope_free else [a0, s0]
                    o = optimize.minimize(self.nll, t0, args=(mu_pred, slope_free), method="Nelder-Mead",
                                          options=dict(maxiter=6000, xatol=1e-6, fatol=1e-7))
                    if best is None or o.fun < best.fun:
                        best = o
        o = optimize.minimize(self.nll, best.x, args=(mu_pred, slope_free), method="Powell",
                              options=dict(maxiter=6000, xtol=1e-6, ftol=1e-7))
        return o if o.fun < best.fun else best


def censored_setup():
    """(likelihood, V) from the result files of stages 2 and 3."""
    pp = load_posteriors()
    S = scored_pairs(pp)
    V, Q = rate_frame(S)
    pw = pd.read_csv(common.NOISE / "ladder_power_curves.csv")
    pw["pc_old_id"] = pw.pc_old_id.astype(int)
    return CensoredLikelihood(V, Q, pw), V


def censored_fits(cl, V):
    models = {"two-gate rate dft": V["log10 k_q pred dft"].to_numpy(),
              "two-gate rate ml": V["log10 k_q pred ml"].to_numpy(),
              "adiabatic rate dft (k_d / (1 + e^(dG/RT)))": V["log10 k_q pred adiabatic dft"].to_numpy(),
              "adiabatic rate ml": V["log10 k_q pred adiabatic ml"].to_numpy(),
              "constant": np.full(len(V), 9.0)}
    rows = []
    for k, mu_pred in models.items():
        for slope_free in ((False, True) if k != "constant" else (False,)):
            o = cl.fit(mu_pred, slope_free)
            npar = 3 if slope_free else 2
            rows.append({"model": k, "form": "mu = a + g * pred" if slope_free else "mu = a + pred",
                         "n": len(V), "n_quench": int(cl.yq.sum()), "nll": o.fun, "AIC": 2 * o.fun + 2 * npar,
                         "offset_a / decades": o.x[0],
                         "slope_g": o.x[1] if slope_free else (1.0 if k != "constant" else 0.0),
                         "scatter_sigma / decades": np.exp(o.x[-1])})
    CL = pd.DataFrame(rows)
    CL["dAIC_vs_best"] = CL.AIC - CL.AIC.min()
    return CL


def main():
    rng = np.random.default_rng(SEED)
    pp = load_posteriors()
    S = scored_pairs(pp)
    CR = criteria(S)
    Yr = S[pairs.QUENCHES].to_numpy().astype(bool)
    grp = S[pairs.PC_NAME].to_numpy()
    pdet, p10 = S[P_DET].to_numpy(), S[P_10].to_numpy()

    out = common.NOISE
    common.write(confidence_counts(pp, S), out / "label_confidence_counts.csv")
    common.write(disagreement(CR, Yr, pdet).round(3), out / "disagreement_vs_uncertainty.csv")
    M = label_metrics(CR, Yr, grp, pdet, p10, rng)
    common.write(M.round(3), out / "metrics_label_uncertainty.csv")

    lab = S[[pairs.PC_NAME, "pc_old_id", pairs.Q_ID, pairs.Q_NAME, pairs.QUENCHES, "rung_dataset",
             "K_SV_dataset", "r2_dataset", "K_det / M-1", "K_med [free]", "K_q025 [free]", "K_q975 [free]",
             "P_K>0 [free]", P_10, P_DET]].rename(columns={"rung_dataset": "rung", "K_SV_dataset": "K_SV / M-1",
                                                           "r2_dataset": "r2"})
    lab["label_class_Kdet"] = classes(pdet)
    lab["label_class_K10"] = classes(p10)
    for k, (a, s) in CR.items():
        lab[f"pred {k}"] = a.astype(int)
    common.write(lab.round(4), out / "pair_labels_probabilistic.csv")

    V, Q = rate_frame(S)
    common.write(rate_parity_noise(Q).round(3), out / "rate_parity_noise.csv")
    pw = pd.read_csv(common.NOISE / "ladder_power_curves.csv")
    pw["pc_old_id"] = pw.pc_old_id.astype(int)
    CL = censored_fits(CensoredLikelihood(V, Q, pw), V)
    common.write(CL.round(3), out / "censored_likelihood.csv")

    pd.set_option("display.width", 250)
    print(M[["criterion", "setting", "n", "MCC"]].round(3).to_string(index=False))
    print(CL.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
