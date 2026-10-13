# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Labels and K_SV of every estimator of robust_fits.py compared with the data set, and the
benchmark metrics under each labelling.

Reads   results/labels/robust/fits_long.csv.gz (robust_fits.py)
        ../../../data/exp_dataset.csv            photocatalyst names
        ../../../data/exp_dataset_ent.csv        (through pairs.py)
Writes  results/labels/robust/method_summary.csv       per estimator and rule: quenching
                                                       pairs, label changes against the
                                                       data set, shifts of log10 K_SV
        results/labels/robust/method_agreement.csv     Cohen kappa of the labels and
                                                       Spearman rho of log10 K_SV between
                                                       estimators
        results/labels/robust/pair_stability.csv       per pair: fraction of the 26
                                                       estimator x rule combinations that
                                                       label it quenching, label confidence
        results/labels/robust/unstable_by_pc.csv       unstable pairs per photocatalyst
        results/labels/robust/unstable_by_ksv_bin.csv  unstable pairs per K_SV bin
        results/labels/robust/downstream_metrics.csv   metrics of the benchmark criteria
                                                       under every labelling

    uv run benchmark/further_analysis/labels/robust_labels.py

The pairs of the 26 benchmark photocatalysts (PC22 left out), 1,248. Rules: ladder, the
label ladder of the data set on the estimator's K_SV and r2; stat0.01 and stat0.05, the
per-photocatalyst noise test. The label of the data set is the CUR ladder label. A pair is
unstable when fewer than 80 % of the 26 combinations (13 estimators x ladder, stat0.01)
agree on its label. Log10 K_SV shifts and Spearman rho are over the quenching pairs of the
data set with K_SV > 0 under both estimators.

Downstream: the scored pairs of the benchmark, criteria at the benchmark settings
(classification.py): adiabatic (DFT, ML), DvTE p = 0.008 normal quantile, two-gate (DFT,
ML) with the per-pair contact distance; each scored against the labels of the data set,
of every estimator and rule, and of the majority of the 26 combinations. Summary tables are
written with three decimals, pair_stability.csv with four.
"""

import numpy as np
import pandas as pd
from scipy import stats

import common  # puts benchmark/ on the path
import classification
import pairs
import scoring

KEY = ["pc_name", "quencher_id"]
RULES = {"ladder": "label_ladder", "stat0.01": "label_stat_0.01", "stat0.05": "label_stat_0.05"}
UNSTABLE_BELOW = 0.8
K_BINS = [-np.inf, 0, 10, 20, 50, 100, 300, 1000, np.inf]
TWO_GATE = classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                  k_sv_crit=classification.K_SV_CRIT, contact_distance=None)
CRITERIA = {"adiabatic dG<=0 (dft)": (classification.Setting("adiabatic"), "dft"),
            "adiabatic dG<=0 (ml)": (classification.Setting("adiabatic"), "ml"),
            "DvTE p=0.008 normal": (classification.Setting("dvte", p="0.008", dvte_kind="normal"), "dft"),
            "two-gate asym t_PC (dft)": (TWO_GATE, "dft"),
            "two-gate asym t_PC (ml)": (TWO_GATE, "ml")}


def logk(k):
    return np.log10(np.where(k > 0, k, np.nan))


def cohen_kappa(a, b):
    """Unweighted Cohen kappa of two label vectors."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == v) * np.mean(b == v) for v in labels))
    return (po - pe) / (1 - pe) if pe != 1 else np.nan


def load():
    L = pd.read_csv(common.ROBUST / "fits_long.csv.gz")
    exp = pd.read_csv(common.EXP_DATASET)
    idmap = exp[["pc_old_id", "pc_new_id", "pc_name", "tau0 / ns"]].drop_duplicates().set_index("pc_old_id")
    L = L.join(idmap, on="pc_old_id")
    return L[L.pc_name != common.LEFT_OUT_PC].reset_index(drop=True)


def method_summary(L, methods, wide_lab, wide_K, cur):
    rows = []
    lk_cur = logk(wide_K["CUR"].to_numpy())
    q_cur = cur.to_numpy() == 1
    for m in methods:
        d = logk(wide_K[m].to_numpy()) - lk_cur
        for r in RULES:
            lab = wide_lab[r][m].to_numpy()
            rows.append(dict(method=m, rule=r, n=len(lab), n_quench=int(lab.sum()),
                             flips_vs_current=int((lab != cur.to_numpy()).sum()),
                             gained=int(((lab == 1) & ~q_cur).sum()), lost=int(((lab == 0) & q_cur).sum()),
                             median_abs_dlog10K_quenching=np.nanmedian(np.abs(d[q_cur])),
                             p90_abs_dlog10K_quenching=np.nanpercentile(np.abs(d[q_cur]), 90),
                             frac_gt_0p1dex_quenching=np.nanmean(np.abs(d[q_cur]) > 0.1),
                             median_dlog10K_quenching=np.nanmedian(d[q_cur]),
                             n_K_nonpositive=int((wide_K[m] <= 0).sum())))
    summ = pd.DataFrame(rows)
    for r, pcol in (("ladder", "p_ladder_boot"), ("stat0.01", "p_stat_0.01_boot"), ("stat0.05", "p_stat_0.05_boot")):
        pb = L.pivot_table(index=KEY, columns="method", values=pcol)
        for m in methods:
            p = pb[m].to_numpy()
            i = (summ.method == m) & (summ.rule == r)
            summ.loc[i, "boot_label_stability"] = np.nanmean(np.maximum(p, 1 - p))
            summ.loc[i, "n_boot_unstable_(0.2<p<0.8)"] = int(((p > 0.2) & (p < 0.8)).sum())
    ciw = L.assign(rel=(L.K_hi - L.K_lo) / L.K_SV.abs()).pivot_table(index=KEY, columns="method", values="rel")
    for m in methods:
        summ.loc[summ.method == m, "median_rel_CI95_width_quenching"] = np.nanmedian(ciw[m].to_numpy()[q_cur])
    return summ


def method_agreement(methods, wide_lab, wide_K, cur):
    q_cur = cur.to_numpy() == 1
    rows = []
    for a in methods:
        for b in methods:
            ka, kb = logk(wide_K[a].to_numpy()), logk(wide_K[b].to_numpy())
            m = np.isfinite(ka) & np.isfinite(kb) & q_cur
            rows.append(dict(a=a, b=b,
                             kappa_ladder=cohen_kappa(wide_lab["ladder"][a], wide_lab["ladder"][b]),
                             kappa_stat001=cohen_kappa(wide_lab["stat0.01"][a], wide_lab["stat0.01"][b]),
                             spearman_logK_quenching=stats.spearmanr(ka[m], kb[m])[0]))
    return pd.DataFrame(rows)


def pair_stability(L, wide_lab, wide_K, cur):
    votes = pd.concat([wide_lab["ladder"].add_suffix("|ladder"), wide_lab["stat0.01"].add_suffix("|stat0.01")],
                      axis=1)
    st = pd.DataFrame(index=votes.index)
    st["frac_quench_all"] = votes.mean(axis=1)
    st["frac_quench_ladder"] = wide_lab["ladder"].mean(axis=1)
    st["frac_quench_stat"] = wide_lab["stat0.01"].mean(axis=1)
    st["label_confidence"] = np.maximum(st.frac_quench_all, 1 - st.frac_quench_all)
    st["current"] = cur
    st["consensus"] = (st.frac_quench_all >= 0.5).astype(int)
    st["unstable"] = (st.label_confidence < UNSTABLE_BELOW).astype(int)
    st["p_boot_current"] = L.pivot_table(index=KEY, columns="method", values="p_ladder_boot")["CUR"]
    st["p_boot_OLS_stat"] = L.pivot_table(index=KEY, columns="method", values="p_stat_0.01_boot")["OLS"]
    lk = np.log10(wide_K.where(wide_K > 0))
    st["K_SV_current"] = wide_K["CUR"]
    st["K_min"] = wide_K.min(axis=1)
    st["K_max"] = wide_K.max(axis=1)
    st["log10K_spread_dex"] = lk.max(axis=1) - lk.min(axis=1)
    fcur = L[L.method == "CUR"].set_index(KEY)
    st["r2_current"] = fcur.r2
    st["intercept_current"] = fcur.intercept
    st["n_used_current"] = fcur.n_used
    st["K_crit_OLS_0.01"] = L[L.method == "OLS"].set_index(KEY)["K_crit_0.01"]
    st = st.reset_index()
    st["quencher_name"] = "Q" + st.quencher_id.astype(str)
    return st


def anatomy(st, by):
    g = st.groupby(by)
    t = pd.DataFrame({"n": g.size(), "n_unstable": g.unstable.sum(),
                      "frac_unstable": g.unstable.mean(),
                      "n_current_quench": g.current.sum(), "n_consensus_quench": g.consensus.sum(),
                      "flips_current_vs_consensus": g.apply(lambda d: int((d.current != d.consensus).sum())),
                      "median_label_conf": g.label_confidence.median()})
    return t.sort_values("frac_unstable", ascending=False).reset_index()


def downstream(methods, wide_lab, st):
    df = pairs.scored(pairs.load())
    pred = {k: classification.predict(df, s, src)[0] for k, (s, src) in CRITERIA.items()}
    sti = st.set_index(KEY)
    idx = list(zip(df.pc_name, df.quencher_id))
    assert (df[pairs.QUENCHES].to_numpy() == sti.loc[idx, "current"].to_numpy()).all()
    schemes = {"current (data set)": sti.current}
    for m in methods:
        for r in ("ladder", "stat0.01"):
            schemes[f"{m} | {r}"] = wide_lab[r][m]
    schemes["consensus (majority of 26)"] = sti.consensus
    stable = sti.loc[idx, "unstable"].to_numpy() == 0
    out = []
    for sname, lab in schemes.items():
        y = lab.loc[idx].to_numpy().astype(bool)
        for subset, msk in (("all", np.ones(len(df), bool)), ("stable pairs only", stable)):
            if subset != "all" and sname not in ("current (data set)", "consensus (majority of 26)"):
                continue
            for pname, p in pred.items():
                mt = scoring.metrics(y[msk], p[msk])
                out.append(dict(labelling=sname, subset=subset, criterion=pname, n=mt["n"],
                                n_quench=int(y[msk].sum()), MCC=mt["MCC"], recall=mt["recall"],
                                specificity=mt["specificity"], precision=mt["precision"],
                                label_flips_vs_current=int((y != df[pairs.QUENCHES].to_numpy(bool)).sum())))
    return pd.DataFrame(out)


def main():
    L = load()
    methods = list(dict.fromkeys(L.method))
    wide_lab = {r: L.pivot_table(index=KEY, columns="method", values=c) for r, c in RULES.items()}
    wide_K = L.pivot_table(index=KEY, columns="method", values="K_SV")
    cur = wide_lab["ladder"]["CUR"]

    out = common.ROBUST
    common.write(method_summary(L, methods, wide_lab, wide_K, cur).round(3), out / "method_summary.csv")
    common.write(method_agreement(methods, wide_lab, wide_K, cur).round(3), out / "method_agreement.csv")
    st = pair_stability(L, wide_lab, wide_K, cur)
    common.write(st.round(4), out / "pair_stability.csv")
    common.write(anatomy(st, "pc_name").round(3), out / "unstable_by_pc.csv")
    st["K_bin_current"] = pd.cut(st.K_SV_current, K_BINS).astype(str)
    common.write(anatomy(st, "K_bin_current").round(3), out / "unstable_by_ksv_bin.csv")
    dm = downstream(methods, wide_lab, st)
    common.write(dm.round(3), out / "downstream_metrics.csv")
    print(dm[(dm.labelling == "current (data set)") & (dm.subset == "all")][["criterion", "MCC"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
