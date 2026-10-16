# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
DvTE and the four-point parameters, symmetric and asymmetric barriers, and the observed
feasibility boundary.

Reads   ../../data/exp_dataset_ent.csv (also through pairs.py)
        results/quencher_classes/quencher_classes.csv
Writes  results/dvte_reorganization/quenchers.csv          per quencher (all 48): DvTE - E_T,
                                                           mean and width of the sampled gaps and
                                                           their Gaussian four-point estimates
        results/dvte_reorganization/fits.csv               least-squares lines of DvTE - E_T and
                                                           mean gap - E_T against lambda
        results/dvte_reorganization/quencher_summary.csv   range of DvTE - E_T, quenchers with
                                                           DvTE below E_T, implied k_B T_eff
        results/dvte_reorganization/barrier_forms.csv      asymmetric against symmetric barrier
        results/dvte_reorganization/barrier_forms_quenchers.csv   pairs only the asymmetric barrier admits
        results/dvte_reorganization/boundary_bins.csv      quenching fraction in dG_EnT bins (DFT)
        results/dvte_reorganization/boundary_summary.csv   pooled and per-photocatalyst threshold,
                                                           Sandros onset, two-gate without the
                                                           energy gate

    uv run benchmark/further_analysis/dvte_reorganization.py

Quencher level (all 48 quenchers, DFT unless marked ML). The Gaussian four-point estimate
of the vertical gap at thermal S0 geometries has mean mu = E_T + lambda_T1-S0 and variance
sigma^2 = 2 lambda_T1-S0 k_B T_eff; k_B T_eff = RT (classical) or (eps/2) coth(eps/2RT) for
one quantum eps of 1000 or 1500 cm-1. DvTE_4pt = mu + Phi^-1(0.008) sigma. The k_B T_eff
implied by a sampled distribution is sd^2 / (2 lambda_T1-S0). R^2 is the squared Pearson
correlation.

Scored pairs. Two-gate is the benchmark criterion (r from DFT sizes); its symmetric form
uses the symmetric barrier with the geometric-mean lambda, and the variant without the
energy gate requires only barrier <= t_PC. The pair reorganization energies are
lambda_f = lambda_T1-S0(PC) + lambda_S0-T1(Q) and lambda_r = lambda_S0-T1(PC) +
lambda_T1-S0(Q). The Youden threshold c is the value of the rule dG_EnT <= c with the
largest TPR - FPR (middle of ties; candidates between consecutive dG_EnT values); per
photocatalyst only where it has at least MIN_EACH pairs of each outcome. The Sandros onset
is dG_EnT = RT ln(k_d tau0 / K_SV,crit - 1).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"
RESULTS = HERE / "results" / "dvte_reorganization"

P_DVTE = 0.008
CM_TO_KCAL = 1 / 349.755
QUANTA = {"classical": None, "1000 cm-1": 1000, "1500 cm-1": 1500}
MIN_EACH = 3
BINS = ((-60, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 60))
TWO_GATE = classification.candidates("two_gate", "r from DFT sizes")[0]


def quencher_parameters():
    """One row per quencher of the dataset (all 48), indexed by name."""
    return pd.read_csv(pairs.DATASET).groupby(pairs.Q_NAME).first()


# ----------------------------------------------------------------------------- quencher level
def kT_eff(quantum):
    """Effective k_B T of one quantum in cm-1 (None: classical)."""
    if quantum is None:
        return theory.RT
    hw = quantum * CM_TO_KCAL
    return hw / 2 / np.tanh(hw / (2 * theory.RT))


def gaussian_estimate(E_T, lambda_t1_s0, quantum):
    """(mean, width, 0.008 quantile) of the Gaussian four-point gap distribution."""
    mu = E_T + lambda_t1_s0
    sd = np.sqrt(2 * lambda_t1_s0 * kT_eff(quantum))
    return mu, sd, mu + stats.norm.ppf(P_DVTE) * sd


def quencher_table(q, classes):
    E, lt, ls = (q[f"q_{c}_dft / kcal mol-1"] for c in ("E_T", "lambda_T1_S0", "lambda_S0_T1"))
    mu, sd, dvte = q["q_dvte_mu / kcal mol-1"], q["q_dvte_std / kcal mol-1"], q[pairs.q_dvte(str(P_DVTE), "normal")]
    t = pd.DataFrame({"class": q.index.map(classes), "E_T / kcal mol-1": E,
                      "lambda_T1_S0 / kcal mol-1": lt, "lambda_S0_T1 / kcal mol-1": ls,
                      "dvte_minus_E_T / kcal mol-1": dvte - E, "dvte_mu_minus_E_T / kcal mol-1": mu - E,
                      "dvte_mu / kcal mol-1": mu, "dvte_std / kcal mol-1": sd, "dvte / kcal mol-1": dvte,
                      "kT_eff_over_RT": sd ** 2 / (2 * lt) / theory.RT})
    for label, quantum in QUANTA.items():
        m4, s4, q4 = gaussian_estimate(E, lt, quantum)
        t["mu_4pt / kcal mol-1"] = m4
        t[f"sd_4pt {label} / kcal mol-1"] = s4
        t[f"dvte_4pt {label} / kcal mol-1"] = q4
    t["dvte_4pt ml 1000 cm-1 / kcal mol-1"] = gaussian_estimate(
        q["q_E_T_ml / kcal mol-1"], q["q_lambda_T1_S0_ml / kcal mol-1"], 1000)[2]
    t.index.name = "quencher"
    return t.sort_index(key=lambda s: s.str[1:].astype(int)).reset_index()


def line(x, y, name):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    return {"fit": name, "n": len(x), "slope": slope, "intercept / kcal mol-1": intercept,
            "R2": stats.pearsonr(x, y)[0] ** 2,
            "rmse / kcal mol-1": np.sqrt(np.mean((y - (slope * x + intercept)) ** 2))}


def fits(t):
    return pd.DataFrame([
        line(t["lambda_T1_S0 / kcal mol-1"], t["dvte_minus_E_T / kcal mol-1"], "DvTE - E_T vs lambda_T1-S0"),
        line(t["lambda_T1_S0 / kcal mol-1"], t["dvte_mu_minus_E_T / kcal mol-1"], "mean gap - E_T vs lambda_T1-S0"),
        line(t["lambda_S0_T1 / kcal mol-1"], t["dvte_minus_E_T / kcal mol-1"], "DvTE - E_T vs lambda_S0-T1")])


def quencher_summary(t):
    d = t["dvte_minus_E_T / kcal mol-1"]
    below = t[d < 0]
    nondiss = t["class"] != "dissociative"
    return pd.DataFrame([
        {"quantity": "DvTE - E_T, minimum / kcal mol-1", "value": d.min()},
        {"quantity": "DvTE - E_T, maximum / kcal mol-1", "value": d.max()},
        {"quantity": "quenchers with DvTE < E_T", "value": len(below)},
        {"quantity": "quenchers with DvTE < E_T, rigid", "value": int((below["class"] == "rigid").sum())},
        {"quantity": "largest lambda_T1-S0 of the quenchers with DvTE < E_T / kcal mol-1",
         "value": below["lambda_T1_S0 / kcal mol-1"].max()},
        {"quantity": "k_B T_eff / RT implied by the DvTE width, median, non-dissociative quenchers",
         "value": t.loc[nondiss, "kT_eff_over_RT"].median(), "n": int(nondiss.sum())},
        {"quantity": "k_B T_eff / RT implied by the DvTE width, median, all quenchers",
         "value": t["kT_eff_over_RT"].median(), "n": len(t)},
        *({"quantity": f"k_B T_eff / RT of one {label} quantum", "value": kT_eff(w) / theory.RT}
          for label, w in QUANTA.items() if w is not None)])


# ----------------------------------------------------------------------------- scored pairs
def two_gate(df, source, model="marcus_asymmetric", energy_gate=True):
    """(prediction, score, dG_EnT, barrier) of the two-gate criterion."""
    dG = pairs.driving_force(df, source)
    forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
    barrier = theory.barrier(model, dG, forward, reverse)
    t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), TWO_GATE.k_sv_crit,
                                   theory.escape_over_frequency(pairs.contact_distances(df)))
    margin = t_pc - barrier
    exergonic = dG <= 0
    predicted = (exergonic & (margin >= 0)) if energy_gate else margin >= 0
    return predicted, np.where(exergonic, margin, margin - classification.ENDERGONIC_OFFSET), dG, barrier


def barrier_forms(df, classes):
    y, g = df[pairs.QUENCHES].to_numpy(bool), df[pairs.PC_NAME].to_numpy()
    rows, per_quencher = [], None
    for source in pairs.SOURCES:
        pa, sa, dG, ba = two_gate(df, source)
        assert np.array_equal(pa, classification.predict(df, TWO_GATE, source)[0])
        ps, ss, _, bs = two_gate(df, source, "marcus_symmetric")
        _, _, q_s0_t1, _ = pairs.lambdas(df, source)
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        exo = dG <= 0
        diff = bs - ba
        cls = df[pairs.Q_NAME].map(classes).to_numpy()
        only_a, only_s = pa & ~ps, ps & ~pa
        row = {"parameters": source, "n_exergonic": int(exo.sum()),
               "rho_barrier_difference_vs_lambda_forward_minus_lambda_reverse": stats.spearmanr(diff[exo], (forward - reverse)[exo])[0],
               "rho_barrier_difference_vs_lambda_S0_T1_Q": stats.spearmanr(diff[exo], q_s0_t1[exo])[0]}
        for c in ("rigid", "flexible", "dissociative"):
            inside = exo & (cls == c)
            row[f"median_barrier_difference_{c} / kcal mol-1"] = np.median(diff[inside]) if inside.any() else np.nan
        row |= {"admitted_only_asymmetric": int(only_a.sum()), "admitted_only_asymmetric_quench": int((only_a & y).sum()),
                "admitted_only_symmetric": int(only_s.sum()), "admitted_only_symmetric_quench": int((only_s & y).sum()),
                "MCC_asymmetric": scoring.metrics(y, pa)["MCC"], "MCC_symmetric": scoring.metrics(y, ps)["MCC"],
                "AUC_within_PC_pooled_asymmetric": scoring.auc_within(y, sa, g)["pooled"],
                "AUC_within_PC_pooled_symmetric": scoring.auc_within(y, ss, g)["pooled"],
                "pairs_inverted": int(np.sum(-dG > theory.geometric_mean(forward, reverse))),
                "min_lambda_plus_dG / kcal mol-1": float(np.min(theory.geometric_mean(forward, reverse) + dG)),
                "min_lambda_reverse_plus_dG / kcal mol-1": float(np.min(reverse + dG))}
        rows.append(row)
        if source == "dft":
            per_quencher = (pd.DataFrame({"quencher": df[pairs.Q_NAME], "admitted_only_asymmetric": only_a,
                                          "admitted_only_asymmetric_quench": only_a & y})
                            .groupby("quencher", sort=False).sum()
                            .query("admitted_only_asymmetric > 0")
                            .sort_values("admitted_only_asymmetric", ascending=False, kind="stable")
                            .reset_index().assign(parameters="dft"))
    return pd.DataFrame(rows), per_quencher[["parameters", "quencher", "admitted_only_asymmetric",
                                             "admitted_only_asymmetric_quench"]]


def youden(dG, y):
    """Threshold c of dG <= c with the largest TPR - FPR; the middle one of ties."""
    dG, y = np.asarray(dG, float), np.asarray(y, bool)
    xs = np.unique(dG)
    candidates = np.r_[xs[0] - 0.5, (xs[1:] + xs[:-1]) / 2, xs[-1] + 0.5]
    J = np.array([((dG <= c) & y).sum() / y.sum() - ((dG <= c) & ~y).sum() / (~y).sum() for c in candidates])
    best = np.flatnonzero(np.isclose(J, J.max()))
    return float(candidates[best[(len(best) - 1) // 2]])


def boundary(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    dG = pairs.driving_force(df, "dft")
    bins = pd.DataFrame([{"parameters": "dft", "dG_low / kcal mol-1": lo, "dG_high / kcal mol-1": hi,
                          "n": int(((dG > lo) & (dG <= hi)).sum()),
                          "quench_fraction": float(y[(dG > lo) & (dG <= hi)].mean())} for lo, hi in BINS])
    c_pc = []                    # per photocatalyst with at least MIN_EACH pairs of each outcome
    for _, g in df.groupby(pairs.PC_NAME, sort=False):
        yy = g[pairs.QUENCHES].to_numpy(bool)
        if yy.sum() >= MIN_EACH and (~yy).sum() >= MIN_EACH:
            c_pc.append(youden(g[pairs.dg_ent("dft")].to_numpy(float), yy))
    c_pc = np.array(c_pc)
    sandros = theory.RT * np.log(theory.K_DIFF * pairs.tau0_seconds(df) / TWO_GATE.k_sv_crit - 1)
    p_gate = two_gate(df, "dft")[0]
    p_free, _, _, _ = two_gate(df, "dft", energy_gate=False)
    endergonic_pass = p_free & (dG > 0)
    summary = pd.DataFrame([
        {"quantity": "Youden threshold, pooled / kcal mol-1", "value": youden(dG, y)},
        {"quantity": "Youden threshold, median over photocatalysts / kcal mol-1",
         "value": float(np.median(c_pc)), "n": len(c_pc)},
        {"quantity": "photocatalysts with a negative threshold", "value": int((c_pc < 0).sum()),
         "n": len(c_pc)},
        {"quantity": "Sandros onset, median over the scored pairs / kcal mol-1", "value": float(np.median(sandros)),
         "n": len(df)},
        {"quantity": "MCC, two-gate DFT", "value": scoring.metrics(y, p_gate)["MCC"]},
        {"quantity": "MCC, two-gate DFT without the energy gate", "value": scoring.metrics(y, p_free)["MCC"]},
        {"quantity": "endergonic pairs passing the barrier gate", "value": int(endergonic_pass.sum())},
        {"quantity": "quench fraction of the endergonic pairs passing the barrier gate",
         "value": float(y[endergonic_pass].mean())}])
    return bins, summary


def rounded(table):
    """kcal/mol to 2 decimals, everything else to 3; in quantity/value tables by row."""
    table = table.copy()
    if {"quantity", "value"} <= set(table.columns):
        table["value"] = [round(float(v), 2 if "kcal" in q else 3) for q, v in zip(table.quantity, table.value)]
        if "n" in table:
            table["n"] = table["n"].astype("Int64")
    for c in table.select_dtypes(float).columns:
        if c != "value":
            table[c] = table[c].round(2 if "kcal" in c else 3)
    return table


def main():
    classes = pd.read_csv(CLASSES).set_index("id")["proposed_class"]
    q = quencher_table(quencher_parameters(), classes)
    df = pairs.scored(pairs.load())
    forms, forms_q = barrier_forms(df, classes)
    bins, summary = boundary(df)
    tables = {"quenchers": q, "fits": fits(q), "quencher_summary": quencher_summary(q),
              "barrier_forms": forms, "barrier_forms_quenchers": forms_q,
              "boundary_bins": bins, "boundary_summary": summary}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        rounded(table).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")
    for name in tables:
        if name != "quenchers":
            print(rounded(tables[name]).to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
