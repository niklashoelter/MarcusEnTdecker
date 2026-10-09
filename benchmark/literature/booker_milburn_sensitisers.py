"""
Elliott, Kayal, George, Booker-Milburn 2020: a thioxanthone sensitiser series -- SI material.

    uv run python benchmark/literature/booker_milburn_sensitisers.py

The paper tunes thioxanthone (TX) with F and MeO groups, trading triplet energy for absorption
wavelength, and measures the triplet energy of eleven TX by phosphorescence. Every TX triplet
is 3pi,pi*, so the donor varies in E_T but hardly in character. Two comparisons:

    T1       donor E_T   our E_T of the 11 TX against the measured phosphorescence. A check of
                         the E_T head only; it says nothing about barriers. No TX is in the
                         training set.
    Table 2  benzophenone (3n,pi*) against three TX on acceptor 13: the barrier, our E_T and the
                         measured E_T against the preparative rate (g/h). Descriptive: n = 4,
                         mixed concentrations and run times, poor BP absorption at 366 nm.

CONVENTION
    Asymmetric form. Every energy in a barrier is predicted -- donor and acceptor -- so the
    measured E_T enter only as baselines.

PASS CRITERION (T1; `pass_*` columns, and booker_milburn_criteria.csv)
    rho >= 0.9, and not more than 0.05 below the paper's TD-DFT rho. The gate is the deployed
    model's mixture mean; the count over all 12 model x estimate runs is reported.

Writes   results/literature/booker_milburn_sensitisers.csv    our and the paper's properties per sensitiser, model, estimate
         results/literature/booker_milburn_t1.csv             T1: donor E_T against phosphorescence, per run, with TD-DFT baselines
         results/literature/booker_milburn_table2.csv         benzophenone comparison (descriptive)
         results/literature/booker_milburn_table2_summary.csv its rank agreement per model and estimate (descriptive)
         results/literature/booker_milburn_criteria.csv       the T1 criterion, with pass/fail
"""

import numpy as np
import pandas as pd

from common import (ESTIMATES, barrier_asymmetric, bootstrap_spearman, literature, models, ours, pair_lambdas,
                    spearman, split_difference, write)

CASE = "booker_milburn_2020"
SENSITISERS = [f"S{i}" for i in range(1, 12)]
PARENT_TX = "S5"
TABLE2 = ["BP", "S6", "S4", "S1"]           # BP, ITX, 3-FTX, 3,3'-MeOTX on 13
TABLE2_ACCEPTOR = "13"

RHO_T1, TDDFT_TOLERANCE = 0.9, 0.05


def terms(pc, acceptor):
    """(dG, lambda_forward, lambda_reverse) of one pair."""
    forward, reverse = pair_lambdas(pc, acceptor)
    return acceptor["E_T"] - pc["E_T"], forward, reverse


def phi_isc(value):
    """Phi_ISC as printed ('>0.9', '0.76 (lit.)') to a number."""
    return float(str(value).replace(">", "").replace("(lit.)", "").strip()) if pd.notna(value) else np.nan


def main():
    lit = literature(CASE).set_index("compound_id")
    preds = models(CASE)
    runs = [(m, e) for m in preds for e in ESTIMATES]
    o = {(m, e): {c: ours(preds[m].loc[c], e) for c in preds[m].index} for m, e in runs}
    E_T_lit = lit.E_T_lit
    phi = lit.phi_ISC_lit.map(phi_isc)

    # ---------------------------------------------------------------- sensitisers and T1
    sens = []
    for m, e in runs:
        for c in SENSITISERS + ["BP"]:
            p = o[(m, e)][c]
            sens.append({"model": m, "estimate": e, "compound_id": c, "label": lit.at[c, "label"],
                         "triplet_character": lit.at[c, "triplet_character"],
                         "E_T_lit": E_T_lit[c], "E_T_calc_lit": lit.at[c, "E_T_calc_lit"],
                         "E_T_calc_631G_lit": lit.at[c, "E_T_calc_631G_lit"],
                         "E_T_ml": p["E_T"], "E_T_sd_ml": p["E_T_sd"], "E_T_error_ml": p["E_T"] - E_T_lit[c],
                         "lambda_S0_T1_ml": p["lambda_S0_T1"], "lambda_T1_S0_ml": p["lambda_T1_S0"],
                         "phi_ISC_lit": phi[c]})
    write(pd.DataFrame(sens), "booker_milburn_sensitisers.csv")

    meas = E_T_lit[SENSITISERS]
    t1 = []
    for name, col in (("paper TD-DFT B3LYP/6-31+G(d,p)", "E_T_calc_lit"), ("paper TD-DFT B3LYP/6-31G(d,p)", "E_T_calc_631G_lit")):
        x = lit.loc[SENSITISERS, col]
        t1.append({"predictor": name, "model": None, "estimate": None, "role": "baseline",
                   "n": spearman(x, meas)[2], "rho": spearman(x, meas)[0],
                   "rho_excl_parent_TX": spearman(x.drop(PARENT_TX), meas.drop(PARENT_TX))[0],
                   "mae": float((x - meas).abs().mean()), "mean_signed_error": float((x - meas).mean())})
    rho_tddft = max(r["rho"] for r in t1)
    for m, e in runs:
        x = pd.Series({c: o[(m, e)][c]["E_T"] for c in SENSITISERS})
        rho, _, n = spearman(x, meas)
        row = {"predictor": "ours", "model": m, "estimate": e, "role": "tested", "n": n, "rho": rho,
               "rho_excl_parent_TX": spearman(x.drop(PARENT_TX), meas.drop(PARENT_TX))[0],
               "mae": float((x - meas).abs().mean()), "mean_signed_error": float((x - meas).mean()),
               "pass_rho": rho >= RHO_T1, "pass_vs_tddft": rho >= rho_tddft - TDDFT_TOLERANCE}
        row["pass_T1"] = row["pass_rho"] and row["pass_vs_tddft"]
        if (m, e) == ("deployed", "mean"):
            row["rho_ci_low"], row["rho_ci_high"] = bootstrap_spearman(x, meas)
        t1.append(row)
    t1 = pd.DataFrame(t1)
    write(t1, "booker_milburn_t1.csv")

    # ---------------------------------------------------------------- descriptive: Table 2
    tab2, tab2s = [], []
    obs = lit.loc[TABLE2]
    g_per_h = obs.mass_table2_g / obs.time_table2_h
    for m, e in runs:
        p = o[(m, e)]
        acc = p[TABLE2_ACCEPTOR]
        bp = terms(p["BP"], acc)
        rows = []
        for c in TABLE2:
            t = terms(p[c], acc)
            full, dg_part, lam_part = split_difference(barrier_asymmetric, bp, t)
            rows.append({"model": m, "estimate": e, "compound_id": c, "label": lit.at[c, "label"],
                         "triplet_character": lit.at[c, "triplet_character"],
                         "yield_percent_exp": obs.at[c, "yield_table2_percent_exp"], "time_h": obs.at[c, "time_table2_h"],
                         "conc_M": obs.at[c, "conc_table2_M"], "g_per_h_exp": g_per_h[c],
                         "E_T_lit": E_T_lit[c], "E_T_ml": p[c]["E_T"],
                         "lambda_S0_T1_ml": p[c]["lambda_S0_T1"], "lambda_T1_S0_ml": p[c]["lambda_T1_S0"],
                         "dG_EnT_ml": t[0], "barrier_ml": barrier_asymmetric(*t),
                         "ddG_BP_minus_this_ml": full, "ddG_BP_minus_this_ml_dG_part": dg_part,
                         "ddG_BP_minus_this_ml_lambda_part": lam_part})
        rows = pd.DataFrame(rows).set_index("compound_id")
        tab2.append(rows.reset_index())
        for score, s in (("barrier", -rows.barrier_ml), ("E_T predicted", rows.E_T_ml), ("E_T measured", rows.E_T_lit)):
            rho = spearman(s, g_per_h)[0]
            tab2s.append({"model": m, "estimate": e, "score": score, "n": len(rows), "rho": rho,
                          "order_exact": bool(np.isclose(rho, 1.0))})
    write(pd.concat(tab2, ignore_index=True), "booker_milburn_table2.csv")
    tab2s = pd.DataFrame(tab2s)
    write(tab2s, "booker_milburn_table2_summary.csv")

    # ---------------------------------------------------------------- criteria
    dep = lambda f: f[(f.model == "deployed") & (f.estimate == "mean")]
    t1r = t1[t1.role == "tested"]
    crit = [
        {"test": "T1", "criterion": f"rho >= {RHO_T1} and >= paper TD-DFT rho - {TDDFT_TOLERANCE}", "gate": "deployed, mean",
         "value_deployed_mean": float(dep(t1r).rho.iloc[0]), "runs_passing": int(t1r.pass_T1.sum()),
         "pass": bool(dep(t1r).pass_T1.iloc[0])},
        {"test": "T1", "criterion": f"rho >= {RHO_T1} (first part alone)", "gate": "deployed, mean",
         "value_deployed_mean": float(dep(t1r).rho.iloc[0]), "runs_passing": int(t1r.pass_rho.sum()),
         "pass": bool(dep(t1r).pass_rho.iloc[0])}]
    crit = pd.DataFrame(crit)
    write(crit, "booker_milburn_criteria.csv")

    # ---------------------------------------------------------------- summary
    rng = lambda s: f"{s.min():+.2f} to {s.max():+.2f}"
    print("\nT1 donor E_T vs phosphorescence (11 TX)")
    print(f"  ours rho {rng(t1r.rho)}, MAE {t1r.mae.min():.2f} to {t1r.mae.max():.2f}; "
          f"deployed mean rho {dep(t1r).rho.iloc[0]:.3f} (95 % CI {dep(t1r).rho_ci_low.iloc[0]:.2f}-{dep(t1r).rho_ci_high.iloc[0]:.2f}), "
          f"MAE {dep(t1r).mae.iloc[0]:.2f}")
    for r in t1[t1.role == "baseline"].itertuples():
        print(f"  {r.predictor}: rho {r.rho:.3f}, MAE {r.mae:.2f}")
    print("\nTable 2 (descriptive): exact order, runs of 12: " + ", ".join(
        f"{s} {int(g.order_exact.sum())}" for s, g in tab2s.groupby("score", sort=False)))
    print("\nT1 criterion")
    print(crit.to_string(index=False))


if __name__ == "__main__":
    main()
