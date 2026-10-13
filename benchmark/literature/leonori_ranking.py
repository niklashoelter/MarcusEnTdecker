"""
Leonori 2025: nitroarene photocatalysts -- triplet character, and which substrates react.

    uv run --extra ml python benchmark/literature/leonori_ranking.py

The paper's claim is that a nitroarene's ability to sensitise is set by which triplet is
lowest: the delocalised 3pi,pi* state works, the 3n,pi* state on the pyramidalised nitro group
does not, because it must planarise before it can transfer its energy -- a reorganisation on
the donor side. Two questions follow for us:

    character     does our predicted reorganisation energy, from one predicted triplet,
                  recover the paper's two-state descriptor (dE_TT = E(3pi,pi*) - E(3n,pi*)) and
                  the triplet class? The catalyst's lambda_T1_S0 is the donor term of the forward
                  barrier and is the one reported first; the other lambda, their sum and the
                  barrier are given beside it. Both axes are computed, so this is a consistency
                  check, and it is run at fixed E_T as well.
    feasibility   does our barrier separate the substrates that react with N24, the paper's
                  isomerisation catalyst, from the alkyl acrylates that do not? The paper
                  explains the failures with adiabatic E_T (SI Table 8), so the driving force
                  is scored beside the barrier.

Reported as well: the measured quenching constants (a table: four catalysts against (E)-1, one
against 27, measured by two methods, each converted to an experimental barrier by the Eyring
equation as Sole-Daura did), and the one measured E/Z pair (N24 with (E)- and (Z)-1).

EXCLUDED
    The 24 h cycloaddition yields (13 catalysts): every descriptor that correlates with them
    does so by separating the two triplet classes (a bare class label does as well), so they
    carry no trend information.
    The Z share of (E)-1 after 5 min (the paper's Fig. 2b), which the paper reads as a threshold
    at dE_TT = 0: as a threshold on our quantities, the class-marking E_T reproduces it better
    than lambda_T1_S0 or the barrier, so it is not used.

Lifetime is a confound the paper cannot remove either: most 3n,pi* catalysts have picosecond
triplets, which no barrier can rescue.

Writes   results/literature/leonori_catalysts.csv     one row per catalyst, model and estimate
         results/literature/leonori_character.csv     lambda against triplet character
         results/literature/leonori_feasibility.csv   substrates with N24
         results/literature/leonori_feasibility_summary.csv   the separation per model and estimate
         results/literature/leonori_quenching.csv     measured k_q beside our barriers, and the E/Z pair
"""

import numpy as np
import pandas as pd

from common import (ESTIMATES, RT, auc, barrier_asymmetric, literature, models,
                    ours, pair_barrier, pair_lambdas, spearman, split_difference, write)

ISOMERISATION_CATALYST = "N24"
K_B, H_PLANCK, T = 1.380649e-23, 6.62607015e-34, 298.15


def eyring_barrier(k):
    """Experimental barrier from a bimolecular rate constant (M-1 s-1, 1 M standard state) by the
    Eyring equation, as Sole-Daura derived theirs; a diffusion correction is negligible here."""
    return float(-RT * np.log(k * H_PLANCK / (K_B * T)))


def partial_spearman(x, y, control):
    """Spearman of x and y with the rank effect of `control` removed from both."""
    d = pd.DataFrame({"x": x, "y": y, "c": control}).dropna()
    rx, ry, rc = d.x.rank(), d.y.rank(), d.c.rank()
    ex = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
    ey = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    lit = literature("leonori_2025").set_index("compound_id")
    cats = lit[lit.role == "photocatalyst"].index
    subs = lit[lit.role == "substrate"].index

    rows = []
    for model, pred in models("leonori_2025").items():
        for estimate in ESTIMATES:
            o = {c: ours(pred.loc[c], estimate) for c in pred.index}
            for c in cats:
                row = {"model": model, "estimate": estimate, "compound_id": c,
                       "triplet_character_lit": lit.at[c, "triplet_character"], "dE_TT_lit": lit.at[c, "dE_TT_lit"],
                       "E_T_lit": lit.at[c, "E_T_lit"], "E_T_ml": o[c]["E_T"],
                       "lambda_S0_T1_ml": o[c]["lambda_S0_T1"], "lambda_T1_S0_ml": o[c]["lambda_T1_S0"],
                       "lambda_sum_ml": o[c]["lambda_S0_T1"] + o[c]["lambda_T1_S0"],
                       "lambda_S0_T1_lit": lit.at[c, "lambda_S0_T1_lit"], "lambda_T1_S0_lit": lit.at[c, "lambda_T1_S0_lit"]}
                row["barrier_ml_1E"] = pair_barrier(o[c], o["1E"])[1]
                row["barrier_ml_2E"] = pair_barrier(o[c], o["2E"])[1]
                row["barrier_lit_2E"] = lit.at[c, "barrier_lit_2E"]
                if pd.notna(lit.at[c, "lambda_T1_S0_lit"]):
                    # our E_T and substrate, the paper's catalyst reorganisation energies
                    pc = {"E_T": o[c]["E_T"], "lambda_S0_T1": lit.at[c, "lambda_S0_T1_lit"],
                          "lambda_T1_S0": lit.at[c, "lambda_T1_S0_lit"]}
                    row["barrier_ml_2E_paper_pc_lambda"] = pair_barrier(pc, o["2E"])[1]
                rows.append(row)
    catalysts = pd.DataFrame(rows)
    catalysts["is_pipi"] = catalysts.triplet_character_lit == "3pi,pi*"
    write(catalysts, "leonori_catalysts.csv")

    character = []
    for (model, estimate), c in catalysts.groupby(["model", "estimate"], sort=False):
        npi = c[~c.is_pipi]
        character.append({
            "model": model, "estimate": estimate, "n": len(c),
            "rho_lambda_gap": spearman(c.lambda_sum_ml, c.dE_TT_lit)[0],
            "rho_lambda_gap_fixed_E_T": partial_spearman(c.lambda_sum_ml, c.dE_TT_lit, c.E_T_ml),
            "rho_lambda_gap_within_npi": spearman(npi.lambda_sum_ml, npi.dE_TT_lit)[0],
            "rho_barrier_1E_gap": spearman(c.barrier_ml_1E, c.dE_TT_lit)[0],
            "rho_barrier_1E_gap_within_npi": spearman(npi.barrier_ml_1E, npi.dE_TT_lit)[0],
            "rho_lambda_T1_S0_gap": spearman(c.lambda_T1_S0_ml, c.dE_TT_lit)[0],
            "pearson_lambda_T1_S0_gap": float(np.corrcoef(c.lambda_T1_S0_ml, c.dE_TT_lit)[0, 1]),
            "rho_lambda_T1_S0_gap_within_npi": spearman(npi.lambda_T1_S0_ml, npi.dE_TT_lit)[0],
            "pearson_lambda_T1_S0_gap_within_npi": float(np.corrcoef(npi.lambda_T1_S0_ml, npi.dE_TT_lit)[0, 1]),
            "rho_lambda_S0_T1_gap": spearman(c.lambda_S0_T1_ml, c.dE_TT_lit)[0],
            "rho_lambda_S0_T1_gap_within_npi": spearman(npi.lambda_S0_T1_ml, npi.dE_TT_lit)[0],
            "pearson_E_T_gap": float(np.corrcoef(c.E_T_ml, c.dE_TT_lit)[0, 1]),
            "pearson_E_T_gap_within_npi": float(np.corrcoef(npi.E_T_ml, npi.dE_TT_lit)[0, 1]),
            "rho_E_T_gap": spearman(c.E_T_ml, c.dE_TT_lit)[0],
            "rho_E_T_gap_within_npi": spearman(npi.E_T_ml, npi.dE_TT_lit)[0],
            "auc_lambda_pipi": auc(-c.lambda_sum_ml, c.is_pipi),
            "auc_lambda_T1_S0_pipi": auc(-c.lambda_T1_S0_ml, c.is_pipi),
            "lambda_T1_S0_mean_pipi": c[c.is_pipi].lambda_T1_S0_ml.mean(),
            "lambda_T1_S0_mean_npi": npi.lambda_T1_S0_ml.mean(),
            "auc_E_T_ml_pipi": auc(-c.E_T_ml, c.is_pipi),
            "auc_E_T_lit_pipi": auc(-c.E_T_lit, c.is_pipi),
            "lambda_sum_mean_pipi": c[c.is_pipi].lambda_sum_ml.mean(),
            "lambda_sum_mean_npi": npi.lambda_sum_ml.mean()})
    write(pd.DataFrame(character), "leonori_character.csv")

    feas, summary = [], []
    for model, pred in models("leonori_2025").items():
        for estimate in ESTIMATES:
            pc = ours(pred.loc[ISOMERISATION_CATALYST], estimate)
            f = []
            for s in subs:
                dG, barrier = pair_barrier(pc, ours(pred.loc[s], estimate))
                f.append({"model": model, "estimate": estimate, "compound_id": s, "label": lit.at[s, "label"],
                          "reacts_lit": bool(lit.at[s, "reacts_lit"]), "pc": ISOMERISATION_CATALYST,
                          "dG_EnT_ml": dG, "barrier_ml": barrier,
                          "dG_EnT_lit": lit.at[s, "E_T_lit"] - lit.at[ISOMERISATION_CATALYST, "E_T_lit"]})
            f = pd.DataFrame(f); feas.append(f)
            yes, no = f[f.reacts_lit], f[~f.reacts_lit]
            summary.append({"model": model, "estimate": estimate, "n_react": len(yes), "n_not": len(no),
                            "margin_barrier": no.barrier_ml.min() - yes.barrier_ml.max(),
                            "margin_dG": no.dG_EnT_ml.min() - yes.dG_EnT_ml.max(),
                            "closest_reacting": yes.loc[yes.barrier_ml.idxmax(), "label"],
                            "auc_barrier": auc(-f.barrier_ml, f.reacts_lit), "auc_dG": auc(-f.dG_EnT_ml, f.reacts_lit)})
    write(pd.concat(feas, ignore_index=True), "leonori_feasibility.csv")
    write(pd.DataFrame(summary), "leonori_feasibility_summary.csv")


    quench = []
    measured = [(c, lit.at[c, "k_q"], lit.at[c, "k_q_quencher"], lit.at[c, "k_q_method"]) for c in cats if pd.notna(lit.at[c, "k_q"])]
    measured += [(ISOMERISATION_CATALYST, lit.at[ISOMERISATION_CATALYST, "k_q_Z1"], "(Z)-1", lit.at[ISOMERISATION_CATALYST, "k_q_method"])]
    quencher_id = {"(E)-1": "1E", "(Z)-1": "1Z", "27": "27"}
    for model, pred in models("leonori_2025").items():
        for estimate in ESTIMATES:
            for c, k_q, q, method in measured:
                quench.append({"model": model, "estimate": estimate, "catalyst": c, "quencher": q, "k_q": k_q,
                               "method": method, "barrier_exp_eyring": eyring_barrier(k_q),
                               "barrier_ml": pair_barrier(ours(pred.loc[c], estimate), ours(pred.loc[quencher_id[q]], estimate))[1]})
            pc = ours(pred.loc[ISOMERISATION_CATALYST], estimate)
            a, b = (ours(pred.loc[s], estimate) for s in ("1E", "1Z"))
            full, dg_part, lam_part = split_difference(
                barrier_asymmetric, (a["E_T"] - pc["E_T"], *pair_lambdas(pc, a)), (b["E_T"] - pc["E_T"], *pair_lambdas(pc, b)))
            quench.append({"model": model, "estimate": estimate, "catalyst": ISOMERISATION_CATALYST,
                           "quencher": "(E)-1 minus (Z)-1", "k_q": np.nan, "method": "ratio of the two constants",
                           "ddG_exp": -RT * np.log(lit.at[ISOMERISATION_CATALYST, "k_q"] / lit.at[ISOMERISATION_CATALYST, "k_q_Z1"]),
                           "ddG_ml": full, "ddG_ml_dG_part": dg_part, "ddG_ml_lambda_part": lam_part,
                           "ddG_EnT_ml": a["E_T"] - b["E_T"]})
    write(pd.DataFrame(quench), "leonori_quenching.csv")

    print(pd.DataFrame(character).round(2).to_string(index=False))
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
