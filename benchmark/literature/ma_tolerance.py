"""
Ma 2023: which functional groups survive energy-transfer catalysis?

    uv run --extra ml python benchmark/literature/ma_tolerance.py

The paper reduces azaarenes by energy transfer and finds a long list of reducible pendant
groups untouched. The question for us is a classification: do our predicted driving force and
reorganisation energy, combined in the barrier, put the reduced azaarenes below the tolerated
groups? The triplet-energy screen alone -- exergonic means quenched -- is scored beside it.

CONVENTION
    The paper's barrier is symmetric on a single reorganisation energy, lambda_A(singlet) +
    lambda_D(triplet), which in our naming is exactly lambda_forward, with the Ir sensitiser's
    published E_T (63.9, not the 62.9 misprinted in its Figure 4a) and lambda_T1_S0. Only the
    substrates are predicted.

THE GROUPS, FROM THE REACTION OUTCOME
    reduced     the five azaarenes the paper reduces (products 16-20). They react as Lewis-acid
                adducts -- without the Lewis acid there is no product -- and the model has no
                Lewis acid, so the free bases stand in for them, as in the paper's own analysis.
    tolerated   the pendant groups a-k, which survive.
    hat         naphthalene (l), anthracene, phenanthrene: they survive, but the paper assigns
                that to the hydrogen-atom-transfer step, because their energy-transfer barriers
                are low in its own DFT too. Scored separately and also folded in, both reported.

THE DFT REFERENCE
    Agreement with the paper's driving forces (its printed dG_EnT) and barriers. Two problems in
    the published table are handled separately, each by a rule fixed on the table alone:
    reference_excluded      the free-base E_T implied by the paper's own Gibbs energies (SI
                            Table S9) exceeds 90 kcal/mol (4,7-phenanthroline, thiophene c,
                            iodobenzene d: 101-104), far above the related chromophores of the
                            same study; these rows are left out of every reference statistic.
    lambda_total_inconsistent  the printed lambda_total is not lambda_S0_T1 + 6.5 (a, g, i),
                            so the printed barrier cannot be reproduced from the paper's own
                            components. Their E_T are plausible; the barrier is recomputed from
                            the paper's components (barrier_lit_recomputed) and used instead.
    Reported over every row with a published barrier, and over the rows not excluded, with the
    recomputed and with the printed barriers.

Writes   results/literature/ma_tolerance.csv           one row per compound, model and estimate
         results/literature/ma_classification.csv      separation scores per model and estimate
         results/literature/ma_reference.csv           agreement with the paper's DFT
"""

import numpy as np
import pandas as pd

from common import ESTIMATES, auc, barrier_symmetric, literature, models, ours, write

REDUCED = ["quinoline", "isoquinoline", "phenanthridine", "benzofquinoline", "47phenanthroline"]
HAT = ["l", "anthracene", "phenanthrene"]
E_T_IMPLAUSIBLE = 90.0


def group(compound_id):
    if compound_id in REDUCED:
        return "reduced"
    if compound_id in HAT:
        return "hat"
    return "tolerated"


def lambda_total_inconsistent(row):
    """Printed lambda_total is not lambda_S0_T1 + the sensitiser's lambda_T1_S0."""
    # printed to one decimal, so sums may be off by 0.1; the inconsistent rows are off by 1.4-4.2
    return bool(pd.notna(row.lambda_total_lit) and pd.notna(row.lambda_S0_T1_lit)
                and abs(row.lambda_total_lit - (row.lambda_S0_T1_lit + row.pc_lambda_T1_S0_lit)) > 0.5)


def reference_excluded(row):
    """The paper's free-base E_T is implausibly high (> 90 kcal/mol)."""
    return bool(pd.notna(row.E_T_lit) and row.E_T_lit > E_T_IMPLAUSIBLE)


def recomputed_barrier(row):
    """The paper's symmetric barrier from its own components, lambda_S0_T1 + 6.5 and dG_EnT."""
    if pd.isna(row.lambda_S0_T1_lit) or pd.isna(row.dG_EnT_lit):
        return row.barrier_lit
    return float(barrier_symmetric(row.dG_EnT_lit, row.lambda_S0_T1_lit + row.pc_lambda_T1_S0_lit))


def main():
    lit = literature("ma_2023").set_index("compound_id")
    pc_E_T, pc_lambda = float(lit.pc_E_T_lit.iloc[0]), float(lit.pc_lambda_T1_S0_lit.iloc[0])

    rows = []
    for model, pred in models("ma_2023").items():
        for estimate in ESTIMATES:
            for compound_id in lit.index:
                o = ours(pred.loc[compound_id], estimate)
                lam, dG = o["lambda_S0_T1"] + pc_lambda, o["E_T"] - pc_E_T
                rows.append({"model": model, "estimate": estimate, "compound_id": compound_id,
                             "label": lit.at[compound_id, "label"], "group": group(compound_id),
                             "E_T_ml": o["E_T"], "lambda_S0_T1_ml": o["lambda_S0_T1"],
                             "dG_EnT_ml": dG, "lambda_total_ml": lam,
                             "barrier_ml": float(barrier_symmetric(dG, lam)),
                             "dG_EnT_lit": lit.at[compound_id, "dG_EnT_lit"],
                             "barrier_lit": lit.at[compound_id, "barrier_lit"],
                             "barrier_lit_recomputed": recomputed_barrier(lit.loc[compound_id]),
                             "reference_excluded": reference_excluded(lit.loc[compound_id]),
                             "lambda_total_inconsistent": lambda_total_inconsistent(lit.loc[compound_id])})
    table = pd.DataFrame(rows)
    write(table, "ma_tolerance.csv")

    scores = []
    for (model, estimate), t in table.groupby(["model", "estimate"], sort=False):
        for arenes in ("separate", "counted as tolerated"):
            s = t[t.group != "hat"] if arenes == "separate" else t
            reactive = s.group == "reduced"
            red, tol = s[reactive], s[~reactive]
            scores.append({
                "model": model, "estimate": estimate, "hat_arenes": arenes,
                "n_reduced": len(red), "n_tolerated": len(tol),
                # threshold-free: how often a reduced compound lies below a tolerated one
                "auc_barrier": auc(-s.barrier_ml, reactive),
                "auc_dG": auc(-s.dG_EnT_ml, reactive),
                "margin_barrier": tol.barrier_ml.min() - red.barrier_ml.max(),
                "lowest_tolerated": tol.loc[tol.barrier_ml.idxmin(), "label"],
                # the triplet-energy screen: exergonic is called quenched
                "E_T_screen_correct": int(((s.dG_EnT_ml < 0) == reactive).sum()),
                "E_T_screen_wrong": ", ".join(s[(s.dG_EnT_ml < 0) != reactive].label),
                "barrier_best_threshold_correct": max(int(((s.barrier_ml < th) == reactive).sum())
                                                      for th in np.unique(s.barrier_ml) + 1e-9),
                "n": len(s)})
    write(pd.DataFrame(scores), "ma_classification.csv")

    ref = []
    for (model, estimate), t in table.groupby(["model", "estimate"], sort=False):
        has = t.barrier_lit.notna()
        for subset, sel, ref_col in (("all with a published barrier, printed", has, "barrier_lit"),
                                     ("E_T plausible, barrier recomputed", has & ~t.reference_excluded, "barrier_lit_recomputed"),
                                     ("E_T plausible, barrier printed", has & ~t.reference_excluded, "barrier_lit")):
            s = t[sel]
            g = s.dropna(subset=["dG_EnT_lit"])
            ref.append({"model": model, "estimate": estimate, "subset": subset, "n": len(s),
                        "pearson_r": float(np.corrcoef(s.barrier_ml, s[ref_col])[0, 1]),
                        "spearman_rho": float(s.barrier_ml.rank().corr(s[ref_col].rank())),
                        "mae_barrier": float((s.barrier_ml - s[ref_col]).abs().mean()),
                        "n_dG": len(g), "pearson_r_dG": float(np.corrcoef(g.dG_EnT_ml, g.dG_EnT_lit)[0, 1]),
                        "spearman_rho_dG": float(g.dG_EnT_ml.rank().corr(g.dG_EnT_lit.rank())),
                        "mae_dG": float((g.dG_EnT_ml - g.dG_EnT_lit).abs().mean()),
                        "excluded": ", ".join(t[has & ~sel].label)})
    write(pd.DataFrame(ref), "ma_reference.csv")

    d = pd.DataFrame(scores)
    print(d[(d.model == "deployed")].drop(columns=["model"]).to_string(index=False))
    print(pd.DataFrame(ref).query("model == 'deployed'").to_string(index=False))


if __name__ == "__main__":
    main()
