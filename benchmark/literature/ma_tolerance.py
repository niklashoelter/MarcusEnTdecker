# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Ma 2023: functional-group tolerance of the azaarene reduction by energy transfer.

    uv run benchmark/literature/ma_tolerance.py

Symmetric barrier on lambda_forward, with the Ir sensitiser's published E_T and
lambda_T1_S0; only the substrates are predicted.

Groups: reduced (the five azaarenes, as free bases), tolerated (pendant groups a-k),
hat (naphthalene l, anthracene, phenanthrene; scored separately and folded in).

Reference: agreement with the paper's dG_EnT and barriers, over every row with a published
barrier and over the rows not excluded, with recomputed and printed barriers.
    reference_excluded         paper's free-base E_T > 90 kcal/mol (see SI)
    lambda_total_inconsistent  printed lambda_total != lambda_S0_T1 + pc lambda_T1_S0; the
                               barrier is recomputed from the components (barrier_lit_recomputed)

Writes   results/literature/ma_tolerance.csv           one row per compound
         results/literature/ma_reference.csv           agreement with the paper's DFT
"""

import numpy as np
import pandas as pd

from common import barrier_symmetric, literature, models, ours, write

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
    # tolerance 0.5 kcal/mol (values printed to one decimal)
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
        estimate = "mean"
        for compound_id in lit.index:
            o = ours(pred.loc[compound_id])
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

    print(pd.DataFrame(ref).query("model == 'deployed'").to_string(index=False))


if __name__ == "__main__":
    main()
