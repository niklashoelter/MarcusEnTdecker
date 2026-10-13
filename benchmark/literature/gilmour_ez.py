# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Molloy, ..., Gilmour 2020: does the barrier predict which beta-boryl acrylate isomer accumulates?

    uv run benchmark/literature/gilmour_ez.py

The Sole-Daura case has one substrate our reorganisation term gets wrong: a beta-boryl
acrylate, whose E isomer accumulates. This paper is that substrate class. With thioxanthone,
every boron substrate isomerises to the isomer with boron cis to the carbonyl (92 to >99 %),
which the authors attribute to the C(sp2)-B bond turning by 90 degrees in that isomer, taking
boron out of conjugation and making re-excitation inefficient. The paper reports no energies,
so the only test is the direction: the B-cis isomer should come out with the higher barrier.

The comparison is by geometry (boron trans or cis to the carbonyl), never by the E/Z label,
which flips with the substituents because boron ranks low in CIP.

CONVENTION
    Asymmetric form, unscaled, as for Sole-Daura. Thioxanthone at its experimental E_T from
    the paper's SI Table S1, with our predicted thioxanthone lambdas; Sole-Daura's published TX
    lambdas are run as a sensitivity check (`pc_source`).

THE SERIES
    The 25 scope substrates that isomerise (1-25). Reported beside them, not scored: 26 (no
    carbonyl; low conversion) and the controls 27-29, which do not isomerise. The sensitiser
    screen of SI Table S1 mixes wavelengths and times and is not used.

Writes   results/literature/gilmour_ez.csv          one row per substrate, model, estimate and TX source
         results/literature/gilmour_ez_summary.csv  the direction tally per model, estimate and TX source
"""

import pandas as pd

from common import ESTIMATES, barrier_asymmetric, literature, models, ours, pair_lambdas, split_difference, write

TX_LAMBDAS_SOLE_DAURA = {"lambda_S0_T1": 4.1, "lambda_T1_S0": 4.8}


def main():
    lit = literature("gilmour_2020")
    tx = lit[lit.pc_label == "TX"].drop_duplicates("compound_id").copy()
    tx["substrate"] = tx.compound_id.str[:-1]
    tx_E_T = float(tx.pc_E_T_lit.iloc[0])

    rows = []
    for model, pred in models("gilmour_2020").items():
        for estimate in ESTIMATES:
            ours_tx = ours(pred.loc["PC_TX"], estimate)
            sources = {"ours": {"E_T": tx_E_T, "lambda_S0_T1": ours_tx["lambda_S0_T1"], "lambda_T1_S0": ours_tx["lambda_T1_S0"]},
                       "Sole-Daura": {"E_T": tx_E_T, **TX_LAMBDAS_SOLE_DAURA}}
            for source, pc in sources.items():
                for substrate, g in tx.groupby("substrate", sort=False):
                    trans = g[g.geometry.str.contains("trans")]
                    cis = g[g.geometry.str.contains("cis")]
                    if len(trans) != 1 or len(cis) != 1:
                        continue
                    t, c = ours(pred.loc[trans.compound_id.iloc[0]], estimate), ours(pred.loc[cis.compound_id.iloc[0]], estimate)
                    a = (t["E_T"] - pc["E_T"], *pair_lambdas(pc, t))
                    b = (c["E_T"] - pc["E_T"], *pair_lambdas(pc, c))
                    full, dg_part, lam_part = split_difference(barrier_asymmetric, a, b)
                    rows.append({"model": model, "estimate": estimate, "pc_source": source, "substrate": substrate,
                                 "outcome_lit": trans.outcome.iloc[0], "geometry_start": trans.geometry.iloc[0],
                                 "z_percent_exp": trans.z_percent_exp.iloc[0],
                                 "barrier_trans_ml": barrier_asymmetric(*a), "barrier_cis_ml": barrier_asymmetric(*b),
                                 # < 0: the trans isomer is sensitised faster, so the cis isomer accumulates (observed)
                                 "ddG_trans_minus_cis_ml": full, "ddG_ml_dG_part": dg_part, "ddG_ml_lambda_part": lam_part,
                                 "dE_T_cis_minus_trans_ml": c["E_T"] - t["E_T"],
                                 "correct": full < 0 if trans.outcome.iloc[0] == "isomerised" else None})
    table = pd.DataFrame(rows)
    write(table, "gilmour_ez.csv")

    scored = table[table.outcome_lit == "isomerised"]
    summary = (scored.groupby(["model", "estimate", "pc_source"], sort=False)
               .agg(n=("correct", "size"), correct=("correct", "sum"),
                    lambda_part_wrong=("ddG_ml_lambda_part", lambda s: int((s > 0).sum())),
                    dG_part_right=("ddG_ml_dG_part", lambda s: int((s < 0).sum())),
                    dE_T_median=("dE_T_cis_minus_trans_ml", "median")).reset_index())
    write(summary, "gilmour_ez_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
