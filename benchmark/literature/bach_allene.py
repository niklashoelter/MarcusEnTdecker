# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Hofer, Bertrams, Kerzig, Bach 2026: allene amides cyclise to beta-lactams by triplet energy
transfer; quenching of 7a by xanthone (XT) and thioxanthone (TXT).

    uv run benchmark/literature/bach_allene.py

Asymmetric form, unscaled; all molecular properties ours. k_act = k_q k_d / (k_d - k_q),
k_d(MeCN) = 1.9e10 M-1 s-1; ddG_exp = RT ln(k_act,XT / k_act,TXT).

kq          ddG_ml = barrier(TXT -> 7a) - barrier(XT -> 7a) against ddG_exp; pass within a
            factor of two. E_T only: ddG from the Sandros k_q = k_d / (1 + exp(dG/RT)).
donor pair  barrier(TXT -> s) - barrier(XT -> s) for 7a and every scope allene s.

7e and 7o excluded (see SI); 7f-d1/d2 kept.

Writes   results/literature/bach_kq_summary.csv       the k_q ratio test, and the E_T-only difference
         results/literature/bach_kq_reference.csv     measured k_q, k_act and ddG_exp
         results/literature/bach_donor_pair.csv       barrier(TXT) - barrier(XT) for 7a and every scope allene
         results/literature/bach_marcus_curve.csv     deployed barrier of 7a vs driving force, beside the Sandros expectation
"""

import numpy as np
import pandas as pd

from common import (ESTIMATES, RT, barrier_asymmetric, literature, models, ours, pair_barrier,
                    pair_lambdas, write)

CASE = "bach_2026"
ACCEPTOR = "7a"
FAST, SLOW = "XT", "TXT"
K_DIFFUSION = 1.9e10            # M-1 s-1, MeCN, from the paper
BAND = 2.0                      # pass: ddG_ml within a factor of two of ddG_exp
SCOPE_EXCLUDED = {"7e": "excluded (see SI)", "7o": "excluded (see SI)"}


def k_act(k_q):
    """Activation-limited rate constant, diffusion removed."""
    return k_q * K_DIFFUSION / (K_DIFFUSION - k_q)


def sandros(dG):
    """Quenching constant of the Sandros equation: triplet energies only."""
    return K_DIFFUSION / (1 + np.exp(dG / RT))


def in_band(value, target):
    return bool(target / BAND <= value <= target * BAND)


def main():
    lit = literature(CASE)
    preds = models(CASE)

    # ---------------------------------------------------------------- experiment, from the table
    kq = lit[lit.experiment == "kq"].set_index("pc_label")
    k_q = {d: float(kq.at[d, "k_q"]) for d in (FAST, SLOW)}
    ddG_exp = RT * np.log(k_act(k_q[FAST]) / k_act(k_q[SLOW]))

    scope = lit[lit.experiment == "scope"].drop_duplicates("compound_id").set_index("compound_id")

    # ---------------------------------------------------------------- measured
    reference = {
        "k_d": K_DIFFUSION, "k_q_XT": k_q[FAST], "k_q_TXT": k_q[SLOW],
        "k_act_XT": k_act(k_q[FAST]), "k_act_TXT": k_act(k_q[SLOW]), "ddG_exp": ddG_exp,
    }
    write(pd.DataFrame([reference]), "bach_kq_reference.csv")

    # ---------------------------------------------------------------- per model and estimate
    kq_rows, donor_pair = [], []
    for model, pred in preds.items():
        for estimate in ESTIMATES:
            o = {c: ours(pred.loc[c], estimate) for c in pred.index}
            acc = o[ACCEPTOR]

            # XT/TXT barrier difference for 7a and every scope allene
            for s_id in [ACCEPTOR] + [c for c in scope.index if c != ACCEPTOR]:
                b_xt = pair_barrier(o[f"PC_{FAST}"], o[s_id])[1]
                b_txt = pair_barrier(o[f"PC_{SLOW}"], o[s_id])[1]
                donor_pair.append({"model": model, "estimate": estimate, "substrate": s_id,
                                   "measured": s_id == ACCEPTOR, "excluded": SCOPE_EXCLUDED.get(s_id, ""),
                                   "E_T_ml": o[s_id]["E_T"], "barrier_XT_ml": b_xt, "barrier_TXT_ml": b_txt,
                                   "ddG_ml": b_txt - b_xt, "ddG_exp": ddG_exp if s_id == ACCEPTOR else np.nan})

            # ------------------------------------------------ kq
            xt, txt = o[f"PC_{FAST}"], o[f"PC_{SLOW}"]
            a = (acc["E_T"] - txt["E_T"], *pair_lambdas(txt, acc))
            b = (acc["E_T"] - xt["E_T"], *pair_lambdas(xt, acc))
            ddG = barrier_asymmetric(*a) - barrier_asymmetric(*b)
            row = {"model": model, "estimate": estimate, "primary": model == "deployed" and estimate == "mean",
                   "E_T_7a_ml": acc["E_T"], "E_T_XT_ml": xt["E_T"], "E_T_TXT_ml": txt["E_T"],
                   "dG_XT": b[0], "dG_TXT": a[0], "lambda_forward_XT": b[1], "lambda_reverse_XT": b[2],
                   "lambda_forward_TXT": a[1], "lambda_reverse_TXT": a[2],
                   "barrier_XT": barrier_asymmetric(*b), "barrier_TXT": barrier_asymmetric(*a),
                   "ddG_ml": ddG, "ddG_exp": ddG_exp, "pass": in_band(ddG, ddG_exp),
                   "ddG_sandros_kq": RT * np.log(sandros(b[0]) / sandros(a[0]))}
            kq_rows.append(row)

    kq_table = pd.DataFrame(kq_rows)
    write(kq_table, "bach_kq_summary.csv")
    write(pd.DataFrame(donor_pair), "bach_donor_pair.csv")

    # Deployed barrier of 7a vs driving force at each donor pair's lambdas; Sandros ln(k/k_d).
    dep = kq_table[(kq_table.model == "deployed") & (kq_table.estimate == "mean")].iloc[0]
    grid = np.round(np.arange(-35.0, 5.01, 0.25), 2)
    write(pd.DataFrame({"dG_EnT": grid,
                        "barrier_ml_XT_lambdas": [barrier_asymmetric(g, dep.lambda_forward_XT, dep.lambda_reverse_XT) for g in grid],
                        "barrier_ml_TXT_lambdas": [barrier_asymmetric(g, dep.lambda_forward_TXT, dep.lambda_reverse_TXT) for g in grid],
                        "ln_k_over_k_d_sandros": [float(-np.log1p(np.exp(g / RT))) for g in grid]}), "bach_marcus_curve.csv")

    # ---------------------------------------------------------------- stdout
    print(f"\nkq: ddG_exp {ddG_exp:.2f}, band {ddG_exp / BAND:.2f} to {ddG_exp * BAND:.2f} kcal/mol")
    print(kq_table[["model", "estimate", "dG_XT", "dG_TXT", "barrier_XT", "barrier_TXT", "ddG_ml", "pass",
                    "ddG_sandros_kq"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
