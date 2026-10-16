# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Hofer, Bertrams, Kerzig, Bach 2026: allene amides cyclise to beta-lactams by triplet energy
transfer -- does the barrier explain a rate ratio that the triplet energy alone cannot?

    uv run benchmark/literature/bach_allene.py

The acceptor is non-vertical by bending: the central allene carbon rehybridises from sp to
sp2 in the triplet. The authors measure the quenching of 7a by two donors (xanthone, XT, and
thioxanthone, TXT; k_q 1.1e9 and ~3e7 M-1 s-1), argue within the Sandros equation that the
rate difference demands an allene triplet "almost 300 kJ/mol" high, and conclude that their
own adiabatic DFT E_T (45.75 kcal/mol) is "at least 0.5 eV too low". Our reading: the
adiabatic E_T is fine, and the large bending reorganisation sets how strongly the donor's
driving force is transmitted to the rate. The test is that one ratio.

All tests below were registered before any prediction was looked at, except those marked
POST HOC.

CONVENTION
    Asymmetric form, unscaled; the paper uses no Marcus form. Every property of every molecule
    is ours, donors included. Rates are compared as activation-limited constants,
    k_act = k_q k_d / (k_d - k_q), with the paper's k_d(MeCN) = 1.9e10 M-1 s-1, so the
    experimental quantity is ddG_exp = RT ln(k_act,XT / k_act,TXT).

THE TESTS
    kq (primary)  ddG_ml = barrier(TXT -> 7a) - barrier(XT -> 7a) against ddG_exp. Passes when
                  ddG_ml lies within a factor of two of ddG_exp; the pre-registered verdict is
                  the deployed model's mixture mean, "robust" only if >= 10 of the 12 runs pass.
                  Baselines on the same band: the Sandros (triplet-energy) screen with our E_T,
                  k_q = k_d / (1 + exp(dG/RT)), on the k_act basis (registered; analytically
                  dG_TXT - dG_XT) and on the uncorrected k_q basis (added, before the results
                  were read, when the k_act form overflowed); the same with the literature donor
                  E_T; the lambda term alone, (lambda_fwd,TXT - lambda_fwd,XT) / 4; and a null
                  "vertical acceptor" with both 7a lambdas set to zero. The paper's own straw
                  man -- Sandros on its adiabatic DFT E_T -- and the Sandros-apparent E_T of 7a
                  from each rate are model-free and written once. One acceptor and two donors:
                  this is ONE independent ratio. The TXT rate is an estimate from a single
                  concentration; absolute barriers are descriptive only.
    donor pair    descriptive (POST HOC, user request 2026-10-06): the same difference,
                  barrier(TXT -> s) - barrier(XT -> s), for 7a and every scope allene s, to show
                  whether the predicted XT/TXT difference is a property of the donor pair. Only
                  7a has a measured ratio.

THE SERIES, AND WHAT IS EXCLUDED
    Scope: 7e and 7o are flagged as excluded (see SI). The deuterated 7f-d1/d2 are kept as
    the table has them.

POST HOC
    kq      the barrier with the literature donor E_T; the Shapley split of ddG_ml into
            driving-force and reorganisation parts; the apparent Bronsted slope
            ddG / (dG_TXT - dG_XT) (0 or 1 under Sandros); the prefactor k_act exp(barrier/RT)
            implied by each rate; the predicted k_act ratio.

Writes   results/literature/bach_kq_summary.csv       the k_q ratio test and its baselines per model and estimate
         results/literature/bach_kq_reference.csv     the model-free numbers: ddG_exp, band, Sandros straw man
         results/literature/bach_donor_pair.csv       barrier(TXT) - barrier(XT) for 7a and every scope allene
         results/literature/bach_marcus_curve.csv     deployed barrier of 7a vs driving force, beside the Sandros expectation
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common import (ESTIMATES, RT, barrier_asymmetric, literature, models, ours, pair_barrier,
                    pair_lambdas, split_difference, write)

CASE = "bach_2026"
ACCEPTOR = "7a"
FAST, SLOW = "XT", "TXT"
K_DIFFUSION = 1.9e10            # M-1 s-1, MeCN, the paper's value (SI p.89, its ref 62)
BAND = 2.0                      # pass: ddG_ml within a factor of two of ddG_exp
ROBUST = 10                     # of 12 runs
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
    E_T_pc_lit = lit.dropna(subset=["pc_E_T_lit"]).groupby("pc_label").pc_E_T_lit.first()
    ddG_exp = RT * np.log(k_act(k_q[FAST]) / k_act(k_q[SLOW]))
    acceptor_lit = lit[lit.compound_id == ACCEPTOR].iloc[0]

    scope = lit[lit.experiment == "scope"].drop_duplicates("compound_id").set_index("compound_id")

    # ---------------------------------------------------------------- model-free reference
    E_single = brentq(lambda e: np.log(sandros(e - E_T_pc_lit[FAST]) / sandros(e - E_T_pc_lit[SLOW]))
                      - np.log(k_q[FAST] / k_q[SLOW]), 50, 90)
    E_dft = acceptor_lit.E_T_lit
    reference = {
        "k_d": K_DIFFUSION, "k_q_XT": k_q[FAST], "k_q_TXT": k_q[SLOW],
        "k_act_XT": k_act(k_q[FAST]), "k_act_TXT": k_act(k_q[SLOW]),
        "ddG_exp": ddG_exp, "ddG_band_low": ddG_exp / BAND, "ddG_band_high": ddG_exp * BAND,
        "ddG_exp_kq_basis": RT * np.log(k_q[FAST] / k_q[SLOW]),
        "E_T_lit_XT": E_T_pc_lit[FAST], "E_T_lit_TXT": E_T_pc_lit[SLOW],
        "bronsted_slope_exp": ddG_exp / (E_T_pc_lit[FAST] - E_T_pc_lit[SLOW]),
        # the paper's straw man: Sandros on its own adiabatic DFT E_T of 7a
        "E_T_7a_dft_lit": E_dft,
        "ddG_sandros_dft_k_act": (E_dft - E_T_pc_lit[SLOW]) - (E_dft - E_T_pc_lit[FAST]),
        "ddG_sandros_dft_kq": RT * np.log(sandros(E_dft - E_T_pc_lit[FAST]) / sandros(E_dft - E_T_pc_lit[SLOW])),
        "k_q_sandros_dft_XT": sandros(E_dft - E_T_pc_lit[FAST]), "k_q_sandros_dft_TXT": sandros(E_dft - E_T_pc_lit[SLOW]),
        # the E_T of 7a that Sandros needs to reproduce each rate on its own, and both together
        "E_T_7a_sandros_apparent_XT": E_T_pc_lit[FAST] + RT * np.log(K_DIFFUSION / k_q[FAST] - 1),
        "E_T_7a_sandros_apparent_TXT": E_T_pc_lit[SLOW] + RT * np.log(K_DIFFUSION / k_q[SLOW] - 1),
        "E_T_7a_sandros_single": E_single,
        "k_q_sandros_single_XT": sandros(E_single - E_T_pc_lit[FAST]),
        "k_q_sandros_single_TXT": sandros(E_single - E_T_pc_lit[SLOW]),
    }
    write(pd.DataFrame([reference]), "bach_kq_reference.csv")

    # ---------------------------------------------------------------- per model and estimate
    kq_rows, donor_pair = [], []
    for model, pred in preds.items():
        for estimate in ESTIMATES:
            o = {c: ours(pred.loc[c], estimate) for c in pred.index}
            acc = o[ACCEPTOR]

            # the XT/TXT difference for 7a and every scope allene (descriptive)
            for s_id in [ACCEPTOR] + [c for c in scope.index if c != ACCEPTOR]:
                b_xt = pair_barrier(o[f"PC_{FAST}"], o[s_id])[1]
                b_txt = pair_barrier(o[f"PC_{SLOW}"], o[s_id])[1]
                donor_pair.append({"model": model, "estimate": estimate, "substrate": s_id,
                                   "measured": s_id == ACCEPTOR, "excluded": SCOPE_EXCLUDED.get(s_id, ""),
                                   "E_T_ml": o[s_id]["E_T"], "barrier_XT_ml": b_xt, "barrier_TXT_ml": b_txt,
                                   "ddG_ml": b_txt - b_xt, "ddG_exp": ddG_exp if s_id == ACCEPTOR else np.nan})

            # ------------------------------------------------ kq: the primary test
            xt, txt = o[f"PC_{FAST}"], o[f"PC_{SLOW}"]
            a = (acc["E_T"] - txt["E_T"], *pair_lambdas(txt, acc))
            b = (acc["E_T"] - xt["E_T"], *pair_lambdas(xt, acc))
            ddG, dg_part, lam_part = split_difference(barrier_asymmetric, a, b)
            dG_xt_lit, dG_txt_lit = acc["E_T"] - E_T_pc_lit[FAST], acc["E_T"] - E_T_pc_lit[SLOW]
            vertical = dict(acc, lambda_S0_T1=0.0, lambda_T1_S0=0.0)
            base = {
                "ddG_sandros_k_act": a[0] - b[0],
                "ddG_sandros_kq": RT * np.log(sandros(b[0]) / sandros(a[0])),
                "ddG_sandros_lit_pc_k_act": dG_txt_lit - dG_xt_lit,
                "ddG_sandros_lit_pc_kq": RT * np.log(sandros(dG_xt_lit) / sandros(dG_txt_lit)),
                "ddG_lambda_only": (a[1] - b[1]) / 4,
                "ddG_null_vertical_acceptor": pair_barrier(txt, vertical)[1] - pair_barrier(xt, vertical)[1],
            }
            posthoc_lit = (barrier_asymmetric(dG_txt_lit, *a[1:]) - barrier_asymmetric(dG_xt_lit, *b[1:]))
            row = {"model": model, "estimate": estimate, "primary": model == "deployed" and estimate == "mean",
                   "E_T_7a_ml": acc["E_T"], "E_T_XT_ml": xt["E_T"], "E_T_TXT_ml": txt["E_T"],
                   "dG_XT": b[0], "dG_TXT": a[0], "lambda_forward_XT": b[1], "lambda_reverse_XT": b[2],
                   "lambda_forward_TXT": a[1], "lambda_reverse_TXT": a[2],
                   "barrier_XT": barrier_asymmetric(*b), "barrier_TXT": barrier_asymmetric(*a),
                   "ddG_ml": ddG, "ddG_exp": ddG_exp, "pass": in_band(ddG, ddG_exp)}
            for name, value in base.items():
                row[name] = value
                row[f"pass_{name[4:]}"] = in_band(value, ddG_exp)
            row.update({
                "posthoc_ddG_barrier_lit_pc": posthoc_lit, "posthoc_pass_barrier_lit_pc": in_band(posthoc_lit, ddG_exp),
                "posthoc_ddG_dG_part": dg_part, "posthoc_ddG_lambda_part": lam_part,
                "posthoc_bronsted_slope_ml": ddG / (a[0] - b[0]) if a[0] != b[0] else np.nan,
                "posthoc_prefactor_XT": k_act(k_q[FAST]) * np.exp(row["barrier_XT"] / RT),
                "posthoc_prefactor_TXT": k_act(k_q[SLOW]) * np.exp(row["barrier_TXT"] / RT),
                "posthoc_k_act_ratio_ml": np.exp(ddG / RT)})
            kq_rows.append(row)

    kq_table = pd.DataFrame(kq_rows)
    write(kq_table, "bach_kq_summary.csv")
    write(pd.DataFrame(donor_pair), "bach_donor_pair.csv")

    # For a small explanatory panel: the deployed model's barrier for 7a as a function of the
    # driving force, with the reorganisation energies of each donor pair held fixed, beside the
    # triplet-energy-only (Sandros) expectation, which is flat (diffusion-limited) once exergonic.
    dep = kq_table[(kq_table.model == "deployed") & (kq_table.estimate == "mean")].iloc[0]
    grid = np.round(np.arange(-35.0, 5.01, 0.25), 2)
    write(pd.DataFrame({"dG_EnT": grid,
                        "barrier_ml_XT_lambdas": [barrier_asymmetric(g, dep.lambda_forward_XT, dep.lambda_reverse_XT) for g in grid],
                        "barrier_ml_TXT_lambdas": [barrier_asymmetric(g, dep.lambda_forward_TXT, dep.lambda_reverse_TXT) for g in grid],
                        "ln_k_over_k_d_sandros": [float(-np.log1p(np.exp(g / RT))) for g in grid]}), "bach_marcus_curve.csv")

    # ---------------------------------------------------------------- stdout
    n = len(kq_table)
    print(f"\nkq (primary): ddG_exp {ddG_exp:.2f}, band {ddG_exp / BAND:.2f} to {ddG_exp * BAND:.2f} kcal/mol")
    print(kq_table[["model", "estimate", "dG_XT", "dG_TXT", "barrier_XT", "barrier_TXT", "ddG_ml", "pass",
                    "ddG_sandros_k_act", "ddG_sandros_kq", "ddG_lambda_only", "ddG_null_vertical_acceptor",
                    "posthoc_ddG_barrier_lit_pc"]].round(2).to_string(index=False))
    primary = bool(kq_table.loc[kq_table.primary, "pass"].iloc[0])
    passes = int(kq_table["pass"].sum())
    print(f"pass of {n}: barrier {passes} (deployed mean {'PASS' if primary else 'FAIL'}; "
          f"robust {'yes' if passes >= ROBUST else 'no'}, needs {ROBUST}); "
          + ", ".join(f"{c[5:]} {int(kq_table[c].sum())}" for c in kq_table.columns if c.startswith("pass_"))
          + f"; post hoc barrier with literature donor E_T {int(kq_table.posthoc_pass_barrier_lit_pc.sum())}")
    print(f"Sandros on the paper's DFT E_T: ddG {reference['ddG_sandros_dft_kq']:.2f} (k_q), "
          f"{reference['ddG_sandros_dft_k_act']:.2f} (k_act); apparent E_T(7a) {reference['E_T_7a_sandros_apparent_XT']:.2f} "
          f"from XT, {reference['E_T_7a_sandros_apparent_TXT']:.2f} from TXT")


if __name__ == "__main__":
    main()
