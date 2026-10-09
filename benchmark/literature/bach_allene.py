"""
Hofer, Bertrams, Kerzig, Bach 2026: allene amides cyclise to beta-lactams by triplet energy
transfer -- does the barrier explain a rate ratio that the triplet energy alone cannot?

    uv run python benchmark/literature/bach_allene.py

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
    Asymmetric form, alpha = 1; the paper uses no Marcus form. Every property of every molecule
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
    screen        the sensitiser screen on 7a (SI Table 1): reacts = lactam isolated under any
                  reported condition. AUC of -barrier against AUC of -dG (the E_T screen) and of
                  -lambda_forward. Passes when AUC(barrier) = 1; "beyond the E_T screen" only if
                  AUC(barrier) > AUC(dG) in >= 10 of 12 runs. The literature donor E_T exists
                  only for reacting donors, so no AUC is possible with it (written as NaN).
                  One acceptor: this tests donor ordering only. Yields among positives are
                  confounded by wavelength, solvent and time and are not used.
    reference     descriptive: our 7a E_T and lambda_T1_S0 against the paper's B3LYP/6-31G
                  adiabatic E_T and vertical-minus-adiabatic gap (= our lambda_T1_S0; the paper
                  gives no lambda_S0_T1), and our donor E_T against the literature values.
    donor pair    descriptive (POST HOC, user request 2026-10-06): the same difference,
                  barrier(TXT -> s) - barrier(XT -> s), for 7a and every scope allene s, to show
                  whether the predicted XT/TXT difference is a property of the donor pair. Only
                  7a has a measured ratio.
    scope         descriptive and declared non-evidentiary: barrier(XT -> substrate) across the
                  scope and its Spearman rho with yield. The paper attributes the yields to the
                  hydrogen-atom-transfer step, not to energy transfer.

THE SERIES, AND WHAT IS EXCLUDED
    Donors: the nine organic sensitisers of the screen. The two iridium complexes cannot be
    encoded and the paper gives no lambda for them, so they are out of every test; the
    no-catalyst entry is not a donor. Screen sensitivity (reported, never promoted): drop the
    negatives whose every entry is flagged "degradation" (benzophenone, anthraquinone, 4CzIPN),
    since a consumed sensitiser does not show that energy transfer failed; 2,7-Br2-TXT is then
    the only negative. Scope: 7e (photoredox side reaction, per the paper) and 7o
    (decomposition, no yield) are excluded as registered; the all-in numbers stand beside them.
    7n gives a different triplet product and counts as reacting. The deuterated 7f-d1/d2 are
    kept as the table has them, although the model is not isotope-aware.

POST HOC
    kq      the barrier with the literature donor E_T; the Shapley split of ddG_ml into
            driving-force and reorganisation parts; the apparent Bronsted slope
            ddG / (dG_TXT - dG_XT) (0 or 1 under Sandros); the prefactor k_act exp(barrier/RT)
            implied by each rate; the predicted k_act ratio.
    screen  AUCs without acetone, which was the solvent and so needs no favourable barrier.
            Fac-Ir(ppy)3 with Sole-Daura's published parameters (their Ir-B) is evaluated in
            the pair table as an illustration and kept out of every score.

Writes   results/literature/bach_allene.csv           one row per donor-acceptor pair, model and estimate
         results/literature/bach_kq_summary.csv       the k_q ratio test and its baselines per model and estimate
         results/literature/bach_kq_reference.csv     the model-free numbers: ddG_exp, band, Sandros straw man
         results/literature/bach_screen_summary.csv   the screen AUCs per model and estimate
         results/literature/bach_dft_reference.csv    7a and the donors against the paper's values
         results/literature/bach_scope_summary.csv    the scope barriers and rho with yield, both subsets
         results/literature/bach_donor_pair.csv       barrier(TXT) - barrier(XT) for 7a and every scope allene
         results/literature/bach_marcus_curve.csv     deployed barrier of 7a vs driving force, beside the Sandros expectation
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common import (ESTIMATES, RT, auc, barrier_asymmetric, literature, models, ours, pair_barrier,
                    pair_lambdas, spearman, split_difference, write)

CASE = "bach_2026"
ACCEPTOR = "7a"
FAST, SLOW = "XT", "TXT"
K_DIFFUSION = 1.9e10            # M-1 s-1, MeCN, the paper's value (SI p.89, its ref 62)
BAND = 2.0                      # pass: ddG_ml within a factor of two of ddG_exp
ROBUST = 10                     # of 12 runs
SCOPE_EXCLUDED = {"7e": "photoredox side reaction (paper)", "7o": "decomposition, no yield"}
ILLUSTRATION_PC = ("sole_daura_2024", "Ir-B")    # fac-Ir(ppy)3, post hoc


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
    some = next(iter(preds.values()))

    # ---------------------------------------------------------------- experiment, from the table
    kq = lit[lit.experiment == "kq"].set_index("pc_label")
    k_q = {d: float(kq.at[d, "k_q"]) for d in (FAST, SLOW)}
    E_T_pc_lit = lit.dropna(subset=["pc_E_T_lit"]).groupby("pc_label").pc_E_T_lit.first()
    ddG_exp = RT * np.log(k_act(k_q[FAST]) / k_act(k_q[SLOW]))
    acceptor_lit = lit[lit.compound_id == ACCEPTOR].iloc[0]

    screen = lit[(lit.experiment == "screen") & (lit.compound_id == ACCEPTOR)]
    by_donor = screen.groupby("pc_label")
    donors = [d for d in screen.pc_label.unique() if f"PC_{d}" in some.index]
    not_encoded = [d for d in screen.pc_label.unique() if d not in donors]
    best_yield = by_donor.yield_percent_exp.max()
    reacts = {d: bool(best_yield[d] > 0) for d in donors}
    degradation_only = {d: (not reacts[d]) and bool(by_donor.get_group(d).outcome.str.contains("degradation").all())
                        for d in donors}

    scope = lit[lit.experiment == "scope"].drop_duplicates("compound_id").set_index("compound_id")

    sd = literature(ILLUSTRATION_PC[0])
    sd = sd[sd.pc_label == ILLUSTRATION_PC[1]].iloc[0]
    illustration = {"E_T": sd.pc_E_T_lit, "lambda_S0_T1": sd.pc_lambda_S0_T1_lit, "lambda_T1_S0": sd.pc_lambda_T1_S0_lit}

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

    # ---------------------------------------------------------------- pairs
    pairs, kq_rows, screen_rows, dft_rows, scope_rows, donor_pair = [], [], [], [], [], []
    for model, pred in preds.items():
        for estimate in ESTIMATES:
            o = {c: ours(pred.loc[c], estimate) for c in pred.index}
            acc = o[ACCEPTOR]

            def pair(pc, substrate, substrate_id, pc_label, pc_source, group, **extra):
                forward, reverse = pair_lambdas(pc, substrate)
                dG, barrier = pair_barrier(pc, substrate)
                row = {"model": model, "estimate": estimate, "group": group, "pc_label": pc_label,
                       "pc_source": pc_source, "substrate": substrate_id,
                       "pc_E_T": pc["E_T"], "pc_lambda_S0_T1": pc["lambda_S0_T1"], "pc_lambda_T1_S0": pc["lambda_T1_S0"],
                       "pc_E_T_lit": E_T_pc_lit.get(pc_label, np.nan),
                       "E_T_ml": substrate["E_T"], "lambda_S0_T1_ml": substrate["lambda_S0_T1"],
                       "lambda_T1_S0_ml": substrate["lambda_T1_S0"], "E_T_sd_ml": substrate["E_T_sd"],
                       "lambda_S0_T1_sd_ml": substrate["lambda_S0_T1_sd"], "lambda_T1_S0_sd_ml": substrate["lambda_T1_S0_sd"],
                       "dG_EnT_ml": dG, "lambda_forward_ml": forward, "lambda_reverse_ml": reverse, "barrier_ml": barrier}
                row.update(extra)
                pairs.append(row)
                return row

            # the screen on 7a (and the two k_q pairs inside it)
            sc = []
            for d in donors:
                sc.append(pair(o[f"PC_{d}"], acc, ACCEPTOR, d, "ours", "screen",
                               reacts_exp=reacts[d], best_yield_percent_exp=best_yield[d],
                               degradation_only=degradation_only[d], k_q=k_q.get(d, np.nan)))
            pair(illustration, acc, ACCEPTOR, "Ir(ppy)3", "Sole-Daura 2024 Ir-B (post hoc illustration)", "illustration")
            # the scope with XT
            for s in scope.index:
                pair(o[f"PC_{FAST}"], o[s], s, FAST, "ours", "scope",
                     yield_percent_exp=scope.at[s, "yield_percent_exp"], outcome_lit=scope.at[s, "outcome"],
                     excluded=SCOPE_EXCLUDED.get(s, ""))

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

            # ------------------------------------------------ screen
            g = pd.DataFrame(sc)
            kept = g[~g.degradation_only]
            no_acetone = g[g.pc_label != "acetone"]
            lit_pc = g.dropna(subset=["pc_E_T_lit"])
            r = {"model": model, "estimate": estimate, "n_react": int(g.reacts_exp.sum()), "n_not": int((~g.reacts_exp).sum()),
                 "auc_barrier": auc(-g.barrier_ml, g.reacts_exp), "auc_dG": auc(-g.dG_EnT_ml, g.reacts_exp),
                 "auc_lambda_forward": auc(-g.lambda_forward_ml, g.reacts_exp),
                 "auc_dG_lit": auc(-(lit_pc.E_T_ml - lit_pc.pc_E_T_lit), lit_pc.reacts_exp)}
            r["pass"] = r["auc_barrier"] == 1
            r["barrier_beats_dG"] = r["auc_barrier"] > r["auc_dG"]
            r.update({"n_not_without_degradation": int((~kept.reacts_exp).sum()),
                      "auc_barrier_without_degradation": auc(-kept.barrier_ml, kept.reacts_exp),
                      "auc_dG_without_degradation": auc(-kept.dG_EnT_ml, kept.reacts_exp),
                      "posthoc_auc_barrier_without_acetone": auc(-no_acetone.barrier_ml, no_acetone.reacts_exp),
                      "posthoc_auc_dG_without_acetone": auc(-no_acetone.dG_EnT_ml, no_acetone.reacts_exp),
                      "excluded_donors": "; ".join(not_encoded)})
            screen_rows.append(r)

            # ------------------------------------------------ reference: 7a and donors against the paper
            lit_donors = [d for d in donors if d in E_T_pc_lit.index]
            dft_rows.append({
                "model": model, "estimate": estimate,
                "E_T_7a_ml": acc["E_T"], "E_T_7a_lit": acceptor_lit.E_T_lit, "E_T_7a_error": acc["E_T"] - acceptor_lit.E_T_lit,
                "lambda_T1_S0_7a_ml": acc["lambda_T1_S0"], "lambda_T1_S0_7a_lit": acceptor_lit.lambda_T1_S0_lit,
                "lambda_T1_S0_7a_error": acc["lambda_T1_S0"] - acceptor_lit.lambda_T1_S0_lit,
                "lambda_S0_T1_7a_ml": acc["lambda_S0_T1"],
                "E_T_7a_sd": acc["E_T_sd"], "lambda_S0_T1_7a_sd": acc["lambda_S0_T1_sd"], "lambda_T1_S0_7a_sd": acc["lambda_T1_S0_sd"],
                "E_T_7a_pseudovertical_lit": acceptor_lit.E_T_pseudovert_lit, "E_T_7a_vertical_lit": acceptor_lit.E_T_vert_lit,
                "pc_n_lit": len(lit_donors),
                "pc_E_T_mae": float(np.mean([abs(o[f"PC_{d}"]["E_T"] - E_T_pc_lit[d]) for d in lit_donors]))})

            # ------------------------------------------------ scope (non-evidentiary)
            s = pd.DataFrame([p for p in pairs if p["model"] == model and p["estimate"] == estimate and p["group"] == "scope"])
            for subset, t in (("as registered (7e, 7o out)", s[s.excluded == ""]), ("all with a yield", s)):
                t = t.dropna(subset=["yield_percent_exp"])
                rho_b, _, n = spearman(-t.barrier_ml, t.yield_percent_exp)
                rho_d, _, _ = spearman(-t.dG_EnT_ml, t.yield_percent_exp)
                scope_rows.append({"model": model, "estimate": estimate, "subset": subset, "n": n,
                                   "barrier_min": t.barrier_ml.min(), "barrier_max": t.barrier_ml.max(),
                                   "barrier_median": t.barrier_ml.median(), "barrier_sd": t.barrier_ml.std(),
                                   "rho_barrier_yield": rho_b, "rho_dG_yield": rho_d})

    write(pd.DataFrame(pairs), "bach_allene.csv")
    kq_table = pd.DataFrame(kq_rows)
    write(kq_table, "bach_kq_summary.csv")
    screen_table = pd.DataFrame(screen_rows)
    write(screen_table, "bach_screen_summary.csv")
    write(pd.DataFrame(dft_rows), "bach_dft_reference.csv")
    scope_table = pd.DataFrame(scope_rows)
    write(scope_table, "bach_scope_summary.csv")
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

    print(f"\nscreen: {screen_table.n_react.iloc[0]} react, {screen_table.n_not.iloc[0]} do not; "
          f"excluded {screen_table.excluded_donors.iloc[0]}")
    print(screen_table.drop(columns=["excluded_donors", "n_react", "n_not", "n_not_without_degradation", "auc_dG_lit"])
          .round(3).to_string(index=False))
    print(f"AUC(barrier) = 1 in {int(screen_table['pass'].sum())}/{n}; barrier > dG in "
          f"{int(screen_table.barrier_beats_dG.sum())}/{n} (beyond the E_T screen needs {ROBUST})")

    print("\nscope (non-evidentiary): Spearman rho with yield, range over runs")
    for subset, t in scope_table.groupby("subset", sort=False):
        print(f"  {subset:28s} n {t.n.iloc[0]}: rho(-barrier) {t.rho_barrier_yield.min():.2f} to {t.rho_barrier_yield.max():.2f}; "
              f"rho(-dG) {t.rho_dG_yield.min():.2f} to {t.rho_dG_yield.max():.2f}")


if __name__ == "__main__":
    main()
