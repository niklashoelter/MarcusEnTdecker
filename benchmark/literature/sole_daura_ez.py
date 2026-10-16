# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Sole-Daura 2024 (experiments: Zaehringer, Wienhold, Gilmour, Kerzig 2023): E/Z
sensitisation barriers.

    uv run benchmark/literature/sole_daura_ez.py

ddG = dG*(E) - dG*(Z) (negative: Z accumulates), its sign against experiment per substrate
and its Shapley split into driving-force and reorganisation parts; also
ddG_EnT = dG_EnT(E) - dG_EnT(Z), ours and the paper's, and the symmetric form.
Asymmetric form (the paper's eq. 4), unscaled. Substrates ours; photocatalysts the paper's
with the experimental E_T (`pc_source` = "paper"). Ru catalyst and substrate 4 excluded
(see SI; substrate 4 flagged in the pair table).
Photostationary state with TX: [Z]/[E] = k_E/k_Z = exp(-ddG/RT).

Writes   results/literature/sole_daura_ez.csv        one row per pair, model and estimate
         results/literature/sole_daura_ez_signs.csv  one row per E/Z group: ddG, its split, sign
         results/literature/sole_daura_pss.csv       photostationary Z share with TX
         results/literature/sole_daura_parity.csv    absolute barriers against experiment, with nulls
"""

import numpy as np
import pandas as pd

from common import (ESTIMATES, PROPERTIES, RT, barrier_asymmetric, barrier_symmetric, literature,
                    models, ours, pair_lambdas, split_difference, write)

EXCLUDED_SUBSTRATE = "4"
LEFT_OUT_PC = "Ru"
DRAWS = 20000
RNG = np.random.default_rng(20261006)


def pc_of(pair):
    """The paper's photocatalyst properties."""
    return {"E_T": pair.pc_E_T_lit, "lambda_S0_T1": pair.pc_lambda_S0_T1_lit,
            "lambda_T1_S0": pair.pc_lambda_T1_S0_lit}


def terms(pc, substrate):
    forward, reverse = pair_lambdas(pc, substrate)
    return substrate["E_T"] - pc["E_T"], forward, reverse


def main():
    lit = literature("sole_daura_2024")
    lit = lit[lit.pc_label != LEFT_OUT_PC].reset_index(drop=True)
    lit["substrate"] = lit.compound_id.str[:-1]
    preds = models("sole_daura_2024")

    rows = []
    for model, pred in preds.items():
        for estimate in ESTIMATES:
            for _, pair in lit.iterrows():
                sub = ours(pred.loc[pair.compound_id], estimate)
                dG, forward, reverse = terms(pc_of(pair), sub)
                rows.append({"model": model, "estimate": estimate, "pc_source": "paper",
                             "substrate": pair.substrate, "isomer": pair.isomer,
                             "compound_id": pair.compound_id, "pc_label": pair.pc_label,
                             "E_T_ml": sub["E_T"], "lambda_S0_T1_ml": sub["lambda_S0_T1"],
                             "lambda_T1_S0_ml": sub["lambda_T1_S0"],
                             "dG_EnT_ml": dG, "dG_EnT_lit": pair.dG_EnT_lit, "lambda_forward_ml": forward, "lambda_reverse_ml": reverse,
                             "barrier_ml": barrier_asymmetric(dG, forward, reverse),
                             "barrier_ml_symmetric": float(barrier_symmetric(dG, (forward + reverse) / 2)),
                             "barrier_asym_lit": pair.barrier_asym_lit, "barrier_exp": pair.barrier_exp,
                             "diffusion_controlled": pair.substrate == EXCLUDED_SUBSTRATE})
    table = pd.DataFrame(rows)
    write(table, "sole_daura_ez.csv")

    signs = []
    for (model, estimate, source, substrate, pc_label), g in table.groupby(
            ["model", "estimate", "pc_source", "substrate", "pc_label"], sort=False):
        if set(g.isomer) != {"E", "Z"}:
            continue
        e, z = g[g.isomer == "E"].iloc[0], g[g.isomer == "Z"].iloc[0]
        full, dg_part, lam_part = split_difference(
            barrier_asymmetric, (e.dG_EnT_ml, e.lambda_forward_ml, e.lambda_reverse_ml),
            (z.dG_EnT_ml, z.lambda_forward_ml, z.lambda_reverse_ml))
        exp = e.barrier_exp - z.barrier_exp
        row = {"model": model, "estimate": estimate, "pc_source": source, "substrate": substrate,
               "pc_label": pc_label, "diffusion_controlled": bool(e.diffusion_controlled),
               "ddG_exp": exp, "ddG_lit": e.barrier_asym_lit - z.barrier_asym_lit,
               "ddG_ml": full, "ddG_ml_dG_part": dg_part, "ddG_ml_lambda_part": lam_part,
               "ddG_ml_symmetric": e.barrier_ml_symmetric - z.barrier_ml_symmetric,
               # triplet-energy-only difference
               "ddG_EnT_ml": e.dG_EnT_ml - z.dG_EnT_ml, "ddG_EnT_lit": e.dG_EnT_lit - z.dG_EnT_lit}
        for name in ("ml", "ml_symmetric", "lit", "EnT_ml", "EnT_lit"):
            row[f"sign_correct_{name}"] = (None if pd.isna(exp) else bool(np.sign(row[f"ddG_{name}"]) == np.sign(exp)))
        signs.append(row)
    signs = pd.DataFrame(signs)

    # Probability of the correct sign: every property of both isomers drawn independently
    # from the deployed model's predicted spread.
    pred = preds["deployed"]
    p_sign = {}
    for (substrate, pc_label), g in lit[lit.barrier_exp.notna()].groupby(["substrate", "pc_label"]):
        e, z = g[g.isomer == "E"].iloc[0], g[g.isomer == "Z"].iloc[0]
        pc = pc_of(e)
        draws = []
        for r in (e, z):
            o = ours(pred.loc[r.compound_id])
            draws.append({p: RNG.normal(o[p], o[f"{p}_sd"], DRAWS) for p in PROPERTIES})
        dd = np.array([barrier_asymmetric(*terms(pc, {p: draws[0][p][i] for p in PROPERTIES}))
                       - barrier_asymmetric(*terms(pc, {p: draws[1][p][i] for p in PROPERTIES}))
                       for i in range(DRAWS)])
        p_sign[(substrate, pc_label)] = float((np.sign(dd) == np.sign(e.barrier_exp - z.barrier_exp)).mean())
    signs["p_sign_correct_deployed_uncertainty"] = [
        p_sign.get((r.substrate, r.pc_label)) if (r.model == "deployed" and r.estimate == "mean" and r.pc_source == "paper")
        else np.nan for r in signs.itertuples()]
    write(signs, "sole_daura_ez_signs.csv")

    pss = []
    for r in signs[signs.pc_label == "TX"].itertuples():
        exp_row = lit[(lit.substrate == r.substrate) & (lit.pc_label == "TX")].iloc[0]
        ratio = np.exp(-r.ddG_ml / RT)
        pss.append({"model": r.model, "estimate": r.estimate, "pc_source": r.pc_source, "substrate": r.substrate,
                    "diffusion_controlled": r.diffusion_controlled,
                    "z_percent_ml": 100 * ratio / (1 + ratio),
                    "z_percent_exp": exp_row.pss_z_percent_exp, "z_percent_calc_lit": exp_row.pss_z_percent_calc_lit})
    write(pd.DataFrame(pss), "sole_daura_pss.csv")

    # Absolute barriers; null: the same catalysts with one constant substrate (E_T 50,
    # lambdas 50/30).
    parity = []
    scored = table[~table.diffusion_controlled & table.barrier_exp.notna()]
    null = {"E_T": 50.0, "lambda_S0_T1": 50.0, "lambda_T1_S0": 30.0}
    for (model, estimate), t in scored.groupby(["model", "estimate"], sort=False):
        pairs = lit[lit.barrier_exp.notna() & (lit.substrate != EXCLUDED_SUBSTRATE)]
        candidates = {"ours, asymmetric": t.barrier_ml.to_numpy(),
                      "paper DFT, asymmetric": t.barrier_asym_lit.to_numpy(),
                      "constant substrate (E_T 50, lambda 50/30)":
                          np.array([barrier_asymmetric(*terms(pc_of(p), null)) for _, p in pairs.iterrows()])}
        tx = (t.pc_label == "TX").to_numpy()
        for name, b in candidates.items():
            for subset, sel in (("all 16 pairs", np.ones(len(t), bool)), ("TX pairs only (substrate varies)", tx)):
                y = t.barrier_exp.to_numpy()[sel]
                parity.append({"model": model, "estimate": estimate, "barriers": name, "subset": subset, "n": int(sel.sum()),
                               # NaN for a constant null
                               "pearson_r": float(np.corrcoef(b[sel], y)[0, 1]) if np.ptp(b[sel]) > 1e-6 else np.nan,
                               "mae_barrier": float(np.abs(b[sel] - y).mean()),
                               "mean_signed_error": float((b[sel] - y).mean())})
    write(pd.DataFrame(parity), "sole_daura_parity.csv")

    s = signs[~signs.diffusion_controlled & signs.ddG_exp.notna()]
    tally = (s.groupby(["substrate", "estimate"]).agg(models_correct=("sign_correct_ml", "sum"), groups=("sign_correct_ml", "size")))
    print(tally.to_string())


if __name__ == "__main__":
    main()
