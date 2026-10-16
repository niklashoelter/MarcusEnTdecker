# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Sole-Daura 2024: sole_daura_ez.py with thioxanthone (TX) parameters from one source at a time.

    uv run benchmark/literature/sole_daura_tx.py

Substrates ours; the other photocatalysts keep the paper's DFT lambdas and experimental E_T.

`tx_source` (TX only)
    paper       experimental E_T, the paper's DFT lambdas (as in sole_daura_ez.py)
    paper_dft   the paper's DFT E_T (`pc_E_T_calc_lit`) and lambdas
    opc_dft     our DFT, organophotocatalyst library
    ml          our prediction, same model as the substrates, mixture mean

Reads    ../data/literature_benchmark/sole_daura_2024.csv
         ../data/computational_data_processed/organophotocatalyst_dft.csv
         results/literature/predictions_members_sole_daura_2024.csv
Writes   results/literature/sole_daura_tx.csv         one row per set, pair, model and estimate
         results/literature/sole_daura_tx_signs.csv   one row per set and E/Z group: ddG, its split, sign
         results/literature/sole_daura_tx_pss.csv     photostationary Z share with TX, per set
         results/literature/sole_daura_tx_parity.csv  absolute barriers against experiment, with nulls, per set
"""

import numpy as np
import pandas as pd

from common import (ESTIMATES, PROPERTIES, REPO, RT, barrier_asymmetric, barrier_symmetric,
                    literature, models, ours, pair_lambdas, split_difference, write)

OPC = REPO / "data" / "computational_data_processed" / "organophotocatalyst_dft.csv"
EXCLUDED_SUBSTRATE = "4"
DRAWS = 20000
SOURCES = ("paper", "paper_dft", "opc_dft", "ml")


def opc_tx():
    """Thioxanthone of the organophotocatalyst library, as written there."""
    opc = pd.read_csv(OPC)
    row = opc[opc.name == "thioxanthone"].iloc[0]
    return {prop: float(row[f"{prop} / kcal mol-1"]) for prop in PROPERTIES}


def pc_of(pair, pred, source, opc):
    """Photocatalyst properties: the paper's, except thioxanthone in the sets other than `paper`."""
    paper = {"E_T": pair.pc_E_T_lit, "lambda_S0_T1": pair.pc_lambda_S0_T1_lit,
             "lambda_T1_S0": pair.pc_lambda_T1_S0_lit}
    if pair.pc_label != "TX" or source == "paper":
        return paper
    if source == "paper_dft":
        return {**paper, "E_T": pair.pc_E_T_calc_lit}
    if source == "opc_dft":
        return dict(opc)
    o = ours(pred.loc["PC_TX"])
    return {prop: o[prop] for prop in PROPERTIES}


def terms(pc, substrate):
    forward, reverse = pair_lambdas(pc, substrate)
    return substrate["E_T"] - pc["E_T"], forward, reverse


def run(source, lit, preds, opc):
    """The four tables of sole_daura_ez.py for one thioxanthone parameter set."""
    rng = np.random.default_rng(20261006)   # seed of sole_daura_ez.py, per set
    rows = []
    for model, pred in preds.items():
        for estimate in ESTIMATES:
            for _, pair in lit.iterrows():
                sub = ours(pred.loc[pair.compound_id], estimate)
                pc = pc_of(pair, pred, source, opc)
                dG, forward, reverse = terms(pc, sub)
                rows.append({"tx_source": source, "model": model, "estimate": estimate,
                             "substrate": pair.substrate, "isomer": pair.isomer,
                             "compound_id": pair.compound_id, "pc_label": pair.pc_label,
                             "pc_E_T": pc["E_T"], "pc_lambda_S0_T1": pc["lambda_S0_T1"],
                             "pc_lambda_T1_S0": pc["lambda_T1_S0"],
                             "E_T_ml": sub["E_T"], "lambda_S0_T1_ml": sub["lambda_S0_T1"],
                             "lambda_T1_S0_ml": sub["lambda_T1_S0"],
                             "dG_EnT_ml": dG, "dG_EnT_lit": pair.dG_EnT_lit, "lambda_forward_ml": forward, "lambda_reverse_ml": reverse,
                             "barrier_ml": barrier_asymmetric(dG, forward, reverse),
                             "barrier_ml_symmetric": float(barrier_symmetric(dG, (forward + reverse) / 2)),
                             "barrier_asym_lit": pair.barrier_asym_lit, "barrier_exp": pair.barrier_exp,
                             "diffusion_controlled": pair.substrate == EXCLUDED_SUBSTRATE})
    table = pd.DataFrame(rows)

    signs = []
    for (model, estimate, substrate, pc_label), g in table.groupby(
            ["model", "estimate", "substrate", "pc_label"], sort=False):
        if set(g.isomer) != {"E", "Z"}:
            continue
        e, z = g[g.isomer == "E"].iloc[0], g[g.isomer == "Z"].iloc[0]
        full, dg_part, lam_part = split_difference(
            barrier_asymmetric, (e.dG_EnT_ml, e.lambda_forward_ml, e.lambda_reverse_ml),
            (z.dG_EnT_ml, z.lambda_forward_ml, z.lambda_reverse_ml))
        exp = e.barrier_exp - z.barrier_exp
        row = {"tx_source": source, "model": model, "estimate": estimate, "substrate": substrate,
               "pc_label": pc_label, "diffusion_controlled": bool(e.diffusion_controlled),
               "ddG_exp": exp, "ddG_lit": e.barrier_asym_lit - z.barrier_asym_lit,
               "ddG_ml": full, "ddG_ml_dG_part": dg_part, "ddG_ml_lambda_part": lam_part,
               "ddG_ml_symmetric": e.barrier_ml_symmetric - z.barrier_ml_symmetric,
               "ddG_EnT_ml": e.dG_EnT_ml - z.dG_EnT_ml, "ddG_EnT_lit": e.dG_EnT_lit - z.dG_EnT_lit}
        for name in ("ml", "ml_symmetric", "lit", "EnT_ml", "EnT_lit"):
            row[f"sign_correct_{name}"] = (None if pd.isna(exp) else bool(np.sign(row[f"ddG_{name}"]) == np.sign(exp)))
        signs.append(row)
    signs = pd.DataFrame(signs)

    # Probability of the correct sign under the deployed model's predicted uncertainty;
    # photocatalyst fixed at the set's values.
    pred = preds["deployed"]
    p_sign = {}
    for (substrate, pc_label), g in lit[lit.barrier_exp.notna()].groupby(["substrate", "pc_label"]):
        e, z = g[g.isomer == "E"].iloc[0], g[g.isomer == "Z"].iloc[0]
        pc = pc_of(e, pred, source, opc)
        draws = []
        for r in (e, z):
            o = ours(pred.loc[r.compound_id])
            draws.append({p: rng.normal(o[p], o[f"{p}_sd"], DRAWS) for p in PROPERTIES})
        dd = np.array([barrier_asymmetric(*terms(pc, {p: draws[0][p][i] for p in PROPERTIES}))
                       - barrier_asymmetric(*terms(pc, {p: draws[1][p][i] for p in PROPERTIES}))
                       for i in range(DRAWS)])
        p_sign[(substrate, pc_label)] = float((np.sign(dd) == np.sign(e.barrier_exp - z.barrier_exp)).mean())
    signs["p_sign_correct_deployed_uncertainty"] = [
        p_sign.get((r.substrate, r.pc_label)) if (r.model == "deployed" and r.estimate == "mean") else np.nan
        for r in signs.itertuples()]

    pss = []
    for r in signs[signs.pc_label == "TX"].itertuples():
        exp_row = lit[(lit.substrate == r.substrate) & (lit.pc_label == "TX")].iloc[0]
        ratio = np.exp(-r.ddG_ml / RT)
        pss.append({"tx_source": source, "model": r.model, "estimate": r.estimate, "substrate": r.substrate,
                    "diffusion_controlled": r.diffusion_controlled,
                    "z_percent_ml": 100 * ratio / (1 + ratio),
                    "z_percent_exp": exp_row.pss_z_percent_exp, "z_percent_calc_lit": exp_row.pss_z_percent_calc_lit})
    pss = pd.DataFrame(pss)

    # Nulls use the set's thioxanthone.
    parity = []
    scored = table[~table.diffusion_controlled & table.barrier_exp.notna()]
    pairs = lit[lit.barrier_exp.notna() & (lit.substrate != EXCLUDED_SUBSTRATE)]
    for (model, estimate), t in scored.groupby(["model", "estimate"], sort=False):
        mean_sub = t[["E_T_ml", "lambda_S0_T1_ml", "lambda_T1_S0_ml"]].mean()
        nulls = {"constant substrate (E_T 50, lambda 50/30)": {"E_T": 50.0, "lambda_S0_T1": 50.0, "lambda_T1_S0": 30.0},
                 "mean of our substrates": {"E_T": mean_sub.E_T_ml, "lambda_S0_T1": mean_sub.lambda_S0_T1_ml,
                                            "lambda_T1_S0": mean_sub.lambda_T1_S0_ml}}
        candidates = {"ours, asymmetric": t.barrier_ml.to_numpy(), "ours, symmetric": t.barrier_ml_symmetric.to_numpy(),
                      "paper DFT, asymmetric": t.barrier_asym_lit.to_numpy()}
        for name, sub in nulls.items():
            candidates[name] = np.array([barrier_asymmetric(*terms(pc_of(p, preds[model], source, opc), sub))
                                         for _, p in pairs.iterrows()])
        tx = (t.pc_label == "TX").to_numpy()
        for name, b in candidates.items():
            for subset, sel in (("all 16 pairs", np.ones(len(t), bool)), ("TX pairs only (substrate varies)", tx)):
                y = t.barrier_exp.to_numpy()[sel]
                parity.append({"tx_source": source, "model": model, "estimate": estimate, "barriers": name,
                               "subset": subset, "n": int(sel.sum()),
                               "pearson_r": float(np.corrcoef(b[sel], y)[0, 1]) if np.ptp(b[sel]) > 1e-6 else np.nan,
                               "mae_barrier": float(np.abs(b[sel] - y).mean()),
                               "mean_signed_error": float((b[sel] - y).mean())})
    return table, signs, pss, pd.DataFrame(parity)


def main():
    lit = literature("sole_daura_2024")
    lit["substrate"] = lit.compound_id.str[:-1]
    preds = models("sole_daura_2024")
    opc = opc_tx()

    tables = [run(source, lit, preds, opc) for source in SOURCES]
    table, signs, pss, parity = (pd.concat(t, ignore_index=True) for t in zip(*tables))
    write(table, "sole_daura_tx.csv")
    write(signs, "sole_daura_tx_signs.csv")
    write(pss, "sole_daura_tx_pss.csv")
    write(parity, "sole_daura_tx_parity.csv")

    # deployed model, mixture mean: TX parameters, parity, ddG and Z share per substrate
    deployed = lambda t: t[(t.model == "deployed") & (t.estimate == "mean")]  # noqa: E731
    tx = deployed(table)[lambda t: t.pc_label == "TX"].groupby("tx_source", sort=False).first()
    fit = deployed(parity)[lambda t: (t.barriers == "ours, asymmetric") & (t.subset == "all 16 pairs")]
    s = deployed(signs)[lambda t: (t.pc_label == "TX") & ~t.diffusion_controlled]
    z = deployed(pss)[lambda t: ~t.diffusion_controlled]
    summary = pd.DataFrame({
        "E_T": tx.pc_E_T, "lambda_S0_T1": tx.pc_lambda_S0_T1, "lambda_T1_S0": tx.pc_lambda_T1_S0,
        "R2": fit.set_index("tx_source").pearson_r ** 2, "MAE": fit.set_index("tx_source").mae_barrier,
        **{f"ddG {k}": s[s.substrate == k].set_index("tx_source").ddG_ml for k in ("1", "2", "3")},
        **{f"Z % {k}": z[z.substrate == k].set_index("tx_source").z_percent_ml for k in ("1", "2", "3")}})
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
