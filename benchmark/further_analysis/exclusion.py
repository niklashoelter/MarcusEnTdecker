# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
The quenchers Q38 and Q31, which the benchmark leaves out.

Reads   ../../data/exp_dataset.csv, ../../data/exp_dataset_ent.csv (also through pairs.py)
        ../../data/experimental_data_processed/stern_volmer_intensities.csv
Writes  results/exclusion/evidence.csv      Q38 and Q31: pairs, outcomes, driving forces,
                                            Stern-Volmer constants, emission ratios and the
                                            predictions of the criteria on their scored pairs

    uv run benchmark/further_analysis/exclusion.py

Q38 and Q31 (pairs.LEFT_OUT_QUENCHERS) are kept on loading and scored (PC22 stays left out).
Criteria: adiabatic (DFT, ML), DvTE (p = 0.008, normal), two-gate (DFT, r from DFT sizes;
ML, r = 7 A).

Energy-transfer data set: all pairs except quenching pairs where electron transfer is not
ruled out. I0/I = mean intensity without quencher / mean intensity at the highest quencher
concentration. Expected I0/I for k_q = K_Q_REFERENCE: 1 + K_Q_REFERENCE tau0 [Q]max.
Two-gate rejections: energy gate (dG_EnT > 0) or barrier gate (exergonic, barrier > t_PC).
dG_EnT = E_T(Q) - E_T(PC).
"""

import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402

DATA = HERE.parent.parent / "data"
SCREEN = DATA / "exp_dataset.csv"
INTENSITIES = DATA / "experimental_data_processed" / "stern_volmer_intensities.csv"
RESULTS = HERE / "results" / "exclusion"

LEFT_OUT = ("Q38", "Q31")
K_Q_REFERENCE = 1e9                      # M-1 s-1
CRITERIA = {
    "adiabatic, dft": (classification.candidates("adiabatic", "")[0], "dft"),
    "adiabatic, ml": (classification.candidates("adiabatic", "")[0], "ml"),
    "dvte": (classification.candidates("dvte", "p = 0.008, normal")[0], "dft"),
    "two_gate, dft": (classification.two_gate_setting("dft"), "dft"),
    "two_gate, ml": (classification.two_gate_setting("ml"), "ml"),
}


@contextmanager
def left_out_quenchers_included():
    """pairs.load() with Q38 and Q31 kept."""
    saved = pairs.LEFT_OUT_QUENCHERS
    pairs.LEFT_OUT_QUENCHERS = {}
    try:
        yield
    finally:
        pairs.LEFT_OUT_QUENCHERS = saved


def scored_with_left_out():
    with left_out_quenchers_included():
        return pairs.scored(pairs.load())


def emission_ratios(intensities):
    """I0/I at the highest quencher concentration of every pair, and that concentration."""
    rows = []
    for (pc, q), t in intensities.groupby(["pc_new_id", "quencher_id"]):
        c_max = t.quencher_conc_M.max()
        rows.append({"pc_new_id": pc, "quencher_id": q, "conc_max / mM": c_max * 1e3,
                     "I0_over_I_at_max": t.loc[t.quencher_conc_M == 0, "intensity_au"].mean()
                     / t.loc[t.quencher_conc_M == c_max, "intensity_au"].mean()})
    return pd.DataFrame(rows)


def predictions(df):
    """Prediction of every criterion, the two-gate barriers and t_PC for the scored pairs."""
    out = pd.DataFrame({pairs.PC_NAME: df[pairs.PC_NAME], pairs.Q_NAME: df[pairs.Q_NAME]})
    for name, (setting, source) in CRITERIA.items():
        out[f"pred {name}"] = classification.predict(df, setting, source)[0]
    two_gate = CRITERIA["two_gate, dft"][0]
    for source in pairs.SOURCES:
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        out[f"barrier_{source} / kcal mol-1"] = theory.marcus_barrier_asymmetric(
            pairs.driving_force(df, source), forward, reverse)
    out["t_PC / kcal mol-1"] = theory.lifetime_barrier(
        pairs.tau0_seconds(df), two_gate.k_sv_crit, theory.escape_over_frequency(pairs.contact_distances(df)))
    return out


def pair_table(screen, ent, ratios, df):
    """Every pair of the left-out quenchers with its status in the benchmark."""
    keys = [pairs.PC_NAME, pairs.Q_NAME]
    t = screen[screen[pairs.Q_NAME].isin(LEFT_OUT)].copy()
    t = t.merge(ratios, on=["pc_new_id", "quencher_id"], how="left", validate="1:1")
    pc_et = ent.groupby(pairs.PC_NAME)[pairs.PC_E_T].first()
    q_et = {s: ent.groupby(pairs.Q_NAME)[f"q_E_T_{s} / kcal mol-1"].first() for s in pairs.SOURCES}
    t[pairs.PC_E_T] = t[pairs.PC_NAME].map(pc_et)
    for s in pairs.SOURCES:
        t[pairs.dg_ent(s)] = t[pairs.Q_NAME].map(q_et[s]) - t[pairs.PC_E_T]
    in_ent = t.set_index(keys).index.isin(ent.set_index(keys).index)
    t = t.merge(predictions(df), on=keys, how="left", validate="1:1")
    scored = t[f"pred {next(iter(CRITERIA))}"].notna().to_numpy()
    t["status"] = np.select([~in_ent, t[pairs.PC_NAME].isin(list(pairs.LEFT_OUT)).to_numpy(), scored],
                            ["not in the energy-transfer data set (SET not ruled out)",
                             "photocatalyst left out", "scored"], "missing input")
    t["I0_over_I expected, k_q = 1e9"] = 1 + K_Q_REFERENCE * t[pairs.TAU0] * 1e-9 * t["conc_max / mM"] * 1e-3
    for c in [c for c in t.columns if c.startswith("pred ")]:
        t[c] = t[c].astype("boolean")
    return t.sort_values(["quencher_id", pairs.PC_E_T], ascending=[True, False]).reset_index(drop=True)


def evidence(table):
    rows = []
    for q in LEFT_OUT:
        t = table[table[pairs.Q_NAME] == q]
        s = t[t.status == "scored"]
        quench = s[s[pairs.QUENCHES] == 1]
        dg = {src: s[pairs.dg_ent(src)] for src in pairs.SOURCES}
        downhill = s[dg["dft"] <= -5]
        set_pcs = t.loc[t.status.str.startswith("not in"), pairs.PC_NAME]
        row = {"quencher": q, "pairs": len(t), "quenching_pairs": int(t[pairs.QUENCHES].sum()),
               "quenching_pairs_not_in_ent_dataset": len(set_pcs),
               "photocatalysts_not_in_ent_dataset": " ".join(set_pcs.sort_values(key=lambda c: c.str[2:].astype(int))),
               "scored_pairs": len(s), "scored_quenching_pairs": int(s[pairs.QUENCHES].sum())}
        for src in pairs.SOURCES:
            row[f"dG_EnT_{src}_min / kcal mol-1"] = dg[src].min()
            row[f"dG_EnT_{src}_max / kcal mol-1"] = dg[src].max()
        row |= {"scored_pairs_dG_dft_le_0": int((dg["dft"] <= 0).sum()),
                "scored_pairs_dG_dft_le_-10": int((dg["dft"] <= -10).sum()),
                "K_SV_max_scored / M-1": s[pairs.K_SV].max(),
                "K_SV_quenching_min / M-1": quench[pairs.K_SV].min(),
                "K_SV_quenching_max / M-1": quench[pairs.K_SV].max(),
                "k_q_quenching_min / M-1 s-1": quench[pairs.K_Q].min(),
                "k_q_quenching_max / M-1 s-1": quench[pairs.K_Q].max(),
                "I0_over_I_all_min": t.I0_over_I_at_max.min(), "I0_over_I_all_max": t.I0_over_I_at_max.max(),
                "I0_over_I_scored_min": s.I0_over_I_at_max.min(), "I0_over_I_scored_max": s.I0_over_I_at_max.max(),
                "I0_over_I_expected_1e9_dG_le_-5_min": downhill["I0_over_I expected, k_q = 1e9"].min(),
                "I0_over_I_expected_1e9_dG_le_-5_max": downhill["I0_over_I expected, k_q = 1e9"].max()}
        for name in CRITERIA:
            row[f"predicted_quench {name}"] = int(s[f"pred {name}"].sum())
        for src in pairs.SOURCES:
            passed = s[f"pred two_gate, {src}"].astype(bool)
            row[f"two_gate_{src}_rejected_energy_gate"] = int((~passed & (dg[src] > 0)).sum())
            row[f"two_gate_{src}_rejected_barrier_gate"] = int((~passed & (dg[src] <= 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def rounded(table):
    """kcal/mol to 2 decimals, rate constants to 3 significant figures, the rest to 3 decimals."""
    table = table.copy()
    for c in table.select_dtypes(float).columns:
        if "M-1 s-1" in c:
            table[c] = table[c].map(lambda v: float(f"{v:.3g}") if pd.notna(v) else v)
        else:
            table[c] = table[c].round(2 if "kcal" in c else 3)
    return table


def main():
    screen = pd.read_csv(SCREEN)
    ent = pd.read_csv(pairs.DATASET)
    df = scored_with_left_out()
    table = pair_table(screen, ent, emission_ratios(pd.read_csv(INTENSITIES)), df)
    t = rounded(evidence(table))
    RESULTS.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS / "evidence.csv", index=False, lineterminator="\n")
    print(t.set_index("quencher").T.to_string(), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
