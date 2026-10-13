# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Quenching against the barrier for short- and long-lived photocatalysts, the errors of the
criteria on the short-lived ones, and the gate that rejects each DvTE false positive.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv   (through metrics_extended.py)
Writes  results/lifetimes/barrier_bins.csv          exergonic pairs (DFT) per lifetime group
                                                    and bin of the asymmetric DFT barrier:
                                                    pairs, quenching pairs, fraction
        results/lifetimes/groups.csv                per lifetime group: photocatalysts,
                                                    pairs, median t_PC, false positives of
                                                    the criteria, pairs the barrier gate
                                                    rejects
        results/lifetimes/constant_threshold.csv    two-gate DFT at r = 7 A with t_PC(tau0)
                                                    and with one constant threshold
        results/lifetimes/dvte_false_positives.csv  per quencher class: DvTE false positives
                                                    and true positives by the gate of
                                                    two-gate DFT that rejects them

    uv run benchmark/further_analysis/lifetimes.py

Lifetime groups: tau0 < SHORT_NS (short) and tau0 >= SHORT_NS (long). Barrier bins of
BIN_WIDTH kcal/mol, the last one open (BIN_EDGES). The criteria are those of
metrics_extended.py (DFT quencher parameters; two-gate with r = r_PC + r_Q unless the
column says 7 A). Gate of two-gate DFT for a pair: energy (dG_EnT > 0), barrier
(exergonic but the asymmetric barrier exceeds t_PC) or pass.

constant_threshold.csv: the constant threshold is the median over the photocatalysts of
t_PC at r = 7 A (which depends on tau0 only), applied to every exergonic pair.

dvte_false_positives.csv: DvTE false positives that two-gate DFT does not predict, split
by the gate that rejects them; DvTE true positives that the energy gate rejects; two-gate
DFT false positives that DvTE does not predict; median and largest dG_EnT (DFT) of the
false positives rejected by the energy gate.

kcal/mol rounded to 2 decimals, MCC to 3.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import metrics_extended as mx  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "lifetimes"
SHORT_NS = 1000.0
BIN_EDGES = (0.0, 3.0, 6.0, 9.0, 12.0, np.inf)       # kcal/mol
GROUPS = ("short", "long")
KC = "kcal mol-1"
FP_COLUMNS = {"adiabatic_dft": ("adiabatic", "dft"), "dvte": ("dvte_2024", "dvte"),
              "two_gate_dft": ("twogate_lifetime", "dft")}
TWO_GATE_7A = ("twogate_lifetime_r7A", "dft")


def rounded(table):
    """kcal/mol columns to 2 decimals, other floats to 3, from full precision."""
    return table.round({c: 2 if KC in c else 3 for c in table.columns})


def gate(dG, predicted):
    """energy, barrier or pass for each pair of a two-gate criterion."""
    return np.where(dG > 0, "energy", np.where(~predicted, "barrier", "pass"))


def barrier_bins(y, short, exergonic, barrier):
    rows = []
    for name, member in (("short", short), ("long", ~short)):
        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            inside = member & exergonic & (barrier >= lo) & (barrier < hi)
            n, nq = int(inside.sum()), int(y[inside].sum())
            rows.append({"tau0_group": name, f"barrier_from / {KC}": lo, f"barrier_to / {KC}": hi,
                         "n_pairs": n, "n_quench": nq, "fraction_quench": nq / n if n else np.nan})
    return pd.DataFrame(rows)


def constant_threshold(y, pcs, dG, barrier, t_7, lifetime_prediction, short):
    """One row: the constant threshold, the MCC and the false positives on the short-lived
    photocatalysts with it and with t_PC(tau0)."""
    t = float(np.median(pd.Series(t_7).groupby(pcs).first()))
    constant = (dG <= 0) & (barrier <= t)
    row = {f"t_constant / {KC}": t,
           "MCC_constant": scoring.metrics(y, constant)["MCC"],
           "MCC_lifetime": scoring.metrics(y, lifetime_prediction)["MCC"],
           "FP_short_constant": scoring.metrics(y[short], constant[short])["FP"],
           "FP_short_lifetime": scoring.metrics(y[short], lifetime_prediction[short])["FP"]}
    return row


def group_table(y, pcs, short, t_pc, predicted, classes, gates):
    rows = []
    for name, member in (("short", short), ("long", ~short)):
        listed = sorted(pd.unique(pcs[member]), key=lambda p: int(p[2:]))
        rejected = member & (gates == "barrier")
        rows.append({"tau0_group": name, "n_photocatalysts": len(listed), "photocatalysts": " ".join(listed),
                     "n_pairs": int(member.sum()), "n_quench": int(y[member].sum()),
                     f"t_PC_median / {KC}": float(np.median(t_pc[member])),
                     **{f"FP_{k}": scoring.metrics(y[member], predicted[k][member])["FP"] for k in predicted},
                     "barrier_gate_rejected": int(rejected.sum()),
                     "barrier_gate_rejected_quench": int(y[rejected].sum()),
                     "barrier_gate_rejected_flexible": int((rejected & (classes == "flexible")).sum())})
    return pd.DataFrame(rows)


def dvte_table(y, classes, dG, dvte, two_gate, gates):
    rejected = dvte & ~y & ~two_gate
    rows = []
    for name in (*mx.CLASS_GROUPS, "all"):
        member = np.isin(classes, mx.CLASS_GROUPS[name]) if name != "all" else np.ones(len(y), bool)
        r, energy = rejected & member, rejected & member & (gates == "energy")
        rows.append({"quencher_class": name,
                     "dvte_FP_rejected": int(r.sum()),
                     "dvte_FP_rejected_energy_gate": int(energy.sum()),
                     "dvte_FP_rejected_barrier_gate": int((r & (gates == "barrier")).sum()),
                     "dvte_TP_rejected_energy_gate": int((dvte & y & ~two_gate & member & (gates == "energy")).sum()),
                     "two_gate_FP_not_dvte": int((two_gate & ~y & ~dvte & member).sum()),
                     f"dG_EnT_median_energy_gate / {KC}": float(np.median(dG[energy])) if energy.any() else np.nan,
                     f"dG_EnT_max_energy_gate / {KC}": float(np.max(dG[energy])) if energy.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    df, classes, crits = mx.load()
    y = df[pairs.QUENCHES].to_numpy(bool)
    pcs = df[pairs.PC_NAME].to_numpy()
    short = df[pairs.TAU0].to_numpy(float) < SHORT_NS
    by_name = {c.name: c for c in crits}
    predicted = {k: by_name[name].predict(y) for k, name in FP_COLUMNS.items()}

    dG = pairs.driving_force(df, "dft")
    barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(df, "dft")))
    tau = pairs.tau0_seconds(df)
    t_pc = theory.lifetime_barrier(tau, classification.K_SV_CRIT,
                                   theory.escape_over_frequency(pairs.contact_distances(df)))
    t_7 = theory.lifetime_barrier(tau, classification.K_SV_CRIT, theory.escape_over_frequency())
    gates = gate(dG, predicted["two_gate_dft"])

    row = constant_threshold(y, pcs, dG, barrier, t_7, by_name[TWO_GATE_7A].predict(y), short)
    groups = group_table(y, pcs, short, t_pc, predicted, classes, gates)
    tables = {"barrier_bins.csv": barrier_bins(y, short, dG <= 0, barrier),
              "groups.csv": groups,
              "constant_threshold.csv": pd.DataFrame([row]),
              "dvte_false_positives.csv": dvte_table(y, classes, dG, predicted["dvte"],
                                                     predicted["two_gate_dft"], gates)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        written = rounded(table)
        written.to_csv(RESULTS / name, index=False, lineterminator="\n")
        print(written.to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
