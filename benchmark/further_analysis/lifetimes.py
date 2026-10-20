# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
The errors of the criteria on short- and long-lived photocatalysts.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv   (through metrics_extended.py)
Writes  results/lifetimes/groups.csv                per lifetime group: photocatalysts,
                                                    pairs, median t_PC, false positives of
                                                    the criteria, pairs the barrier gate
                                                    rejects

    uv run benchmark/further_analysis/lifetimes.py

Lifetime groups: short tau0 < SHORT_NS, long tau0 >= SHORT_NS. Criteria of
metrics_extended.py (DFT quencher parameters; two-gate with r = r_PC + r_Q). Two-gate DFT
gate per pair: energy (dG_EnT > 0), barrier (exergonic, barrier > t_PC) or pass.

kcal/mol: 2 decimals; fractions: 3.
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
GROUPS = ("short", "long")
KC = "kcal mol-1"
FP_COLUMNS = {"adiabatic_dft": ("adiabatic", "dft"), "dvte": ("dvte_2024", "dvte"),
              "two_gate_dft": ("twogate_lifetime", "dft")}


def rounded(table):
    """kcal/mol columns to 2 decimals, other floats to 3, from full precision."""
    return table.round({c: 2 if KC in c else 3 for c in table.columns})


def gate(dG, predicted):
    """energy, barrier or pass for each pair of a two-gate criterion."""
    return np.where(dG > 0, "energy", np.where(~predicted, "barrier", "pass"))


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


def main():
    df, classes, crits = mx.load()
    y = df[pairs.QUENCHES].to_numpy(bool)
    pcs = df[pairs.PC_NAME].to_numpy()
    short = df[pairs.TAU0].to_numpy(float) < SHORT_NS
    by_name = {c.name: c for c in crits}
    predicted = {k: by_name[name].prediction for k, name in FP_COLUMNS.items()}

    dG = pairs.driving_force(df, "dft")
    tau = pairs.tau0_seconds(df)
    t_pc = theory.lifetime_barrier(tau, classification.K_SV_CRIT,
                                   theory.escape_over_frequency(pairs.contact_distances(df)))
    gates = gate(dG, predicted["two_gate_dft"])

    tables = {"groups.csv": group_table(y, pcs, short, t_pc, predicted, classes, gates)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        written = rounded(table)
        written.to_csv(RESULTS / name, index=False, lineterminator="\n")
        print(written.to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
