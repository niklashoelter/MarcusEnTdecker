# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "rdkit==2025.9.4"]
# ///
"""
CPU time per quencher for the inputs of each criterion; photocatalyst calculations are
not counted.

Reads   ../data/computational_data_processed/quencher_timings.csv
        ../data/computational_data_processed/quencher_timings_ml.csv
        ../data/compound_identification.xlsx               (SMILES, for atom counts)
Writes  results/cost/per_quencher.csv   CPU hours of each group per quencher, per molecule
                                        and divided by the quencher's number of atoms
        results/cost/statistics.csv     min, max, mean, standard deviation and median over
                                        the quenchers, per molecule and per atom
        results/cost/dataset.csv        totals over all quenchers of the dataset

    uv run benchmark/cost.py

Group            calculations                                        criteria
adiabatic DFT    S0 and T1 opt+freq, S0 and T1 single points         adiabatic (dft)
four-point DFT   the above and both vertical spin-change points      two_gate, marcus_barrier (dft)
DvTE             Gaussian opt+freq, MD trajectories and the          dvte
                 vertical-gap single points (failed or timed-out
                 single points not counted)
ML, per call     inference of one quencher per call                  all ml criteria
ML, batch        one call for all quenchers, split equally per quencher

CPU hours: wall time x cores (DFT, DvTE); process CPU time, mean over repeats (ML).
Atom counts include hydrogens. A group is blank for a quencher missing one of its
calculations and left out of that group's statistics and totals. Sample standard deviations.
"""

import sys

import numpy as np
import pandas as pd

import pairs

DATA = pairs.HERE.parent / "data"
TIMINGS = DATA / "computational_data_processed" / "quencher_timings.csv"
TIMINGS_ML = DATA / "computational_data_processed" / "quencher_timings_ml.csv"
RESULTS = pairs.RESULTS / "cost"

ADIABATIC = ("dft S0 opt_freq", "dft T1 opt_freq", "dft S0 sp", "dft T1 sp")
CALCULATIONS = {
    "adiabatic DFT": ADIABATIC,
    "four-point DFT": (*ADIABATIC, "dft S0 sp_vert_spin_change", "dft T1 sp_vert_spin_change"),
    "DvTE": ("dvte opt_freq", "dvte md", "dvte sp singlet", "dvte sp triplet"),
}
NORMALIZATIONS = {"per molecule": "cpu / h", "per atom": "cpu_per_atom / h"}


def per_quencher(timings):
    """CPU hours of every group in CALCULATIONS, one row per quencher."""
    hours = timings.groupby(["quencher_name", "calculation"])["cpu / h"].sum().unstack()
    table = pd.DataFrame(index=hours.index)
    for group, calculations in CALCULATIONS.items():
        present = hours.reindex(columns=list(calculations))
        table[group] = present.sum(axis=1, min_count=len(calculations)).where(present.notna().all(axis=1))
    return table.reset_index()


def ml_hours(timings_ml, quenchers):
    """CPU hours of one quencher's inference, per call and as an equal share of the batch call."""
    individual = timings_ml[timings_ml["mode"] == "individual"].set_index("quencher_name")
    batch = timings_ml[timings_ml["mode"] == "batch"].iloc[0]
    return pd.DataFrame({
        "quencher_name": list(quenchers),
        "ML, per call": individual["cpu_mean / s"].reindex(quenchers).to_numpy() / 3600.0,
        "ML, batch": np.full(len(quenchers), batch["cpu_mean / s"] / batch["n_molecules"] / 3600.0),
    })


def normalized(wide, atoms):
    """Long table: one row per quencher and group, per molecule and per atom."""
    long = wide.melt(id_vars="quencher_name", var_name="group", value_name="cpu / h")
    long.insert(1, "n_atoms", long["quencher_name"].map(atoms))
    long["cpu_per_atom / h"] = long["cpu / h"] / long["n_atoms"]
    return long


def statistics(long):
    rows = []
    for group, part in long.groupby("group", sort=False):
        for normalization, column in NORMALIZATIONS.items():
            values = part[column].dropna()
            rows.append({"group": group, "normalization": normalization, "quenchers": len(values),
                         "min / h": values.min(), "max / h": values.max(),
                         "mean / h": values.mean(), "std / h": values.std(ddof=1),
                         "median / h": values.median()})
    return pd.DataFrame(rows)


def dataset_totals(long):
    rows = []
    for group, part in long.groupby("group", sort=False):
        present = part.dropna(subset=["cpu / h"])
        total, atoms = present["cpu / h"].sum(), int(present["n_atoms"].sum())
        rows.append({"group": group, "quenchers": len(present), "atoms": atoms,
                     "total / h": total, "total_per_atom / h": total / atoms if atoms else np.nan})
    return pd.DataFrame(rows)


def significant(table, digits=4):
    """Floats to `digits` significant figures; |x| < 1e-12 becomes 0."""
    def one(x):
        if pd.isna(x):
            return x
        return 0.0 if abs(x) < 1e-12 else float(f"{x:.{digits}g}")

    out = table.copy()
    for column in out.select_dtypes("float").columns:
        out[column] = out[column].map(one)
    return out


def atom_counts():
    """{quencher name: number of atoms including hydrogens}, from the SMILES."""
    from rdkit import Chem

    sys.path.insert(0, str(DATA))
    from compound_identification import QUENCHERS

    return {f"Q{q}": Chem.AddHs(Chem.MolFromSmiles(r["smiles"])).GetNumAtoms()
            for q, r in QUENCHERS.items()}


def main():
    for path in (TIMINGS, TIMINGS_ML):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the timing scripts in data/ first")
    wide = per_quencher(pd.read_csv(TIMINGS))
    wide = wide.merge(ml_hours(pd.read_csv(TIMINGS_ML, comment="#"), wide["quencher_name"]),
                      on="quencher_name")
    wide = wide.sort_values("quencher_name", key=lambda s: s.str[1:].astype(int))
    long = normalized(wide, atom_counts())
    stats, totals = statistics(long), dataset_totals(long)

    RESULTS.mkdir(parents=True, exist_ok=True)
    significant(long).to_csv(RESULTS / "per_quencher.csv", index=False, lineterminator="\n")
    significant(stats).to_csv(RESULTS / "statistics.csv", index=False, lineterminator="\n")
    significant(totals).to_csv(RESULTS / "dataset.csv", index=False, lineterminator="\n")

    number = lambda x: f"{x:.3g}"  # noqa: E731
    print(stats.to_string(index=False, float_format=number))
    print()
    print(totals.to_string(index=False, float_format=number))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
