# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
The contact distance r in the lifetime threshold of the two_gate criterion.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        ../../data/computational_data_processed/molecular_extent.csv
Writes  results/contact_distance/pair_distances.csv      r_PC, r_Q and r_PC + r_Q of every
                                                         scored pair, from the van der Waals
                                                         volume and from the molecular extent
        results/contact_distance/r_ij_distribution.csv   their distribution over the scored
                                                         pairs and over the molecules, and
                                                         the shift of t_PC against r = 7 A
        results/contact_distance/dt_pc_per_angstrom.csv  K_A, k_-d/nu, the shift and slope
                                                         of t_PC and its range over the
                                                         photocatalysts, for r = 4-12 A
        results/contact_distance/metrics_r.csv           two_gate metrics with r = 7 A, the
                                                         per-pair r of either radius, and a
                                                         fixed r from 4 to 12 A

    uv run benchmark/further_analysis/contact_distance.py

The lifetime threshold is t_PC = RT ln((k_d tau0 / K_SV,crit - 1) / (k_-d/nu)) with
k_-d/nu = (k_d / K_A) / (k_B T / h) and K_A = 4 pi N_A r^3 / 3 (theory.lifetime_barrier),
so t_PC = const(tau0) + 3 RT ln r and dt_PC/dr = 3RT/r.

Two radii per molecule, both from the S0 DFT geometry with Bondi van der Waals radii:
  volume   the radius of the sphere with the molecule's van der Waals volume, as in the
           dataset (pairs.PC_RADIUS, pairs.Q_RADIUS); r_PC + r_Q is pairs.CONTACT
  extent   half the largest atom-atom extent including the van der Waals radii,
           max_ij (d_ij + R_i + R_j) / 2, from molecular_extent.csv

two_gate: dG_EnT <= 0 and asymmetric Marcus barrier <= t_PC(tau0, r), K_SV,crit = 10 M-1,
as in classification.py, with the continuous score t_PC - barrier for AUC. Quencher
parameters from DFT or ML; photocatalyst parameters are DFT.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "contact_distance"
EXTENT = pairs.HERE.parent / "data" / "computational_data_processed" / "molecular_extent.csv"

R_REFERENCE = theory.CONTACT_DISTANCE                       # A
R_GRID = np.round(np.arange(4.0, 12.0001, 0.5), 2)          # A
RADII = {"volume": "vdW volume spheres", "extent": "extent (half max distance + vdW)"}


def extent_radii(df):
    """(r_PC, r_Q) from the molecular extent, as arrays over the rows of df."""
    table = pd.read_csv(EXTENT)
    by_kind = {kind: part.set_index("id")["r_extent / A"] for kind, part in table.groupby("kind")}
    return (df["pc_old_id"].map(by_kind["photocatalyst"]).to_numpy(float),
            df[pairs.Q_ID].map(by_kind["quencher"]).to_numpy(float))


def radii(df):
    """{radius: (r_PC, r_Q)} for both radius definitions."""
    return {"volume": (df[pairs.PC_RADIUS].to_numpy(float), df[pairs.Q_RADIUS].to_numpy(float)),
            "extent": extent_radii(df)}


def threshold_shift(r):
    """t_PC(r) - t_PC(7 A) = 3 RT ln(r / 7 A) in kcal/mol."""
    return 3.0 * theory.RT * np.log(np.asarray(r, float) / R_REFERENCE)


def two_gate(df, source, r):
    """(predicted quenching, continuous score) of two_gate at the contact distance r in A,
    one value or one per pair."""
    dG = pairs.driving_force(df, source)
    barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(df, source)))
    t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), classification.K_SV_CRIT,
                                   theory.escape_over_frequency(np.asarray(r, float)))
    margin = t_pc - barrier
    exergonic = dG <= 0
    return (exergonic & (margin >= 0),
            np.where(exergonic, margin, margin - classification.ENDERGONIC_OFFSET))


def pair_distances(df, r):
    out = pd.DataFrame({"pc": df[pairs.PC_NAME], "quencher": df[pairs.Q_NAME],
                        "quenches": df[pairs.QUENCHES].astype(int)})
    for name, (r_pc, r_q) in r.items():
        out[f"r_PC_{name} / A"] = r_pc
        out[f"r_Q_{name} / A"] = r_q
        out[f"r_ij_{name} / A"] = r_pc + r_q
    return out


def distribution(df, r):
    rows = []
    for name, (r_pc, r_q) in r.items():
        values = r_pc + r_q
        shift = threshold_shift(values)
        rows.append({"definition": RADII[name], "n": len(values), **_quantiles(values),
                     "fraction below 7 A": float((values < R_REFERENCE).mean()),
                     "dt_PC vs 7 A median / kcal mol-1": float(np.median(shift)),
                     "dt_PC vs 7 A min / kcal mol-1": float(shift.min()),
                     "dt_PC vs 7 A max / kcal mol-1": float(shift.max())})
        for part, values_, ids in (("r_PC", r_pc, df[pairs.PC_NAME]), ("r_Q", r_q, df[pairs.Q_NAME])):
            per_molecule = pd.Series(values_, index=ids.to_numpy()).groupby(level=0).first()
            rows.append({"definition": f"{RADII[name]}: {part} (per molecule)", "n": len(per_molecule),
                         **_quantiles(per_molecule.to_numpy())})
    return pd.DataFrame(rows)


def _quantiles(values):
    values = np.asarray(values, float)
    return {"min / A": values.min(), "q05 / A": np.quantile(values, 0.05),
            "q25 / A": np.quantile(values, 0.25), "median / A": np.median(values),
            "q75 / A": np.quantile(values, 0.75), "q95 / A": np.quantile(values, 0.95),
            "max / A": values.max(), "mean / A": values.mean()}


def threshold_per_angstrom(df):
    tau0 = df.groupby(pairs.PC_NAME)[pairs.TAU0].first().to_numpy(float) * 1e-9
    t_pc = np.array([theory.lifetime_barrier(tau0, classification.K_SV_CRIT,
                                             theory.escape_over_frequency(r)) for r in R_GRID])
    return pd.DataFrame({"r / A": R_GRID, "K_A / M-1": theory.encounter_constant(R_GRID),
                         "k_-d/nu": theory.escape_over_frequency(R_GRID),
                         "dt_PC vs 7 A / kcal mol-1": threshold_shift(R_GRID),
                         "dt_PC/dr = 3RT/r / kcal mol-1 A-1": 3.0 * theory.RT / R_GRID,
                         "t_PC min / kcal mol-1": t_pc.min(axis=1),
                         "t_PC median / kcal mol-1": np.median(t_pc, axis=1),
                         "t_PC max / kcal mol-1": t_pc.max(axis=1)})


def metrics(df, r):
    y = df[pairs.QUENCHES].to_numpy(bool)
    photocatalysts = df[pairs.PC_NAME].to_numpy()

    def row(kind, label, source, distance):
        predicted, score = two_gate(df, source, distance)
        m = scoring.metrics(y, predicted)
        within = scoring.auc_within(y, score, photocatalysts)
        return {"r": kind, "r / A": label, "source": source,
                **{k: m[k] for k in ("n", "TP", "FP", "TN", "FN", "accuracy", "MCC")},
                "ROC_AUC": scoring.auc(y, score), "AUC_within_PC_pooled": within["pooled"],
                "AUC_within_PC_median": within["median"], "specificity": m["specificity"],
                "recall": m["recall"]}

    rows = []
    for source in pairs.SOURCES:
        rows.append(row("fixed", f"{R_REFERENCE:g}", source, R_REFERENCE))
        for name, (r_pc, r_q) in r.items():
            r_ij = r_pc + r_q
            rows.append(row(f"per pair r_PC + r_Q, {RADII[name]}", f"{np.median(r_ij):.2f} (median)",
                            source, r_ij))
        rows += [row("sweep", f"{d:g}", source, float(d)) for d in R_GRID]
    return pd.DataFrame(rows)


def main():
    df = pairs.scored(pairs.load())
    r = radii(df)
    if any(np.isnan(r_pc).any() or np.isnan(r_q).any() for r_pc, r_q in r.values()):
        raise ValueError(f"a scored pair has no radius; run data/computational_data_scripts/"
                         f"molecular_extent.py to write {EXTENT.name}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pair_distances(df, r).round(2).to_csv(RESULTS / "pair_distances.csv", index=False,
                                          lineterminator="\n")
    dist = distribution(df, r)
    dist.round(2).to_csv(RESULTS / "r_ij_distribution.csv", index=False, lineterminator="\n")
    per_angstrom = threshold_per_angstrom(df)
    per_angstrom.round({"K_A / M-1": 4, "k_-d/nu": 7, "dt_PC vs 7 A / kcal mol-1": 2,
                        "dt_PC/dr = 3RT/r / kcal mol-1 A-1": 3, "t_PC min / kcal mol-1": 2,
                        "t_PC median / kcal mol-1": 2, "t_PC max / kcal mol-1": 2}).to_csv(
        RESULTS / "dt_pc_per_angstrom.csv", index=False, lineterminator="\n")
    table = metrics(df, r)
    scoring.rounded(table).to_csv(RESULTS / "metrics_r.csv", index=False, lineterminator="\n")

    print(dist.round(2).to_string(index=False))
    print()
    print(table[["r", "r / A", "source", "MCC", "ROC_AUC", "AUC_within_PC_pooled"]]
          .round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
