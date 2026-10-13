# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Per-quencher, per-photocatalyst, cost and lifetime-threshold tables of the scored pairs.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        ../../data/compound_identification.xlsx               (photocatalyst names)
        ../../data/computational_data_processed/quencher_timings.csv
        ../../data/computational_data_processed/quencher_timings_ml.csv
        results/quencher_classes/quencher_classes.csv         (through metrics_extended.py)
Writes  results/figure_tables/quenchers.csv        one row per quencher with scored pairs
        results/figure_tables/photocatalysts.csv   one row per photocatalyst with scored pairs
        results/figure_tables/cost.csv             CPU and wall time per quencher of each
                                                   cost method, with the metrics of every
                                                   criterion it enables
        results/figure_tables/lifetime_curve.csv   t_PC against tau0 for several contact
                                                   distances

    uv run benchmark/further_analysis/figure_tables.py

quenchers.csv: DFT and ML parameters, DvTE at p = 0.008 (normal) with the mean and standard
deviation of the vertical gaps and DvTE - E_T (DFT), the radius r, the number of scored
pairs and the fraction that quench.

photocatalysts.csv: DFT parameters, tau0, r, the median t_PC over the photocatalyst's
scored pairs (r = r_PC + r_Q) and t_PC at r = 7 A, pairs and quench fraction. The complex
class follows from the name: Ru -> Ru(II) tris(diimine); pmb -> Ir(III) tris(NHC);
no brackets -> neutral Ir(C^N)3; otherwise cationic [Ir(C^N)2(N^N)]+.

cost.csv: over the quenchers with scored pairs, the median, quartiles, IQR and mean per
quencher of the CPU time (DFT and DvTE: job wall time x cores; ML: process CPU time of the
inference) and of the wall time (DFT and DvTE: sum of the job wall times; ML: one call,
or the batch call divided by the number of molecules), grouped by ../cost.py. Ratios are
to DvTE. One row per method and criterion it enables, with that criterion's metrics from
metrics_extended.py (both ML methods enable the ML criteria).

lifetime_curve.csv: t_PC = RT ln((k_d tau0 / K_SV,crit - 1) / (k_-d/nu)) on 601 log-spaced
tau0 from 10 ns to 10 us, with the Eyring k_-d/nu at the median, smallest and largest
r_PC + r_Q of the scored pairs and at r = 7 A, and with k_-d/nu = 0.25.

kcal/mol, V and A rounded to 2 decimals, fractions and metrics to 3, times and ratios to
4 significant figures.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import cost  # noqa: E402
import metrics_extended as mx  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "figure_tables"
KC = "kcal mol-1"
TAU_GRID_NS = np.logspace(1, 4, 601)
FIXED_ESCAPE_OVER_NU = 0.25

COST_METHODS = {"adiabatic DFT": "adiabatic DFT", "four-point DFT": "four-point DFT", "DvTE": "DvTE",
                "ML, per call": "ML per call", "ML, batch": "ML batch share per molecule"}
WALL_KIND = {"adiabatic DFT": "sum of the job wall times (jobs may run in parallel)",
             "four-point DFT": "sum of the job wall times (jobs may run in parallel)",
             "DvTE": "sum of the job wall times (jobs may run in parallel)",
             "ML, per call": "wall time of one call (median of 20 repeats)",
             "ML, batch": "wall time of the 48-molecule batch call (median of 20 repeats) / 48"}
# criterion of metrics_extended.py: the cost methods that provide its quencher parameters
ENABLES = {("adiabatic", "dft"): ("adiabatic DFT",), ("adiabatic", "ml"): ("ML, per call", "ML, batch"),
           ("dvte_2024", "dvte"): ("DvTE",),
           ("twogate_lifetime", "dft"): ("four-point DFT",), ("twogate_lifetime", "ml"): ("ML, per call", "ML, batch"),
           ("twogate_lifetime_r7A", "dft"): ("four-point DFT",),
           ("twogate_lifetime_r7A", "ml"): ("ML, per call", "ML, batch")}
COST_METRICS = ("accuracy", "MCC", "ROC_AUC", "AUC_within_PC_median", "AUC_within_PC_pooled")


def r2(x):
    return np.round(np.asarray(x, float), 2)


def significant(x, digits=4):
    return float(f"{x:.{digits}g}")


def numbered(table, column, prefix):
    """Rows ordered by the number after the prefix (Q2 before Q10)."""
    order = np.argsort([int(v[len(prefix):]) for v in table[column]], kind="stable")
    return table.iloc[order].reset_index(drop=True)


def t_pc(df, distance=None):
    r = pairs.contact_distances(df) if distance is None else distance
    return theory.lifetime_barrier(pairs.tau0_seconds(df), classification.K_SV_CRIT,
                                   theory.escape_over_frequency(r))


def quencher_table(df, classes):
    y = df[pairs.QUENCHES].to_numpy(bool)
    g = pd.DataFrame({"quencher": df[pairs.Q_NAME], "quenches": y}).groupby("quencher", sort=False).quenches
    first = df.groupby(pairs.Q_NAME, sort=False).first()
    q = first.index.to_numpy()
    t = pd.DataFrame({"quencher": q, "name": classes["name"].reindex(q).to_numpy(),
                      "quencher_class": classes["proposed_class"].reindex(q).to_numpy()})
    for s in pairs.SOURCES:
        t[f"E_T_{s} / {KC}"] = r2(first[f"q_E_T_{s} / {KC}"])
    for s in pairs.SOURCES:
        t[f"lambda_S0_T1_{s} / {KC}"] = r2(first[pairs.q_lambda_s0_t1(s)])
    for s in pairs.SOURCES:
        t[f"lambda_T1_S0_{s} / {KC}"] = r2(first[pairs.q_lambda_t1_s0(s)])
    dvte = first[pairs.q_dvte("0.008", "normal")].to_numpy(float)
    t[f"dvte_0.008 / {KC}"] = r2(dvte)
    t[f"dvte_mean / {KC}"] = r2(first[f"q_dvte_mu / {KC}"])
    t[f"dvte_std / {KC}"] = r2(first[f"q_dvte_std / {KC}"])
    t[f"dvte_0.008_minus_E_T_dft / {KC}"] = r2(dvte - first[f"q_E_T_dft / {KC}"].to_numpy(float))
    t["r / A"] = r2(first[pairs.Q_RADIUS])
    t["n_pairs"] = g.size().reindex(q).to_numpy()
    t["fraction_quenching"] = np.round(g.mean().reindex(q).to_numpy(), 3)
    return numbered(t, "quencher", "Q")


def complex_class(name):
    if not isinstance(name, str):
        return ""
    if name.lstrip("[").startswith("Ru"):
        return "Ru(II) tris(diimine)"
    if "pmb" in name:
        return "Ir(III) homoleptic tris(NHC), neutral"
    if not name.startswith("["):
        return "Ir(III) homoleptic Ir(C^N)3, neutral"
    return "Ir(III) heteroleptic [Ir(C^N)2(N^N)]+, cationic"


def photocatalyst_table(df, names):
    y = df[pairs.QUENCHES].to_numpy(bool)
    per_pair = pd.DataFrame({"pc": df[pairs.PC_NAME], "quenches": y, "t": t_pc(df),
                             "t7": t_pc(df, theory.CONTACT_DISTANCE)}).groupby("pc", sort=False)
    first = df.groupby(pairs.PC_NAME, sort=False).first()
    p = first.index.to_numpy()
    t = pd.DataFrame({"pc": p, "compound": [names.get(x, "") for x in p]})
    t["complex_class"] = t["compound"].map(complex_class)
    t[f"E_T_dft / {KC}"] = r2(first[pairs.PC_E_T])
    t[f"lambda_S0_T1_dft / {KC}"] = r2(first[pairs.PC_LAMBDA_S0_T1])
    t[f"lambda_T1_S0_dft / {KC}"] = r2(first[pairs.PC_LAMBDA_T1_S0])
    t[pairs.TAU0] = first[pairs.TAU0].to_numpy(float)
    t["r / A"] = r2(first[pairs.PC_RADIUS])
    t[f"t_PC_median / {KC}"] = r2(per_pair.t.median().reindex(p))
    t[f"t_PC_7A / {KC}"] = r2(per_pair.t7.first().reindex(p))
    t["n_pairs"] = per_pair.size().reindex(p).to_numpy()
    t["fraction_quenching"] = np.round(per_pair.quenches.mean().reindex(p).to_numpy(), 3)
    return numbered(t, "pc", "PC")


def quartiles(v, prefix):
    q25, q75 = v.quantile(0.25), v.quantile(0.75)
    return {f"{prefix}_median": v.median(), f"{prefix}_q25": q25, f"{prefix}_q75": q75,
            f"{prefix}_IQR": q75 - q25, f"{prefix}_mean": v.mean()}


def per_quencher_seconds(timings, timings_ml):
    """(CPU seconds, wall seconds) per quencher (rows) and cost method (columns)."""
    cpu_h = cost.per_quencher(timings).merge(
        cost.ml_hours(timings_ml, cost.per_quencher(timings)["quencher_name"]),
        on="quencher_name").set_index("quencher_name")
    wall = cost.per_quencher(timings.assign(**{"cpu / h": timings["wall / s"]})).set_index("quencher_name")
    individual = timings_ml[timings_ml["mode"] == "individual"].set_index("quencher_name")
    batch = timings_ml[timings_ml["mode"] == "batch"].iloc[0]
    wall["ML, per call"] = individual["wall / s"].reindex(wall.index)
    wall["ML, batch"] = batch["wall / s"] / batch["n_molecules"]
    return cpu_h * 3600.0, wall


def cost_table(cpu_s, wall_s, metrics):
    rows = []
    for method, label in COST_METHODS.items():
        v, w = cpu_s[method], wall_s[method]
        base = {"method": label, "n_quenchers": int(v.notna().sum()), **quartiles(v, "cpu_s"),
                **quartiles(w, "wall_s"), "wall_kind": WALL_KIND[method],
                "cpu_h_median": v.median() / 3600.0, "cpu_h_mean": v.mean() / 3600.0,
                "ratio_of_cpu_medians_to_DvTE": v.median() / cpu_s["DvTE"].median(),
                "median_per_quencher_cpu_ratio_to_DvTE": (v / cpu_s["DvTE"]).median()}
        for _, m in metrics.iterrows():
            if method in ENABLES.get((m["criterion"], m["source"]), ()):
                rows.append({**base, "criterion": m["criterion"], "source": m["source"], "role": m["role"],
                             **{x: m[x] for x in COST_METRICS}})
    t = pd.DataFrame(rows)
    times = [c for c in t.columns if c.startswith(("cpu_", "wall_s_", "ratio_", "median_per"))]
    t[times] = t[times].map(significant)
    return t


def lifetime_curve(r_pairs):
    r_med, r_min, r_max = np.median(r_pairs), r_pairs.min(), r_pairs.max()
    models = [(f"Eyring k_-d/nu, r = median contact distance ({r_med:.2f} A)", r_med, "main"),
              (f"Eyring k_-d/nu, r = smallest contact distance ({r_min:.2f} A)", r_min, "range"),
              (f"Eyring k_-d/nu, r = largest contact distance ({r_max:.2f} A)", r_max, "range"),
              (f"Eyring k_-d/nu, r = {theory.CONTACT_DISTANCE:.0f} A", theory.CONTACT_DISTANCE, "secondary")]
    curves = [(name, round(float(r), 2), theory.escape_over_frequency(r), role) for name, r, role in models]
    curves.append((f"k_-d/nu = {FIXED_ESCAPE_OVER_NU}", np.nan, FIXED_ESCAPE_OVER_NU, "SI"))
    parts = [pd.DataFrame({"model": name, "role": role, "r / A": r, "k_-d_over_nu": significant(ratio),
                           pairs.TAU0: np.round(TAU_GRID_NS, 3),
                           f"t_PC / {KC}": r2(theory.lifetime_barrier(TAU_GRID_NS * 1e-9, classification.K_SV_CRIT,
                                                                      ratio))})
             for name, r, ratio, role in curves]
    return pd.concat(parts, ignore_index=True)


def main():
    sys.path.insert(0, str(cost.DATA))
    from compound_identification import PHOTOCATALYSTS

    df, _, crits = mx.load()
    classes = pd.read_csv(mx.CLASSES).set_index("id")
    names = {f"PC{r['new_id']}": r["name"] for r in PHOTOCATALYSTS.values()}
    metrics = scoring.rounded(mx.metrics_table(df, crits))

    for path in (cost.TIMINGS, cost.TIMINGS_ML):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the timing scripts in data/ first")
    cpu_s, wall_s = per_quencher_seconds(pd.read_csv(cost.TIMINGS), pd.read_csv(cost.TIMINGS_ML, comment="#"))
    scored = sorted(df[pairs.Q_NAME].unique(), key=lambda q: int(q[1:]))
    missing = set(scored) - set(cpu_s.index)
    if missing:
        raise KeyError(f"no timings for {', '.join(sorted(missing))}")

    tables = {"quenchers.csv": quencher_table(df, classes),
              "photocatalysts.csv": photocatalyst_table(df, names),
              "cost.csv": cost_table(cpu_s.loc[scored], wall_s.loc[scored], metrics),
              "lifetime_curve.csv": lifetime_curve(pairs.contact_distances(df))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(RESULTS / name, index=False, lineterminator="\n")
        print(f"{name:20s} {table.shape[0]:5d} rows x {table.shape[1]:2d} columns")
    c = tables["cost.csv"].drop_duplicates("method")
    print()
    print(c[["method", "n_quenchers", "cpu_h_median", "cpu_s_IQR", "ratio_of_cpu_medians_to_DvTE"]]
          .to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
