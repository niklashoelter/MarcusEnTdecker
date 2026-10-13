# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Predicted against measured quenching rate constants within series that share one partner,
and the pairs of single quenchers across all photocatalysts.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/kinetics_series/series.csv     Spearman rho of predicted against measured
                                               log10 k_q in every photocatalyst series and
                                               every quencher series
        results/kinetics_series/summary.csv    per kind of series and source: number of
                                               series, number with rho > 0, median, minimum
                                               and maximum rho
        results/kinetics_series/quenchers.csv  every scored pair of the quenchers in
                                               QUENCHERS: measurement, predicted log10 k and
                                               the predictions of the criteria
        results/kinetics_series/quencher_summary.csv
                                               the same quenchers: counts, driving force of
                                               the quenching pairs, errors of the criteria,
                                               rho and median bias

    uv run benchmark/further_analysis/kinetics_series.py

Predicted log10 k: the full rate law with the asymmetric barrier and k_-d/nu at the contact
distance r_PC + r_Q of each pair (rate_predictions.log_k_full), DFT or ML quencher
parameters. A series is every pair with k_q > 0 (quenching or not) that shares one
photocatalyst or one quencher; series with fewer than MIN_SERIES pairs are left out.

QUENCHERS: the quencher with the most pairs of k_q > 0 (chosen by that rule) and Q2.
Criteria: adiabatic (DFT), DvTE (p = 0.008, normal) and two_gate (r from DFT sizes, DFT and
ML), as in classification.py. Minimum driving force: the smallest -dG_EnT (DFT) of the
quenching pairs. Excess barrier: barrier minus t_PC of the quenching pairs that two_gate
(DFT) misses. Median bias: median of predicted minus measured log10 k_q over the quenching
pairs with k_q > 0.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import rate_predictions  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "kinetics_series"
MIN_SERIES = 5
SERIES = {"photocatalyst": pairs.PC_NAME, "quencher": pairs.Q_NAME}
FIXED_QUENCHERS = ("Q2",)
MEASURED = "log10 k_q measured"
Q_E_T = "q_E_T_dft / kcal mol-1"
CRITERIA = {
    "adiabatic dft": (classification.Setting("adiabatic"), "dft"),
    "dvte": (classification.Setting("dvte", p="0.008", dvte_kind="normal"), "dft"),
    **{f"two_gate {s}": (classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                                k_sv_crit=classification.K_SV_CRIT), s)
       for s in pairs.SOURCES},
}


def predicted_columns(df):
    return {s: rate_predictions.log_k_full(df, s) for s in pairs.SOURCES}


def measured(df):
    k_q = df[pairs.K_Q].to_numpy(float)
    return np.where(k_q > 0, np.log10(np.where(k_q > 0, k_q, 1.0)), np.nan)


def series(df, predicted, log_k_q, min_size=MIN_SERIES):
    """Spearman rho of every series with at least min_size pairs of k_q > 0."""
    has = np.isfinite(log_k_q)
    rows = []
    for kind, column in SERIES.items():
        names = df[column].to_numpy()
        for name in pd.unique(names[has]):
            inside = has & (names == name)
            if inside.sum() < min_size:
                continue
            for source in pairs.SOURCES:
                rows.append({"kind": kind, "series": name, "parameters": source,
                             "n": int(inside.sum()),
                             "Spearman rho": spearmanr(predicted[source][inside],
                                                       log_k_q[inside]).statistic})
    return pd.DataFrame(rows)


def summarize(table):
    return (table.groupby(["kind", "parameters"], sort=False)["Spearman rho"]
            .agg(n_series="size", rho_positive=lambda r: int((r > 0).sum()),
                 median="median", minimum="min", maximum="max")
            .reset_index())


def rule_quencher(df, log_k_q):
    """The quencher with the most pairs of k_q > 0."""
    counts = df.loc[np.isfinite(log_k_q), pairs.Q_NAME].value_counts()
    top = counts[counts == counts.max()]
    if len(top) > 1:
        raise ValueError(f"no single quencher has the most pairs with k_q > 0: {list(top.index)}")
    return top.index[0]


def quencher_pairs(df, predicted, log_k_q, names):
    """Every scored pair of the named quenchers."""
    t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), classification.K_SV_CRIT,
                                   theory.escape_over_frequency(pairs.contact_distances(df)))
    dG = pairs.driving_force(df, "dft")
    barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(df, "dft")))
    table = pd.DataFrame({
        "quencher": df[pairs.Q_NAME], "photocatalyst": df[pairs.PC_NAME],
        pairs.PC_E_T: df[pairs.PC_E_T], Q_E_T: df[Q_E_T],
        pairs.QUENCHES: df[pairs.QUENCHES].astype(int), MEASURED: log_k_q,
        **{f"log10 k predicted {s}": predicted[s] for s in pairs.SOURCES},
        "dG_EnT_dft / kcal mol-1": dG, "barrier_dft / kcal mol-1": barrier,
        "t_PC / kcal mol-1": t_pc,
        **{f"predicted {name}": classification.predict(df, setting, s)[0].astype(int)
           for name, (setting, s) in CRITERIA.items()},
    })
    return pd.concat([table[table.quencher == name].sort_values(pairs.PC_E_T, kind="stable")
                      for name in names], ignore_index=True)


def quencher_summary(table, series_table):
    rows = []
    q_series = series_table[series_table.kind == "quencher"]
    for name, g in table.groupby("quencher", sort=False):
        y = g[pairs.QUENCHES].to_numpy(bool)
        has = g[MEASURED].notna().to_numpy()
        row = {"quencher": name, "photocatalysts": len(g), "with k_q > 0": int(has.sum()),
               "quenching": int(y.sum()),
               "E_T,Q dft / kcal mol-1": g[Q_E_T].iloc[0],
               "min E_T,PC of quenching pairs / kcal mol-1": g.loc[y, pairs.PC_E_T].min(),
               "min driving force of quenching pairs / kcal mol-1": -g.loc[y, "dG_EnT_dft / kcal mol-1"].max()}
        for criterion in CRITERIA:
            p = g[f"predicted {criterion}"].to_numpy(bool)
            row[f"FP {criterion}"] = int((p & ~y).sum())
            row[f"FN {criterion}"] = int((~p & y).sum())
        missed = y & ~g["predicted two_gate dft"].to_numpy(bool)
        row["median excess barrier of two_gate dft misses / kcal mol-1"] = (
            (g["barrier_dft / kcal mol-1"] - g["t_PC / kcal mol-1"])[missed].median())
        for source in pairs.SOURCES:
            s = q_series[(q_series.series == name) & (q_series.parameters == source)]
            row[f"Spearman rho {source}"] = s["Spearman rho"].iloc[0] if len(s) else np.nan
            if source == "dft" and len(s):
                ranked = q_series[q_series.parameters == "dft"]["Spearman rho"]
                row["rank of rho dft among quencher series"] = int((ranked > s["Spearman rho"].iloc[0]).sum()) + 1
            on = y & has
            row[f"median bias {source} / decades"] = np.median(
                g[f"log10 k predicted {source}"].to_numpy()[on] - g[MEASURED].to_numpy()[on])
        rows.append(row)
    return pd.DataFrame(rows)


def rounded(table):
    """kcal/mol to 2 decimals, every other float to 3."""
    table = table.copy()
    for column in table.columns:
        if table[column].dtype.kind == "f":
            table[column] = table[column].round(2 if "kcal mol-1" in column else 3)
    return table


def main():
    df = pairs.scored(pairs.load())
    predicted, log_k_q = predicted_columns(df), measured(df)
    table = series(df, predicted, log_k_q)
    names = list(dict.fromkeys([rule_quencher(df, log_k_q), *FIXED_QUENCHERS]))
    single = quencher_pairs(df, predicted, log_k_q, names)
    tables = {"series": table, "summary": summarize(table), "quenchers": single,
              "quencher_summary": quencher_summary(single, table)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        rounded(t).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")
    print(f"{int(np.isfinite(log_k_q).sum())} pairs with k_q > 0")
    print(tables["summary"].round(3).to_string(index=False))
    print(tables["quencher_summary"].round(3).T.to_string(header=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
