# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Two-gate classification with machine-learned instead of DFT quencher parameters.

Reads   ../../data/exp_dataset_ent.csv (also through pairs.py)
        results/quencher_classes/quencher_classes.csv
Writes  results/ml_vs_dft/quenchers.csv   ML - DFT parameters, correct classifications,
                                          and pairs rescued and lost for each quencher
        results/ml_vs_dft/summary.csv     counts over pairs and quenchers

    uv run benchmark/further_analysis/ml_vs_dft.py

Two-gate: DFT with r from DFT sizes, ML with r = 7 A; photocatalyst parameters DFT.
Rescued: correct with ML, wrong with DFT; lost: the reverse.
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

CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"
RESULTS = HERE / "results" / "ml_vs_dft"

TWO_GATE = {s: classification.two_gate_setting(s) for s in pairs.SOURCES}
PARAMETERS = {"E_T": "q_E_T", "lambda_S0_T1": "q_lambda_S0_T1", "lambda_T1_S0": "q_lambda_T1_S0"}


def two_gate(df):
    return classification.predict(df, TWO_GATE["ml"], "ml")


def quencher_table(df, q, classes):
    y = df[pairs.QUENCHES].to_numpy(bool)
    p_dft, _ = classification.predict(df, TWO_GATE["dft"], "dft")
    p_ml, _ = two_gate(df)
    names = df[pairs.Q_NAME].to_numpy()
    rows = []
    for name in q.index:
        r = {"quencher": name, "class": classes.get(name)}
        for p, col in PARAMETERS.items():
            r[f"{p} ML - DFT / kcal mol-1"] = q.loc[name, f"{col}_ml / kcal mol-1"] - q.loc[name, f"{col}_dft / kcal mol-1"]
        m = names == name
        r["n"] = int(m.sum())
        if m.any():
            yy, pd_, pm = y[m], p_dft[m], p_ml[m]
            r |= {"n_quench": int(yy.sum()), "correct_dft": int((pd_ == yy).sum()), "correct_ml": int((pm == yy).sum()),
                  "TP_dft": int((pd_ & yy).sum()), "TP_ml": int((pm & yy).sum()),
                  "FP_dft": int((pd_ & ~yy).sum()), "FP_ml": int((pm & ~yy).sum()),
                  "rescued_quench": int(((pm == yy) & (pd_ != yy) & yy).sum()),
                  "rescued_no_quench": int(((pm == yy) & (pd_ != yy) & ~yy).sum()),
                  "lost_quench": int(((pm != yy) & (pd_ == yy) & yy).sum()),
                  "lost_no_quench": int(((pm != yy) & (pd_ == yy) & ~yy).sum())}
            r["dcorrect_ml_minus_dft"] = r["correct_ml"] - r["correct_dft"]
        rows.append(r)
    t = pd.DataFrame(rows)
    order = ["quencher", "class", *[c for c in t.columns if "ML - DFT" in c],
             "n", "n_quench", "correct_dft", "correct_ml", "dcorrect_ml_minus_dft",
             "TP_dft", "TP_ml", "FP_dft", "FP_ml", "rescued_quench", "rescued_no_quench", "lost_quench",
             "lost_no_quench"]
    t = t[order]
    for c in order[order.index("n_quench"):]:
        t[c] = t[c].astype("Int64")
    return t


def summary(df, quenchers):
    y = df[pairs.QUENCHES].to_numpy(bool)
    p_dft = classification.predict(df, TWO_GATE["dft"], "dft")[0]
    p_ml = two_gate(df)[0]
    rescued, lost = (p_ml == y) & (p_dft != y), (p_ml != y) & (p_dft == y)
    scored = quenchers[quenchers.n > 0].set_index("quencher")
    dc = scored.dcorrect_ml_minus_dft.astype(int)
    losses = dc[dc < 0].sort_values(kind="stable")
    rows = [
        ("pairs", len(y), ""),
        ("correct, DFT", int((p_dft == y).sum()), ""), ("correct, ML", int((p_ml == y).sum()), ""),
        ("rescued by ML", int(rescued.sum()), ""), ("rescued by ML, quenching", int((rescued & y).sum()), ""),
        ("rescued by ML, non-quenching", int((rescued & ~y).sum()), ""),
        ("quenchers with rescued pairs", int((scored.rescued_quench + scored.rescued_no_quench > 0).sum()), ""),
        ("lost by ML", int(lost.sum()), ""), ("lost by ML, quenching", int((lost & y).sum()), ""),
        ("lost by ML, non-quenching", int((lost & ~y).sum()), ""),
        ("quenchers with lost pairs", int((scored.lost_quench + scored.lost_no_quench > 0).sum()), ""),
        ("net change in correct pairs, quenchers losing", int(-losses.sum()), ""),
        ("quenchers with a net loss", len(losses), ""),
        ("net change in correct pairs, quenchers gaining", int(dc[dc > 0].sum()), ""),
        ("quenchers with a net gain", int((dc > 0).sum()), ""),
        ("net loss of the five quenchers losing most", int(-losses.iloc[:5].sum()), " ".join(losses.index[:5])),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "detail"])


def rounded(table):
    """kcal/mol to 2 decimals, everything else to 3; in quantity/value tables by row."""
    table = table.copy()
    if {"quantity", "value"} <= set(table.columns):
        table["value"] = [round(float(v), 2 if "kcal" in qq else 3) for qq, v in zip(table.quantity, table.value)]
    for c in table.select_dtypes(float).columns:
        if c != "value":
            table[c] = table[c].round(2 if "kcal" in c else 3)
    return table


def main():
    q = pd.read_csv(pairs.DATASET).groupby(pairs.Q_NAME).first()
    q = q.sort_index(key=lambda s: s.str[1:].astype(int))
    classes = pd.read_csv(CLASSES).set_index("id")["proposed_class"]
    df = pairs.scored(pairs.load())
    quenchers = quencher_table(df, q, classes)
    tables = {"quenchers": quenchers, "summary": summary(df, quenchers)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        rounded(table).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    print(rounded(tables["summary"]).to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
