# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Two-gate classification with machine-learned instead of DFT quencher parameters.

Reads   ../../data/exp_dataset_ent.csv (also through pairs.py)
        results/quencher_classes/quencher_classes.csv
Writes  results/ml_vs_dft/quenchers.csv         per quencher: ML - DFT parameters, summed ML
                                                uncertainty, pairs classified correctly with DFT
                                                and with ML, pairs rescued and lost, and the
                                                change in the within-PC AUC when only this
                                                quencher gets DFT parameters
        results/ml_vs_dft/summary.csv           counts and correlations over pairs and quenchers

    uv run benchmark/further_analysis/ml_vs_dft.py

Two-gate, r from DFT sizes; photocatalyst parameters DFT. ML/DFT: source of E_T(Q),
lambda_S0-T1(Q), lambda_T1-S0(Q). "Only this quencher DFT": all three from DFT for its
pairs, ML for the rest. Rescued: correct with ML, wrong with DFT; lost: the reverse.
Summed ML uncertainty: sum of the three predictive SDs of a quencher.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402

CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"
RESULTS = HERE / "results" / "ml_vs_dft"

TWO_GATE = classification.candidates("two_gate", "r from DFT sizes")[0]
PARAMETERS = {"E_T": "q_E_T", "lambda_S0_T1": "q_lambda_S0_T1", "lambda_T1_S0": "q_lambda_T1_S0"}


def with_dft(df, parameters, rows=None):
    """df whose ML quencher `parameters` are the DFT ones (on `rows` only, if given)."""
    d = df.copy()
    rows = np.ones(len(df), bool) if rows is None else rows
    for p in parameters:
        ml, dft = ((pairs.dg_ent(s) if p == "E_T" else f"{PARAMETERS[p]}_{s} / kcal mol-1") for s in ("ml", "dft"))
        d[ml] = np.where(rows, df[dft], df[ml])
    return d


def two_gate(df):
    return classification.predict(df, TWO_GATE, "ml")


def pooled_within(y, score, groups):
    return scoring.auc_within(y, score, groups)["pooled"]


def quencher_table(df, q, classes):
    y, g = df[pairs.QUENCHES].to_numpy(bool), df[pairs.PC_NAME].to_numpy()
    p_dft, _ = classification.predict(df, TWO_GATE, "dft")
    p_ml, s_ml = classification.predict(df, TWO_GATE, "ml")
    base = pooled_within(y, s_ml, g)
    names = df[pairs.Q_NAME].to_numpy()
    rows = []
    for name in q.index:
        r = {"quencher": name, "class": classes.get(name)}
        for p, col in PARAMETERS.items():
            r[f"{p} ML - DFT / kcal mol-1"] = q.loc[name, f"{col}_ml / kcal mol-1"] - q.loc[name, f"{col}_dft / kcal mol-1"]
        r["ML uncertainty sum / kcal mol-1"] = sum(q.loc[name, pairs.q_ml_uncertainty(p)] for p in PARAMETERS)
        m = names == name
        r["n"] = int(m.sum())
        if m.any():
            yy, pd_, pm = y[m], p_dft[m], p_ml[m]
            _, s_x = two_gate(with_dft(df, PARAMETERS, m))
            r |= {"n_quench": int(yy.sum()), "correct_dft": int((pd_ == yy).sum()), "correct_ml": int((pm == yy).sum()),
                  "TP_dft": int((pd_ & yy).sum()), "TP_ml": int((pm & yy).sum()),
                  "FP_dft": int((pd_ & ~yy).sum()), "FP_ml": int((pm & ~yy).sum()),
                  "rescued_quench": int(((pm == yy) & (pd_ != yy) & yy).sum()),
                  "rescued_no_quench": int(((pm == yy) & (pd_ != yy) & ~yy).sum()),
                  "lost_quench": int(((pm != yy) & (pd_ == yy) & yy).sum()),
                  "lost_no_quench": int(((pm != yy) & (pd_ == yy) & ~yy).sum()),
                  "dAUC_within_PC_pooled_if_dft": pooled_within(y, s_x, g) - base}
            r["dcorrect_ml_minus_dft"] = r["correct_ml"] - r["correct_dft"]
        rows.append(r)
    t = pd.DataFrame(rows)
    order = ["quencher", "class", *[c for c in t.columns if "ML - DFT" in c], "ML uncertainty sum / kcal mol-1",
             "n", "n_quench", "correct_dft", "correct_ml", "dcorrect_ml_minus_dft",
             "TP_dft", "TP_ml", "FP_dft", "FP_ml", "rescued_quench", "rescued_no_quench", "lost_quench",
             "lost_no_quench", "dAUC_within_PC_pooled_if_dft"]
    t = t[order]
    for c in order[order.index("n_quench"):-1]:
        t[c] = t[c].astype("Int64")
    return t


def summary(df, quenchers, q):
    y = df[pairs.QUENCHES].to_numpy(bool)
    p_dft = classification.predict(df, TWO_GATE, "dft")[0]
    p_ml = classification.predict(df, TWO_GATE, "ml")[0]
    rescued, lost = (p_ml == y) & (p_dft != y), (p_ml != y) & (p_dft == y)
    scored = quenchers[quenchers.n > 0].set_index("quencher")
    dc = scored.dcorrect_ml_minus_dft.astype(int)
    losses = dc[dc < 0].sort_values(kind="stable")
    abs_dev = sum((q[f"{c}_ml / kcal mol-1"] - q[f"{c}_dft / kcal mol-1"]).abs() for c in PARAMETERS.values())
    sigma = sum(q[pairs.q_ml_uncertainty(p)] for p in PARAMETERS)
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
        ("Spearman rho, ML uncertainty sum vs within-PC AUC gain if DFT, scored quenchers",
         float(stats.spearmanr(scored["ML uncertainty sum / kcal mol-1"], scored.dAUC_within_PC_pooled_if_dft)[0]),
         f"n = {len(scored)}"),
        ("Spearman rho, ML uncertainty sum vs summed |ML - DFT|, all quenchers",
         float(stats.spearmanr(sigma, abs_dev)[0]), f"n = {len(q)}"),
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
    tables = {"quenchers": quenchers, "summary": summary(df, quenchers, q)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        rounded(table).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    print(rounded(tables["summary"]).to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
