"""
Loads the energy-transfer dataset and names its columns.

Reads   ../data/exp_dataset_ent.csv

PC22 is left out on loading. Of the remaining pairs, a pair is scored only if every
input of every criterion is present; `excluded` names the blank inputs of the others.

Not a script; imported by the benchmark and further_analysis scripts.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASET = HERE.parent / "data" / "exp_dataset_ent.csv"

PC_NAME = "pc_name"
Q_ID = "quencher_id"
Q_NAME = "quencher_name"
K_SV = "K_SV / M-1"
TAU0 = "tau0 / ns"
K_Q = "k_q / M-1 s-1"
QUENCHES = "quenches"
EXCLUDED = "excluded"

PC_E_T = "pc_E_T_dft / kcal mol-1"
PC_LAMBDA_S0_T1 = "pc_lambda_S0_T1_dft / kcal mol-1"
PC_LAMBDA_T1_S0 = "pc_lambda_T1_S0_dft / kcal mol-1"

# Quencher E_T, lambda and the driving force exist from DFT and from the ML model.
SOURCES = ("dft", "ml")
DVTE_P = ("0.005", "0.008", "0.01", "0.015")
DVTE_KINDS = ("normal", "kde")

LEFT_OUT = {"PC22": "emission too weak for Stern-Volmer analysis"}


def q_lambda_s0_t1(source):
    return f"q_lambda_S0_T1_{source} / kcal mol-1"


def q_lambda_t1_s0(source):
    return f"q_lambda_T1_S0_{source} / kcal mol-1"


def dg_ent(source):
    return f"dG_EnT_{source} / kcal mol-1"


def q_dvte(p, kind):
    """The p-quantile of the vertical gaps, from a normal distribution or a KDE."""
    return f"q_dvte_{p}{'_kde' if kind == 'kde' else ''} / kcal mol-1"


def q_ml_uncertainty(name):
    """The model's uncertainty on E_T, lambda_S0_T1 or lambda_T1_S0."""
    return f"q_{name}_ml_uncertainty / kcal mol-1"


# Every input a benchmark criterion reads.
INPUTS = [TAU0, PC_E_T, PC_LAMBDA_S0_T1, PC_LAMBDA_T1_S0,
          *(q_dvte(p, kind) for p in DVTE_P for kind in DVTE_KINDS),
          *(column(s) for s in SOURCES for column in (q_lambda_s0_t1, q_lambda_t1_s0, dg_ent))]


def load() -> pd.DataFrame:
    """The dataset without LEFT_OUT, with `excluded` naming the blank inputs of a pair."""
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET} not found; run `uv run reproduce.py data` first")
    df = pd.read_csv(DATASET)
    missing = [c for c in INPUTS + [PC_NAME, Q_ID, Q_NAME, K_Q, QUENCHES] if c not in df.columns]
    if missing:
        raise KeyError(f"{DATASET.name} has no column {', '.join(missing)}")

    left_out = {name: int((df[PC_NAME] == name).sum()) for name in LEFT_OUT}
    df = df[~df[PC_NAME].isin(list(LEFT_OUT))].reset_index(drop=True)
    df.attrs["left_out"] = left_out

    reason = pd.Series("", index=df.index)
    blank = df[INPUTS].isna()
    reason[blank.any(axis=1)] = [
        "missing input: " + ", ".join(c.split(" / ")[0] for c in blank.columns[row])
        for row in blank[blank.any(axis=1)].to_numpy()]
    df[EXCLUDED] = reason
    return df


def scored(df: pd.DataFrame) -> pd.DataFrame:
    """The pairs every criterion is scored on."""
    return df[df[EXCLUDED] == ""].reset_index(drop=True)


def driving_force(df, source):
    return df[dg_ent(source)].to_numpy(float)


def lambdas(df, source):
    """(pc_lambda_S0_T1, pc_lambda_T1_S0, q_lambda_S0_T1, q_lambda_T1_S0) as arrays.

    Photocatalyst values are DFT for both sources.
    """
    return (df[PC_LAMBDA_S0_T1].to_numpy(float), df[PC_LAMBDA_T1_S0].to_numpy(float),
            df[q_lambda_s0_t1(source)].to_numpy(float),
            df[q_lambda_t1_s0(source)].to_numpy(float))


def tau0_seconds(df):
    return df[TAU0].to_numpy(float) * 1e-9


def summary(df) -> str:
    kept = scored(df)
    left_out = df.attrs.get("left_out", {})
    lines = [f"{len(df) + sum(left_out.values())} pairs in {DATASET.name}, {len(kept)} scored "
             f"({int(kept[QUENCHES].sum())} quench, {int((kept[QUENCHES] == 0).sum())} do not)"]
    for name, n in left_out.items():
        lines.append(f"  left out    {n:4d}  {name}: {LEFT_OUT[name]}")
    for reason, n in df.loc[df[EXCLUDED] != "", EXCLUDED].value_counts().items():
        lines.append(f"  not scored  {n:4d}  {reason}")
    return "\n".join(lines)
