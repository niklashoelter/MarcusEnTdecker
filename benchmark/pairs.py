"""
Loads the energy-transfer dataset and names its columns.

Reads   ../data/exp_dataset_ent.csv

The dataset holds every pair that does not quench, and the quenching pairs for which
single-electron transfer is ruled out (see ../data/build_exp_dataset_ent.py). It carries
the measurement, the DFT and ML properties and the driving forces, so nothing else is
read.

A pair is scored unless `excluded` gives a reason:
    PC22            its emission was too weak for the Stern-Volmer intensities
    missing input   a property a criterion needs is blank; every criterion is scored on
                    the same pairs

Not a script; imported by classification.py and rates.py.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASET = HERE.parent / "data" / "exp_dataset_ent.csv"

PC_NAME = "pc_name"
Q_NAME = "quencher_name"
TAU0 = "tau0 / ns"
K_Q = "k_q / M-1 s-1"
QUENCHES = "quenches"
EXCLUDED = "excluded"

PC_E_T = "pc_E_T_dft / kcal mol-1"
PC_LAMBDA_S0_T1 = "pc_lambda_S0_T1_dft / kcal mol-1"
PC_LAMBDA_T1_S0 = "pc_lambda_T1_S0_dft / kcal mol-1"
Q_DVTE_MU = "q_dvte_mu / kcal mol-1"
Q_DVTE_STD = "q_dvte_std / kcal mol-1"

# Quencher E_T, lambda and the driving force exist from DFT and from the ML model.
SOURCES = ("dft", "ml")


def q_lambda_s0_t1(source):
    return f"q_lambda_S0_T1_{source} / kcal mol-1"


def q_lambda_t1_s0(source):
    return f"q_lambda_T1_S0_{source} / kcal mol-1"


def dg_ent(source):
    return f"dG_EnT_{source} / kcal mol-1"


NOT_SCORED = {"PC22": "PC22: emission too weak"}

# Every input a classification criterion reads.
INPUTS = [TAU0, PC_E_T, PC_LAMBDA_S0_T1, PC_LAMBDA_T1_S0, Q_DVTE_MU, Q_DVTE_STD,
          *(column(s) for s in SOURCES for column in (q_lambda_s0_t1, q_lambda_t1_s0, dg_ent))]


def load() -> pd.DataFrame:
    """The dataset, with `excluded` saying which pairs are not scored."""
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET} not found; run `uv run reproduce.py data` first")
    df = pd.read_csv(DATASET)
    missing = [c for c in INPUTS + [PC_NAME, Q_NAME, K_Q, QUENCHES] if c not in df.columns]
    if missing:
        raise KeyError(f"{DATASET.name} has no column {', '.join(missing)}")

    reason = pd.Series("", index=df.index)
    blank = df[INPUTS].isna()
    reason[blank.any(axis=1)] = [
        "missing input: " + ", ".join(c.split(" / ")[0] for c in blank.columns[row])
        for row in blank[blank.any(axis=1)].to_numpy()]
    for name, why in NOT_SCORED.items():
        reason[df[PC_NAME] == name] = why
    df[EXCLUDED] = reason
    return df


def scored(df: pd.DataFrame) -> pd.DataFrame:
    """The pairs every benchmark is scored on."""
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
    lines = [f"{len(df)} pairs in {DATASET.name}, {len(kept)} scored "
             f"({int(kept[QUENCHES].sum())} quench, {int((kept[QUENCHES] == 0).sum())} do not)"]
    for reason, n in df.loc[df[EXCLUDED] != "", EXCLUDED].value_counts().items():
        lines.append(f"  not scored  {n:4d}  {reason}")
    return "\n".join(lines)
