# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "openpyxl>=3.1", "pandas"]
# ///
"""
The pair table: one row per photocatalyst-quencher pair, with everything the benchmarks
read.

Reads   ../data/exp_dataset.csv
        ../data/computational_data_processed/photocatalyst_properties_calc.xlsx
        ../data/computational_data_processed/quencher_properties_calc.xlsx
        ../data/computational_data_processed/quencher_properties_ml.xlsx
        ../data/experimental_data_processed/quencher_properties_exp.xlsx
Writes  results/pairs.csv

    uv run benchmark/pairs.py

Measured values come from exp_dataset. Calculated photocatalyst properties, calculated
and predicted quencher properties and measured quencher potentials are joined on by id,
and the driving forces are derived from them.

A pair is scored unless `excluded` gives a reason:
    PC22                  its emission was too weak for the Stern-Volmer intensities
    SET not ruled out     it quenches, and electron transfer is not more than 0.217 V
                          uphill in every channel with a measured quencher potential
    missing input         a property a criterion needs is blank

Signs: a positive driving force is uphill. Energies kcal/mol, potentials V vs SCE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE / "results"

EXP_DATASET = DATA / "exp_dataset.csv"
PC_CALC = DATA / "computational_data_processed" / "photocatalyst_properties_calc.xlsx"
Q_CALC = DATA / "computational_data_processed" / "quencher_properties_calc.xlsx"
Q_ML = DATA / "computational_data_processed" / "quencher_properties_ml.xlsx"
Q_EXP = DATA / "experimental_data_processed" / "quencher_properties_exp.xlsx"

# ---------------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------------

PC_ID = "pc_new_id"
PC_NAME = "pc_name"
Q_ID = "quencher_id"
Q_NAME = "quencher_name"
K_SV = "K_SV / M-1"
TAU0 = "tau0 / ns"
K_Q = "k_q / M-1 s-1"
QUENCHES = "quenches"
CENSORED = "censored"

PC_E_T = "pc_E_T / kcal mol-1"
PC_LAMBDA_S0_T1 = "pc_lambda_S0_T1 / kcal mol-1"
PC_LAMBDA_T1_S0 = "pc_lambda_T1_S0 / kcal mol-1"
Q_DVTE_MU = "q_dvte_mu / kcal mol-1"
Q_DVTE_STD = "q_dvte_std / kcal mol-1"

# Quencher E_T and lambda exist twice: from DFT and predicted by the ML model.
SOURCES = ("dft", "ml")


def q_e_t(source):
    return f"q_E_T_{source} / kcal mol-1"


def q_lambda_s0_t1(source):
    return f"q_lambda_S0_T1_{source} / kcal mol-1"


def q_lambda_t1_s0(source):
    return f"q_lambda_T1_S0_{source} / kcal mol-1"


def dg_ent(source):
    return f"dG_EnT_{source} / kcal mol-1"


DE_SET_OX = "dE_SET_ox / V"
DE_SET_RED = "dE_SET_red / V"
SET_EXCLUDED = "set_excluded"
EXCLUDED = "excluded"

# ---------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------

NOT_SCORED = {"PC22": "PC22: emission too weak"}

# Electron transfer counts as ruled out when every channel with a measured quencher
# potential is more than this far uphill (0.217 V = 5 kcal/mol).
SET_EXCLUSION_V = 0.217

# (workbook, id column, {workbook header: pair-table header})
JOINS = [
    (PC_CALC, "new_id", PC_ID, {
        "E_T / kcal mol-1": PC_E_T,
        "lambda_S0_T1 / kcal mol-1": PC_LAMBDA_S0_T1,
        "lambda_T1_S0 / kcal mol-1": PC_LAMBDA_T1_S0,
        "E_PCstar_PCminus / V": "pc_E_star_minus / V",
        "E_PCplus_PCstar / V": "pc_E_plus_star / V",
    }),
    (Q_CALC, "quencher_id", Q_ID, {
        "E_T / kcal mol-1": q_e_t("dft"),
        "lambda_S0_T1 / kcal mol-1": q_lambda_s0_t1("dft"),
        "lambda_T1_S0 / kcal mol-1": q_lambda_t1_s0("dft"),
        "dvte_mu / kcal mol-1": Q_DVTE_MU,
        "dvte_std / kcal mol-1": Q_DVTE_STD,
    }),
    (Q_ML, "quencher_id", Q_ID, {
        "E_T / kcal mol-1": q_e_t("ml"),
        "lambda_S0_T1 / kcal mol-1": q_lambda_s0_t1("ml"),
        "lambda_T1_S0 / kcal mol-1": q_lambda_t1_s0("ml"),
    }),
    (Q_EXP, "quencher_id", Q_ID, {
        "E_Ox_vs_SCE_exp / V": "q_E_ox / V",
        "E_Ox_bound": "q_E_ox_bound",
        "E_Red_vs_SCE_exp / V": "q_E_red / V",
        "E_Red_bound": "q_E_red_bound",
    }),
]

# Every input a classification criterion reads. A pair missing any of them is not scored,
# so every criterion is scored on the same pairs.
INPUTS = [TAU0, PC_E_T, PC_LAMBDA_S0_T1, PC_LAMBDA_T1_S0, Q_DVTE_MU, Q_DVTE_STD,
          *(column(s) for s in SOURCES
            for column in (q_e_t, q_lambda_s0_t1, q_lambda_t1_s0))]


# ---------------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------------

def _workbook(path, key, columns):
    """The listed columns of a property workbook, headers on row 2, keyed by id."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run reproduce.py data` first")
    sheet = pd.read_excel(path, header=1)
    sheet = sheet.dropna(subset=[key])
    sheet[key] = sheet[key].astype(int)
    return sheet[[key, *columns]]


def _join(df):
    for path, key, pair_key, columns in JOINS:
        sheet = _workbook(path, key, columns).rename(columns={key: pair_key, **columns})
        df = df.merge(sheet, on=pair_key, how="left", validate="many_to_one")
    return df


def _set_excluded(df):
    """Whether electron transfer is ruled out, from the measured quencher potentials.

    Oxidation of the quencher by PC*:  dE_ox  = E_ox(Q) - E(PC*/PC-)
    Reduction of the quencher by PC*:  dE_red = E(PC+/PC*) - E_red(Q)

    A limit (E_ox '>' or E_red '<') makes the channel at least that far uphill, so it
    is read as its value: past the threshold it rules the channel out, short of it it
    does not.
    """
    for column, allowed in (("q_E_ox_bound", ">"), ("q_E_red_bound", "<")):
        found = set(df[column].dropna().astype(str).str.strip()) - {"", allowed}
        if found:
            raise ValueError(f"{column}: unexpected bound {sorted(found)}")

    df[DE_SET_OX] = df["q_E_ox / V"] - df["pc_E_star_minus / V"]
    df[DE_SET_RED] = df["pc_E_plus_star / V"] - df["q_E_red / V"]
    channels = df[[DE_SET_OX, DE_SET_RED]]
    measured = channels.notna()
    uphill = (channels > SET_EXCLUSION_V) | ~measured
    df[SET_EXCLUDED] = uphill.all(axis=1) & measured.any(axis=1)
    return df


def _excluded(df):
    """The reason a pair is not scored, or '' if it is."""
    reason = pd.Series("", index=df.index)
    reason[(df[QUENCHES] == 1) & ~df[SET_EXCLUDED]] = "SET not ruled out"
    missing = df[INPUTS].isna()
    reason[missing.any(axis=1)] = [
        "missing input: " + ", ".join(c.split(" / ")[0] for c in missing.columns[row])
        for row in missing[missing.any(axis=1)].to_numpy()]
    for name, why in NOT_SCORED.items():
        reason[df[PC_NAME] == name] = why
    return reason


def load() -> pd.DataFrame:
    """The full pair table, all 1296 pairs, with `excluded` saying which are not scored."""
    if not EXP_DATASET.exists():
        raise FileNotFoundError(f"{EXP_DATASET} not found; run `uv run reproduce.py data` first")
    df = _set_excluded(_join(pd.read_csv(EXP_DATASET)))
    df[dg_ent("dft")] = df[q_e_t("dft")] - df[PC_E_T]
    df[dg_ent("ml")] = df[q_e_t("ml")] - df[PC_E_T]
    df[EXCLUDED] = _excluded(df)
    return df


def scored(df: pd.DataFrame) -> pd.DataFrame:
    """The pairs every benchmark is scored on."""
    return df[df[EXCLUDED] == ""].reset_index(drop=True)


# ---------------------------------------------------------------------------------
# Reading it
# ---------------------------------------------------------------------------------

def driving_force(df, source):
    return df[dg_ent(source)].to_numpy(float)


def lambdas(df, source):
    """(pc_lambda_S0_T1, pc_lambda_T1_S0, q_lambda_S0_T1, q_lambda_T1_S0) as arrays.

    The photocatalyst values are DFT for both sources; the ML model covers organic
    molecules only.
    """
    return (df[PC_LAMBDA_S0_T1].to_numpy(float), df[PC_LAMBDA_T1_S0].to_numpy(float),
            df[q_lambda_s0_t1(source)].to_numpy(float),
            df[q_lambda_t1_s0(source)].to_numpy(float))


def tau0_seconds(df):
    return df[TAU0].to_numpy(float) * 1e-9


def summary(df) -> str:
    kept = scored(df)
    lines = [f"{len(df)} pairs, {len(kept)} scored "
             f"({int(kept[QUENCHES].sum())} quench, {int((kept[QUENCHES] == 0).sum())} do not)"]
    for reason, n in df.loc[df[EXCLUDED] != "", EXCLUDED].value_counts().items():
        lines.append(f"  not scored  {n:4d}  {reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    table = load()
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "pairs.csv", index=False)
    print(summary(table))
    print(f"-> {(RESULTS / 'pairs.csv').relative_to(HERE.parent)}")
