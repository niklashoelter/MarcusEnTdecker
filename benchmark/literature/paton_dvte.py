# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Paton 2024: dynamic vertical triplet energies (DvTE) against our predicted reorganisation energies.

    uv run benchmark/literature/paton_dvte.py

Popescu and Paton (Chem 2024, 10, 3428), Figure 3: DvTE and Gibbs adiabatic E_T per
molecule, beside our predicted E_T, lambda_S0_T1 and lambda_T1_S0.
Cyclohexyl azide is kept in the table and excluded downstream (see SI).

Writes paton_dvte_analysis.csv. Predictions from predict.py (case "paton_2024").
"""

import pandas as pd

from common import column, literature, predictions, write

CASE = "paton_2024"

# compound classes by the paper's Figure 3 labels
CLASS = {"1": "alkene", "3": "alkene", "4": "alkene", "5": "alkene", "6": "alkene", "7": "alkene",
         "8": "conj. alkene", "9": "conj. alkene", "10": "conj. alkene", "11": "conj. alkene",
         "2-E": "conj. alkene", "2-Z": "conj. alkene", "12": "conj. alkene", "13": "enone",
         "14": "alkyne", "15": "azo", "16": "arene", "17": "arene", "18": "arene", "19": "arene",
         "20": "oxime", "21": "oxime", "22": "azide", "23": "azide"}


def main():
    lit = literature(CASE)
    pred = predictions(CASE).loc[lit.compound_id]
    table = lit[["compound_id", "label", "name", "smiles", "isomer"]].copy()
    table["class"] = table.label.astype(str).map(CLASS)
    if table["class"].isna().any():
        raise LookupError(f"no class for {list(table.label[table['class'].isna()])}")
    table["E_T_adiabatic_lit"] = lit.E_T_adiabatic_lit.values
    table["dvte_lit"] = lit.dvte_lit.values
    for prop in ("E_T", "lambda_S0_T1", "lambda_T1_S0"):
        table[f"{prop}_ml"] = pred[column(prop)].values
    table["lambda_sum_ml"] = table.lambda_S0_T1_ml + table.lambda_T1_S0_ml
    write(table, "paton_dvte_analysis.csv")


if __name__ == "__main__":
    main()
