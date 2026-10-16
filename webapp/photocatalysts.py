"""The packaged photocatalyst library: data/photocatalysts.csv.

The table is written by scripts/build_photocatalysts.py from the repository's data and
is not edited by hand. One row per photocatalyst, the screened ones (PC<n>) first, then
the organic photocatalysts offered only here (OPC<n>):

    id                  PC<n> or OPC<n>
    name                the compound
    smiles              organic photocatalysts only; the metal complexes have none
    structure_image     a drawing in data/pc_structures/, where there is one
    E_T                 adiabatic triplet energy, DFT, kcal/mol
    lambda_S0_T1        reorganization energy on the S0 surface (at the T1 geometry), kcal/mol
    lambda_T1_S0        reorganization energy on the T1 surface (at the S0 geometry), kcal/mol
    E_PCplus_PCstar     E(PC+/PC*), DFT, V vs SCE; blank where not computed
    E_PCstar_PCminus    E(PC*/PC-), DFT, V vs SCE; blank where not computed
    tau0_ns             measured excited-state lifetime, ns; blank where none was measured
"""

from __future__ import annotations

import importlib.resources as ir
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

COLUMNS = ("id", "name", "smiles", "structure_image", "E_T", "lambda_S0_T1", "lambda_T1_S0",
           "E_PCplus_PCstar", "E_PCstar_PCminus", "tau0_ns")
TEXT = ("id", "name", "smiles", "structure_image")
REQUIRED = ("E_T", "lambda_S0_T1", "lambda_T1_S0")


@lru_cache(maxsize=1)
def load_photocatalysts_df() -> pd.DataFrame:
    """The packaged photocatalysts.csv as a DataFrame (cached)."""
    with ir.files("webapp.data").joinpath("photocatalysts.csv").open("rb") as f:
        df = pd.read_csv(f, dtype={c: str for c in TEXT}, keep_default_na=False,
                         na_values={c: [""] for c in COLUMNS if c not in TEXT})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"photocatalysts.csv has no column {', '.join(missing)}")
    if df["id"].duplicated().any():
        raise ValueError("photocatalysts.csv lists a photocatalyst twice")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError(f"photocatalysts.csv has a blank {', '.join(REQUIRED)}")
    return df


def _record(row: pd.Series) -> Dict[str, Any]:
    """One row with str for the text columns and float or None for the numbers."""
    out: Dict[str, Any] = {}
    for column in COLUMNS:
        value = row[column]
        if column in TEXT:
            out[column] = str(value)
        else:
            out[column] = None if value is None or math.isnan(float(value)) else float(value)
    return out


def photocatalysts() -> List[Dict[str, Any]]:
    """Every photocatalyst of the library, in table order."""
    return [_record(row) for _, row in load_photocatalysts_df().iterrows()]


def get_photocatalyst(pc_id: str) -> Optional[Dict[str, Any]]:
    """The photocatalyst with this id (e.g. "PC1", "OPC2"), or None."""
    df = load_photocatalysts_df()
    match = df[df["id"] == str(pc_id)]
    return None if match.empty else _record(match.iloc[0])


def structure_image(pc_id: str) -> Optional[bytes]:
    """The PNG drawing of a photocatalyst, or None where there is none."""
    pc = get_photocatalyst(pc_id)
    if pc is None or not pc["structure_image"]:
        return None
    image = ir.files("webapp.data").joinpath("pc_structures", pc["structure_image"])
    return image.read_bytes() if image.is_file() else None
