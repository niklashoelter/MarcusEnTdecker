# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas==2.3.3", "openpyxl==3.1.5", "rdkit==2025.9.4"]
# ///
"""
The photocatalyst table of the web application, built from the repository's data.

Reads   ../../data/compound_identification.xlsx                              PC names
        ../../data/computational_data_processed/photocatalyst_properties_calc.xlsx
                                                     E_T, both lambda, excited-state potentials
        ../../data/experimental_data_processed/photocatalyst_properties_exp.xlsx   tau0
        ../../data/computational_data_raw/organophotocatalysts/organophotocatalyst_dft.csv
                                                     OPC names, E_T and both lambda
        ../../data/computational_data_raw/organophotocatalysts/OPC<n>/S0/opt_geometry.xyz
                                                     OPC SMILES
        ../data/pc_structures/<id>.png                which drawings exist
Writes  ../data/photocatalysts.csv

    uv run webapp/scripts/build_photocatalysts.py

One row per photocatalyst of the library: the screened photocatalysts PC<n> by their
new_id, then the organic photocatalysts OPC<n>, which were not screened and are offered
only here. Each quantity has one source, the same one the benchmark's thermokinetic
criterion uses:

    E_T, lambda_S0_T1, lambda_T1_S0     DFT (kcal/mol); lambda_S0_T1 is the S0 surface at the
                                        T1 geometry, lambda_T1_S0 the T1 surface at the S0
                                        geometry
    E_PCplus_PCstar, E_PCstar_PCminus   DFT excited-state potentials (V vs SCE), E(PC+/PC*) and
                                        E(PC*/PC-); none for the OPCs (S0 and T1 only)
    tau0_ns                             measured lifetime (TCSPC, ns); none for the OPCs

The values are copied as the data tables hold them (kcal/mol and V at 2 decimals). The
metal complexes have no SMILES by design; their structures are drawings in
data/pc_structures/, named <id>.png, where one exists. The OPC SMILES are read off the
optimised S0 geometries (bond orders from connectivity, the neutral resonance form).

Not part of the package: webapp/scripts/ has no __init__.py, so it is neither installed
nor imported by the application.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

WEBAPP = Path(__file__).resolve().parent.parent
ROOT = WEBAPP.parent
DATA = ROOT / "data"
IDENTIFICATION = DATA / "compound_identification.xlsx"
CALC = DATA / "computational_data_processed" / "photocatalyst_properties_calc.xlsx"
EXP = DATA / "experimental_data_processed" / "photocatalyst_properties_exp.xlsx"
OPC_DIR = DATA / "computational_data_raw" / "organophotocatalysts"
OPC_TABLE = OPC_DIR / "organophotocatalyst_dft.csv"
IMAGES = WEBAPP / "data" / "pc_structures"
OUT = WEBAPP / "data" / "photocatalysts.csv"

COLUMNS = ["id", "name", "smiles", "structure_image", "E_T", "lambda_S0_T1", "lambda_T1_S0",
           "E_PCplus_PCstar", "E_PCstar_PCminus", "tau0_ns"]
DECIMALS = 2                    # kcal/mol and V

# column of the source table -> column written
CALC_COLUMNS = {"E_T / kcal mol-1": "E_T",
                "lambda_S0_T1 / kcal mol-1": "lambda_S0_T1",
                "lambda_T1_S0 / kcal mol-1": "lambda_T1_S0",
                "E_PCplus_PCstar / V": "E_PCplus_PCstar",
                "E_PCstar_PCminus / V": "E_PCstar_PCminus"}
OPC_COLUMNS = {"E_T / kcal mol-1": "E_T",
               "lambda_S0_T1 / kcal mol-1": "lambda_S0_T1",
               "lambda_T1_S0 / kcal mol-1": "lambda_T1_S0"}
TAU0 = "tau0 / ns"


def read_workbook(path: Path) -> pd.DataFrame:
    """A processed property workbook: block header in row 0, column names in row 1."""
    df = pd.read_excel(path, sheet_name=0, header=1)
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


def smiles_from_geometry(xyz: Path) -> str:
    """Canonical SMILES of a neutral closed-shell molecule from its xyz geometry."""
    mol = Chem.MolFromXYZBlock(xyz.read_text())
    if mol is None:
        raise ValueError(f"{xyz} is not a readable xyz file")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    if any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        # e.g. a thioxanthone assigned as the S+/O- form; take the neutral resonance form
        forms = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL | Chem.ALLOW_CHARGE_SEPARATION)
        neutral = [form for form in forms
                   if not any(atom.GetFormalCharge() for atom in form.GetAtoms())]
        if not neutral:
            raise ValueError(f"{xyz}: no neutral bond assignment")
        mol = Chem.Mol(neutral[0])
        Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def image_for(pc_id: str) -> str:
    return f"{pc_id}.png" if (IMAGES / f"{pc_id}.png").is_file() else ""


def screened() -> list[dict]:
    names = pd.read_excel(IDENTIFICATION, sheet_name="Photocatalysts").set_index("new_id")["name"]
    calc = read_workbook(CALC).set_index("new_id")
    exp = read_workbook(EXP).set_index("new_id")
    for label, table in (("calc", calc), ("exp", exp)):
        if list(table.index) != list(names.index):
            raise ValueError(f"the {label} workbook does not list the PCs of {IDENTIFICATION.name}")
        clash = table["compound"] != names
        if clash.any():
            raise ValueError(f"the {label} workbook names PC{list(table.index[clash])} differently")

    rows = []
    for new_id, name in names.items():
        pc_id = f"PC{int(new_id)}"
        row = {"id": pc_id, "name": name, "smiles": "", "structure_image": image_for(pc_id)}
        row |= {out: round(float(calc.at[new_id, col]), DECIMALS) for col, out in CALC_COLUMNS.items()}
        tau0 = exp.at[new_id, TAU0]
        row["tau0_ns"] = float(tau0) if pd.notna(tau0) else None
        rows.append(row)
    return rows


def organic() -> list[dict]:
    table = pd.read_csv(OPC_TABLE).sort_values("opc_id")
    rows = []
    for _, opc in table.iterrows():
        pc_id = f"OPC{int(opc['opc_id'])}"
        smiles = smiles_from_geometry(OPC_DIR / pc_id / "S0" / "opt_geometry.xyz")
        atoms = Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms()
        if atoms != int(opc["n_atoms"]):
            raise ValueError(f"{pc_id}: {atoms} atoms in the SMILES, {opc['n_atoms']} in the table")
        row = {"id": pc_id, "name": opc["name"], "smiles": smiles,
               "structure_image": image_for(pc_id)}
        row |= {out: round(float(opc[col]), DECIMALS) for col, out in OPC_COLUMNS.items()}
        row |= {"E_PCplus_PCstar": None, "E_PCstar_PCminus": None, "tau0_ns": None}
        rows.append(row)
    return rows


def main() -> int:
    RDLogger.DisableLog("rdApp.*")
    rows = screened() + organic()
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    n_pc = sum(r["id"].startswith("PC") for r in rows)
    missing = [r["id"] for r in rows if not r["smiles"] and not r["structure_image"]]
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {n_pc} PC + {len(rows) - n_pc} OPC")
    if missing:
        print(f"no structure drawing or SMILES: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
