# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""
Compound records -> identity and experimental redox potentials.

Reads   ../experimental_data_raw/compound_records/properties_quenchers.xlsx
        ../experimental_data_raw/compound_records/properties_photocatalysts.xlsx
Writes  ../experimental_data_processed/compound_records_quenchers.csv
        ../experimental_data_processed/compound_records_photocatalysts.csv

These two spreadsheets are the only inputs in the package that are records rather than
instrument output: someone wrote them by hand. They are read for exactly what cannot be
derived from a measurement or a calculation -- the identity of each compound, and the
experimental redox potentials.

WHAT IS DELIBERATELY NOT TAKEN
    Both spreadsheets carry columns that this package computes for itself, and taking
    the spreadsheet's copy would put a second, unverifiable version of the same
    quantity into circulation:

      * the 18 epsilon columns in the quencher sheet     -> absorption_extinction.py
      * `t0 / ns` and `t0_err / ns` in the photocatalyst sheet -> lifetimes_tau0.py

    Both were checked against the derived values before being dropped: the extinction
    coefficients agree to the six-significant-figure rounding on all 864, and the
    lifetimes agree exactly on 26 of the 27 screened photocatalysts, PC4 excepted.

BOUNDS ARE NOT MEASUREMENTS
    Six oxidation and several reduction potentials are recorded as limits rather than
    values -- `> 2.0`, `< -2.10` -- meaning the compound was not oxidized or reduced
    within the accessible window. That is a one-sided observation, exactly like a
    Stern-Volmer non-quencher.

    The number and the fact that it is a bound are stored in separate columns, so no
    downstream consumer can read `2.0` without also being able to see that the true
    potential is somewhere above it. Earlier code in this project parsed these strings
    to their face value with no flag, which silently converts "we could not oxidize
    this" into "this oxidizes at 2.0 V" -- and since the SET channel is ruled out by
    comparing against these potentials, that error makes SET look more feasible than
    the data supports.

EXPERIMENTAL VERSUS COMPUTED
    Everything here is the experimental side, suffixed `_exp`. The computed potentials
    are a separate quantity with a separate provenance and live in
    computational_data_processed/quencher_dft.csv as `E_Ox_vs_SCE_calc` and
    `E_Red_vs_SCE_calc`. They are never merged into one column: a measured potential
    and a DFT potential are not interchangeable, and which one an analysis used is
    part of what it has to report.

    Where each experimental value came from -- measured here, or read out of another
    paper's supporting information -- is not yet recorded. Those columns are to be
    added; see the processed README.
"""

import csv
import re
import sys
from pathlib import Path

import openpyxl

RAW = Path(__file__).resolve().parent.parent / "experimental_data_raw" / "compound_records"
OUT_DIR = Path(__file__).resolve().parent.parent / "experimental_data_processed"

QUENCHERS_IN = RAW / "properties_quenchers.xlsx"
PHOTOCATALYSTS_IN = RAW / "properties_photocatalysts.xlsx"
QUENCHERS_OUT = OUT_DIR / "compound_records_quenchers.csv"
PHOTOCATALYSTS_OUT = OUT_DIR / "compound_records_photocatalysts.csv"

SIGFIGS = 6

# The bound strings were typed by hand and come in four shapes across the 19 of them:
# "> 2.0", "<-2.5" with no space, "< -2.10 " with a trailing one, and "< -2,0" with a
# German decimal comma. All four are accepted; anything else is an error rather than a
# silent skip, because a potential that fails to parse would otherwise become a
# missing value and quietly drop its pairs from the analysis.
BOUND = re.compile(r"^\s*(?P<op>[<>]=?)\s*(?P<value>-?\d+(?:[.,]\d+)?)\s*$")

QUENCHER_COLUMNS = [
    "quencher_id", "quencher_name", "name", "smiles",
    "E_Ox_vs_SCE_exp", "E_Ox_bound", "E_Red_vs_SCE_exp", "E_Red_bound",
]
PHOTOCATALYST_COLUMNS = [
    "pc_id", "pc_name", "name", "excitation_nm", "emission_nm",
]


def sig(x, n=SIGFIGS):
    if x is None:
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def parse_potential(cell):
    """-> (value, bound) where bound is '', 'lower' or 'upper'.

    A bare number is a measurement and carries no bound. `> 2.0` means the compound was
    not oxidized below 2.0 V, so 2.0 is a LOWER bound on the true potential; `< -2.1`
    means it was not reduced above -2.1 V, an UPPER bound. The value is kept because it
    is the only quantitative information there is, but never without the flag.
    """
    if cell is None or (isinstance(cell, str) and not cell.strip()):
        return None, ""
    if isinstance(cell, (int, float)):
        return float(cell), ""

    match = BOUND.match(str(cell))
    if match:
        value = float(match.group("value").replace(",", "."))
        return value, "lower" if match.group("op").startswith(">") else "upper"

    raise ValueError(f"cannot parse potential {cell!r}")


def headers(sheet):
    """Column index by header name, 1-based, blank headers ignored."""
    found = {}
    for column in range(1, sheet.max_column + 1):
        name = sheet.cell(1, column).value
        if name is not None and str(name).strip():
            found[str(name).strip()] = column
    return found


def read_sheet(path):
    if not path.exists():
        sys.exit(f"missing input: {path}")
    sheet = openpyxl.load_workbook(path, data_only=True).active
    return sheet, headers(sheet)


def quenchers():
    sheet, column = read_sheet(QUENCHERS_IN)
    rows, bounded = [], []

    for r in range(2, sheet.max_row + 1):
        qid = sheet.cell(r, column["quencher_id"]).value
        if qid is None:
            continue

        ox, ox_bound = parse_potential(sheet.cell(r, column["E_Ox_vs_SCE"]).value)
        red, red_bound = parse_potential(sheet.cell(r, column["E_Red_vs_SCE"]).value)
        if ox_bound or red_bound:
            bounded.append(int(qid))

        rows.append({
            "quencher_id": int(qid),
            "quencher_name": sheet.cell(r, column["quencher_name"]).value,
            "name": sheet.cell(r, column["name"]).value,
            "smiles": sheet.cell(r, column["smiles"]).value,
            "E_Ox_vs_SCE_exp": sig(ox),
            "E_Ox_bound": ox_bound,
            "E_Red_vs_SCE_exp": sig(red),
            "E_Red_bound": red_bound,
        })

    rows.sort(key=lambda row: row["quencher_id"])
    return rows, bounded


def photocatalysts():
    sheet, column = read_sheet(PHOTOCATALYSTS_IN)

    # The chemical name sits in an unlabelled column between PC_name and excitation.
    named = column.get("PC_name")
    name_column = next((c for c in range(named + 1, sheet.max_column + 1)
                        if sheet.cell(1, c).value is None
                        and isinstance(sheet.cell(2, c).value, str)), None)

    rows = []
    for r in range(2, sheet.max_row + 1):
        pc_id = sheet.cell(r, column["PC ID"]).value
        if pc_id is None:
            continue
        rows.append({
            "pc_id": int(pc_id),
            "pc_name": sheet.cell(r, named).value,
            "name": sheet.cell(r, name_column).value if name_column else "",
            "excitation_nm": sheet.cell(r, column["excitation"]).value or "",
            "emission_nm": sheet.cell(r, column["emission"]).value or "",
        })

    rows.sort(key=lambda row: row["pc_id"])
    return rows


def write(path, columns, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{c: ("" if row.get(c) is None else row.get(c)) for c in columns}
                          for row in rows])


def main():
    quencher_rows, bounded = quenchers()
    write(QUENCHERS_OUT, QUENCHER_COLUMNS, quencher_rows)

    pc_rows = photocatalysts()
    write(PHOTOCATALYSTS_OUT, PHOTOCATALYST_COLUMNS, pc_rows)

    print(f"{len(quencher_rows)} quenchers -> {QUENCHERS_OUT.name}")
    missing_smiles = [r["quencher_id"] for r in quencher_rows if not r["smiles"]]
    print(f"  with SMILES: {len(quencher_rows) - len(missing_smiles)}"
          + (f", MISSING for {missing_smiles}" if missing_smiles else ""))

    for field, label in (("E_Ox_vs_SCE_exp", "oxidation"), ("E_Red_vs_SCE_exp", "reduction")):
        have = sum(1 for r in quencher_rows if r[field] != "")
        limits = sum(1 for r in quencher_rows if r[field.replace("_vs_SCE_exp", "_bound")])
        print(f"  experimental {label}: {have} values, of which {limits} are bounds")
    print(f"  quenchers with at least one bound: {sorted(set(bounded))}")

    print(f"\n{len(pc_rows)} photocatalysts -> {PHOTOCATALYSTS_OUT.name}")


if __name__ == "__main__":
    main()
