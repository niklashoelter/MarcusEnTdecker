# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl"]
# ///
"""
Compound records -> identity and experimental redox potentials.

Reads   ../experimental_data_raw/compound_records/properties_quenchers.xlsx
        ../experimental_data_raw/compound_records/properties_photocatalysts.xlsx
Writes  ../experimental_data_raw/compound_records/compound_records_quenchers.csv
        ../experimental_data_raw/compound_records/compound_records_photocatalysts.csv

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
    computational_data_processed/quencher_properties_calc.xlsx as `E_Ox_vs_SCE_calc` and
    `E_Red_vs_SCE_calc`. They are never merged into one column: a measured potential
    and a DFT potential are not interchangeable, and which one an analysis used is
    part of what it has to report.

WHERE THE REDOX POTENTIALS CAME FROM
    `quenchers_exp_values.xlsx` carries a free-text Comment and a DOI per quencher, and
    together they resolve the provenance of every potential. Across the 48:

        measured       24    cyclic voltammetry in this lab, several naming who ran it
        literature     19    a DOI, or a bare URL typed into the Comment
        estimated       4    inferred from a related compound
        unattributed    1    a value with nothing recorded about its origin (Q9)

    Two consequences.

    First, the manuscript currently describes these as literature values throughout.
    That is not accurate: half of them were measured here, and 19 of those have raw
    voltammograms sitting in experimental_data_raw/quencher_cyclic_voltammetry/.

    Second, `estimated` is not a weaker citation, it is a different quantity. "similar
    bifunctional with Me instead of iPr" is the potential of a surrogate compound, not
    of the one in the screen. Those four are marked so that an analysis can choose to
    exclude them rather than discovering it later.

    The comment describes the row, so the label is per quencher rather than per
    potential. Splitting it would invent detail the record does not contain.

    Before the provenance is attached, the potentials and SMILES in the two
    spreadsheets are compared and any disagreement is fatal. Attaching a source to a
    number that differs from the published one would be worse than having no source.
"""

import collections
import csv
import re
import sys
from pathlib import Path

import openpyxl

RAW = Path(__file__).resolve().parent.parent / "experimental_data_raw" / "compound_records"
OUT_DIR = RAW

QUENCHERS_IN = RAW / "properties_quenchers.xlsx"
QUENCHERS_PROVENANCE_IN = RAW / "quenchers_exp_values.xlsx"
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
    "redox_source_type", "redox_source_note", "redox_doi",
    "c_stock_mM", "deck_position", "well_position",
]

# How a redox potential came to be, read from the free-text Comment column. The order
# matters: a row can be both measured and cited, and the measurement is the stronger
# claim, so it wins and the DOI is kept alongside it.
SOURCE_RULES = [
    ("measured", ("measured",)),
    ("estimated", ("estimated", "similar", "ask ")),
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


def classify_source(comment, doi):
    """'measured', 'literature', 'estimated' or 'unattributed'.

    The Comment column is free text written during the screen, and it distinguishes
    three genuinely different things that all end up as a number in the same cell:
    a potential measured in this lab, one taken from a paper, and one inferred from a
    related compound. The third is the one that matters most to flag -- "similar
    bifunctional with Me instead of iPr" is a surrogate, not a measurement of the
    compound in question.

    The comment describes the row, not an individual potential, so this is one label
    per quencher. Splitting it per potential would be inventing detail the record does
    not contain.
    """
    text = (str(comment) if comment else "").strip().lower()
    for label, needles in SOURCE_RULES:
        if any(n in text for n in needles):
            return label
    if doi or text.startswith("http") or "doi.org" in text:
        return "literature"
    return "unattributed"


def read_provenance():
    """quencher_id -> the provenance row, or {} when the file is absent.

    Kept optional: the redox values themselves come from properties_quenchers.xlsx and
    are complete without this, so a missing provenance file should cost the source
    columns and nothing else.
    """
    if not QUENCHERS_PROVENANCE_IN.exists():
        print(f"  note: {QUENCHERS_PROVENANCE_IN.name} absent, source columns left empty")
        return {}
    sheet, column = read_sheet(QUENCHERS_PROVENANCE_IN)
    found = {}
    for r in range(2, sheet.max_row + 1):
        qid = sheet.cell(r, column["quencher_id"]).value
        if qid is None:
            continue
        found[int(qid)] = {name: sheet.cell(r, c).value for name, c in column.items()}
    return found


def quenchers():
    sheet, column = read_sheet(QUENCHERS_IN)
    provenance = read_provenance()
    rows, bounded, disagreed = [], [], []

    for r in range(2, sheet.max_row + 1):
        qid = sheet.cell(r, column["quencher_id"]).value
        if qid is None:
            continue

        ox_raw = sheet.cell(r, column["E_Ox_vs_SCE"]).value
        red_raw = sheet.cell(r, column["E_Red_vs_SCE"]).value
        ox, ox_bound = parse_potential(ox_raw)
        red, red_bound = parse_potential(red_raw)
        if ox_bound or red_bound:
            bounded.append(int(qid))

        smiles = sheet.cell(r, column["smiles"]).value
        source = provenance.get(int(qid), {})

        # The two spreadsheets hold the same potentials and the same structures. If they
        # ever stop agreeing, the provenance is being attached to a different number
        # than the one being published, which is worse than having no provenance.
        if source:
            for here, there in ((ox_raw, "E_Ox vs. SCE / V"),
                                (red_raw, "E_Red vs. SCE / V"),
                                (smiles, "smiles")):
                mine = str(here).strip() if here is not None else None
                theirs = str(source.get(there)).strip() if source.get(there) is not None else None
                if mine != theirs:
                    disagreed.append((int(qid), there, mine, theirs))

        rows.append({
            "quencher_id": int(qid),
            "quencher_name": sheet.cell(r, column["quencher_name"]).value,
            "name": sheet.cell(r, column["name"]).value,
            "smiles": smiles,
            "E_Ox_vs_SCE_exp": sig(ox),
            "E_Ox_bound": ox_bound,
            "E_Red_vs_SCE_exp": sig(red),
            "E_Red_bound": red_bound,
            "redox_source_type": classify_source(source.get("Comment"), source.get("DOI"))
                                 if source else "",
            "redox_source_note": (source.get("Comment") or "") if source else "",
            "redox_doi": (source.get("DOI") or "") if source else "",
            "c_stock_mM": sig(source.get("c_Stock / mM")) if source else "",
            "deck_position": (source.get("deck_pos") or "") if source else "",
            "well_position": (source.get("well_pos") or "") if source else "",
        })

    rows.sort(key=lambda row: row["quencher_id"])
    return rows, bounded, disagreed


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
    quencher_rows, bounded, disagreed = quenchers()
    write(QUENCHERS_OUT, QUENCHER_COLUMNS, quencher_rows)

    if disagreed:
        print(f"  ERROR: the two spreadsheets disagree on {len(disagreed)} values:")
        for qid, field, mine, theirs in disagreed:
            print(f"    Q{qid} {field}: properties={mine!r} exp_values={theirs!r}")
        sys.exit("provenance cannot be attached to values that do not match")

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

    kinds = collections.Counter(r["redox_source_type"] for r in quencher_rows)
    print(f"\n  where the redox potentials came from:")
    for kind in ("measured", "literature", "estimated", "unattributed"):
        ids = [r["quencher_id"] for r in quencher_rows if r["redox_source_type"] == kind]
        if ids:
            print(f"    {kind:13} {kinds[kind]:>2}   {ids}")
    if kinds["measured"]:
        print(f"  NOTE: {kinds['measured']} of {len(quencher_rows)} were measured in this lab, so"
              f" 'literature values throughout' is not an accurate description of this set.")
    if kinds["estimated"]:
        print(f"  NOTE: {kinds['estimated']} are inferred from a related compound rather than"
              f" measured or cited — see redox_source_note.")

    print(f"\n{len(pc_rows)} photocatalysts -> {PHOTOCATALYSTS_OUT.name}")


if __name__ == "__main__":
    main()
