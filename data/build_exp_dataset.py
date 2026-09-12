# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
"""
The processed experimental data -> one row per photocatalyst-quencher pair: what was
measured, the labels read off it, and nothing else.

Reads   experimental_data_processed/stern_volmer_fits.csv
        experimental_data_processed/photocatalyst_properties_exp.xlsx
        experimental_data_processed/quencher_properties_exp.xlsx
Writes  exp_dataset.csv     the table ../benchmark/ starts from
        exp_dataset.xlsx    the same table in the property-workbook layout

    uv run data/build_exp_dataset.py

THE LABEL IS SET ON K_SV, NOT ON k_q
    k_q = K_SV / tau_0, both measured here. Both labels come from stern_volmer_fits.py
    unchanged: `quenches`, a ladder of (K_SV, r2) rungs where weaker quenching demands a
    straighter line, and `quenches_strict`, the published K_SV >= 25 and r2 >= 0.90.
    K_SV is the observable, and dividing it by a photocatalyst-specific tau_0 would move
    the detection floor with the photocatalyst, so no cutoff is set on k_q.

    A pair that does not quench is left-censored, not missing: `censored = 1`, its fitted
    K_SV kept, and `k_q_floor = 10 / tau_0` the smallest rate it could have shown.

NO CALCULATED COLUMN, AND NO DRIVING FORCE
    A computed property is a property of a compound and stays one row per compound in
    computational_data_processed/. dG_EnT and dG_SET are differences across a pair, so
    they are built by whoever compares the two -- ../benchmark/pairs.py joins the
    calculated workbooks onto this table and derives them there. dG_SET shows why the rule
    is worth stating: half of it is a measured quencher potential and half a calculated
    photocatalyst one, so a column of it would sit among the measurements without being
    one. The measured potentials stay in quencher_properties_exp.xlsx with their bound
    flags.

PHOTOCATALYST NUMBERING
    Each row carries `pc_new_id` (1-27, used downstream and in the paper), `pc_old_id`
    (what the raw files use) and `pc_name` = PC<new_id>; the mapping is
    compound_identification.py. All 27 are here, PC22 (old PC24) included -- its emission
    was too weak and the theory benchmark drops it, this table does not.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))

from openpyxl import load_workbook  # noqa: E402

from compound_identification import NEW_ID  # noqa: E402
from properties_workbook import Block, Column, write_workbook  # noqa: E402

EXPERIMENTAL = DATA / "experimental_data_processed"
OUT_CSV = DATA / "exp_dataset.csv"
OUT_XLSX = DATA / "exp_dataset.xlsx"

# Row 1 of every property workbook names the column blocks; the headers are on row 2.
WORKBOOK_HEADER_ROW = 2

# The K_SV a pair has to clear to count as detected, and so the censoring limit. The
# label's lowest rung admits smaller K_SV on a near-perfect line, but 10 is the slope
# below which this plate reader cannot see a drop at all.
K_SV_FLOOR = 10.0

SIGFIGS = 6

# (block, [(column, description, source)]). The order is the order of the table.
LAYOUT = [
    ("Identity", [
        ("pc_new_id", "Photocatalyst number used downstream and in the paper, 1-27.",
         "photocatalyst_properties_exp.xlsx: new_id"),
        ("pc_old_id", "Photocatalyst number during the screen, used by the raw files.",
         "photocatalyst_properties_exp.xlsx: old_id"),
        ("pc_name", "PC<pc_new_id>.", "computed here"),
        ("quencher_id", "Quencher number.", "quencher_properties_exp.xlsx"),
        ("quencher_name", "Q<n>.", "quencher_properties_exp.xlsx"),
    ]),
    ("Measurement", [
        ("K_SV / M-1", "Stern-Volmer constant, the fitted slope.", "stern_volmer_fits.csv: K_SV"),
        ("tau0 / ns", "Unquenched lifetime of the photocatalyst.",
         "photocatalyst_properties_exp.xlsx"),
        ("k_q / M-1 s-1", "Quenching rate constant, K_SV / tau0.", "computed here"),
        ("k_q_floor / M-1 s-1", "Smallest k_q the pair could have shown, 10 / tau0.",
         "computed here"),
    ]),
    ("Ground truth", [
        ("quenches", "1: the fit clears one rung of the (K_SV, r2) ladder. The label.",
         "stern_volmer_fits.csv: quenches"),
        ("quenches_strict", "1: K_SV >= 25 and r2 >= 0.90, the published criterion.",
         "stern_volmer_fits.csv: quenching_observed"),
        ("censored", "1 where quenches = 0: K_SV lies somewhere below the floor.",
         "computed here"),
    ]),
]
COLUMNS = [name for _, columns in LAYOUT for name, _, _ in columns]


def sig(x, n=SIGFIGS):
    if x in (None, ""):
        return ""
    return 0.0 if float(x) == 0 else float(f"{float(x):.{n}g}")


def load(path, key):
    """A property workbook as {key: row}, blanks as "". A missing file is fatal."""
    if not path.exists():
        sys.exit(f"missing input: {path}\nrun the data stage first: uv run reproduce.py data")
    sheet = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = sheet.iter_rows(min_row=WORKBOOK_HEADER_ROW, values_only=True)
    header = next(rows)
    table = {}
    for values in rows:
        row = {h: ("" if v is None else v) for h, v in zip(header, values) if h}
        if row.get(key) not in ("", None):
            table[int(row[key])] = row
    return table


def number(row, field):
    """A numeric field, or None when absent or blank."""
    if row is None:
        return None
    value = row.get(field, "")
    return float(value) if value not in ("", None) else None


def main():
    fits_path = EXPERIMENTAL / "stern_volmer_fits.csv"
    if not fits_path.exists():
        sys.exit(f"missing input: {fits_path}\nrun the data stage first")

    pc_exp = load(EXPERIMENTAL / "photocatalyst_properties_exp.xlsx", "old_id")
    q_exp = load(EXPERIMENTAL / "quencher_properties_exp.xlsx", "quencher_id")

    rows = []
    with fits_path.open(encoding="utf-8", newline="") as fh:
        for fit in csv.DictReader(fh):
            pc_id, q_id = int(fit["pc_old_id"]), int(fit["quencher_id"])
            if pc_id not in NEW_ID:
                sys.exit(f"Stern-Volmer fits for old PC{pc_id}, which is not in the data")
            pce, qe = pc_exp.get(pc_id), q_exp.get(q_id)

            tau0 = number(pce, "tau0 / ns")
            k_sv = number(fit, "K_SV")
            # tau0 is in ns; k_q comes out in M-1 s-1.
            k_q = k_sv / (tau0 * 1e-9) if (k_sv is not None and tau0) else None
            floor = K_SV_FLOOR / (tau0 * 1e-9) if tau0 else None
            quenches = fit["quenches"] == "1"

            rows.append({
                "pc_new_id": NEW_ID[pc_id],
                "pc_old_id": pc_id,
                "pc_name": f"PC{NEW_ID[pc_id]}",
                "quencher_id": q_id,
                "quencher_name": (qe or {}).get("quencher_name", ""),
                "K_SV / M-1": sig(k_sv),
                "tau0 / ns": sig(tau0),
                "k_q / M-1 s-1": sig(k_q),
                "k_q_floor / M-1 s-1": sig(floor),
                "quenches": int(quenches),
                "quenches_strict": int(fit["quenching_observed"] == "True"),
                "censored": int(not quenches),
            })

    rows.sort(key=lambda row: (row["pc_new_id"], row["quencher_id"]))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    blocks = [Block(title, [Column(name, description, source,
                                   lambda i, name=name: rows[i][name] if rows[i][name] != "" else None)
                            for name, description, source in columns])
              for title, columns in LAYOUT]
    write_workbook(OUT_XLSX, "Pairs", list(range(len(rows))), blocks, """
        One row per photocatalyst-quencher pair, sorted by pc_new_id then quencher_id. Built
        by data/build_exp_dataset.py; exp_dataset.csv is the same table. Measured values and
        the labels only: nothing calculated is here, and the driving forces are derived by
        ../benchmark/pairs.py. PC numbers are new ids (pc_old_id: the raw files'). PC22 (old
        PC24) is included; its emission was too weak, and the theory benchmark drops it.
    """)

    n_pc = len({r["pc_new_id"] for r in rows})
    n_q = len({r["quencher_id"] for r in rows})
    print(f"{len(rows)} pairs ({n_pc} photocatalysts x {n_q} quenchers) -> "
          f"{OUT_CSV.name}, {OUT_XLSX.name}")
    print(f"  quenches {sum(r['quenches'] for r in rows)}, "
          f"quenches_strict {sum(r['quenches_strict'] for r in rows)}, "
          f"censored {sum(r['censored'] for r in rows)}")
    for name in COLUMNS:
        missing = sum(1 for r in rows if r[name] == "")
        if missing:
            print(f"  {name}: blank for {missing} pairs")


if __name__ == "__main__":
    main()
