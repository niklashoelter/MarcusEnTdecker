# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Quencher redox potentials, one table from the two hand-edited ones.

Reads   ../compound_identification.xlsx   (quencher ids)
        ../experimental_data_processed/quencher_redox_literature.csv
        ../experimental_data_processed/quencher_redox_measured.csv
Writes  ../experimental_data_processed/quencher_redox_potentials.csv

    uv run data/experimental_data_scripts/quencher_redox_potentials.py

Inputs and output share the columns quencher_id, E_Ox_vs_SCE_V, E_Ox_bound,
E_Red_vs_SCE_V, E_Red_bound, source, comment. Measured: source `CV`, read by hand from the
voltammograms in `experimental_data_raw/quencher_cyclic_voltammetry/measured data/`;
literature: source a DOI or `estimated`. Each quencher in exactly one file. Bound: `>` on
E_Ox, `<` on E_Red, with a value. Values copied as written, sorted by quencher id.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from compound_identification import QUENCHERS  # noqa: E402

FOLDER = DATA / "experimental_data_processed"
LITERATURE = FOLDER / "quencher_redox_literature.csv"
MEASURED = FOLDER / "quencher_redox_measured.csv"
OUT = FOLDER / "quencher_redox_potentials.csv"

COLUMNS = [
    "quencher_id", "E_Ox_vs_SCE_V", "E_Ox_bound", "E_Red_vs_SCE_V", "E_Red_bound",
    "source", "comment",
]


def read(path):
    """quencher_id -> row, after checking the columns, ids and bounds."""
    if not path.exists():
        sys.exit(f"missing input: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"{path.name}: columns must be {','.join(COLUMNS)}, "
                     f"got {','.join(reader.fieldnames or [])}")
        rows = {}
        for row in reader:
            if not any(row.values()):
                continue                      # blank line left by a spreadsheet editor
            try:
                qid = int(row["quencher_id"])
            except ValueError:
                sys.exit(f"{path.name}: quencher_id {row['quencher_id']!r} is not a number")
            if qid in rows:
                sys.exit(f"{path.name}: Q{qid} appears twice")
            for side, sign in (("Ox", ">"), ("Red", "<")):
                value, bound = row[f"E_{side}_vs_SCE_V"], row[f"E_{side}_bound"]
                if bound not in ("", sign):
                    sys.exit(f"{path.name}: Q{qid}: E_{side}_bound must be '{sign}' or "
                             f"blank, got {bound!r}")
                if bound and not value:
                    sys.exit(f"{path.name}: Q{qid}: a bound on E_{side} without a value")
                if value:
                    try:
                        float(value)
                    except ValueError:
                        sys.exit(f"{path.name}: Q{qid}: E_{side}_vs_SCE_V {value!r} "
                                 f"is not a number")
            row["quencher_id"] = str(qid)
            rows[qid] = row
    return rows


def main():
    literature, measured = read(LITERATURE), read(MEASURED)

    both = sorted(set(literature) & set(measured))
    if both:
        sys.exit(f"quenchers in both {LITERATURE.name} and {MEASURED.name}: {both}; "
                 f"keep each in one file")
    rows = literature | measured
    stray = sorted(set(rows) - set(QUENCHERS))
    if stray:
        sys.exit(f"quenchers not in compound_identification.xlsx: {stray}")
    missing = sorted(set(QUENCHERS) - set(rows))
    if missing:
        sys.exit(f"quenchers in neither {LITERATURE.name} nor {MEASURED.name}: {missing}")

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows[qid] for qid in sorted(rows))
    print(f"{len(literature)} literature + {len(measured)} measured -> "
          f"{len(rows)} quenchers in {OUT.name}")


if __name__ == "__main__":
    main()
