# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
"""
Experimental tables -> one property workbook per compound class.

Reads   ../compound_identification.xlsx   (ids, names, SMILES)
        ../experimental_data_raw/pc_tcspc/lifetimes_tcspc.csv
        ../experimental_data_raw/stern_volmer/plate_reader_filters.csv
        ../experimental_data_raw/quencher_absorption/absorption_extinction.csv
        ../experimental_data_raw/quencher_cyclic_voltammetry/quencher_redox_potentials.csv
Writes  ../experimental_data_processed/photocatalyst_properties_exp.xlsx
        ../experimental_data_processed/quencher_properties_exp.xlsx

Everything per compound that was measured or recorded, in one place. The per-measurement
tables stay in the raw folders beside the instrument files they were derived from; this
only selects from them and lays the result out in blocks (see ../properties_workbook.py).
No value is computed here.

Downstream, `../build_exp_dataset.py` reads these two workbooks and the Stern-Volmer fits,
and nothing else from the experimental side. The quencher workbook is read a second time,
by `../../benchmark/pairs.py`, for the measured redox potentials and their bound flags:
the SET driving force is half a measured potential and half a calculated one, so it is
built where the two are compared rather than carried in the pair table.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from compound_identification import NEW_ID, PHOTOCATALYSTS, QUENCHERS  # noqa: E402
from properties_workbook import Block, Column, field, read_table, write_workbook  # noqa: E402

RAW = DATA / "experimental_data_raw"
OUT_DIR = DATA / "experimental_data_processed"
PC_OUT = OUT_DIR / "photocatalyst_properties_exp.xlsx"
Q_OUT = OUT_DIR / "quencher_properties_exp.xlsx"

IDENTIFICATION = "compound_identification.xlsx"
TAU0 = RAW / "pc_tcspc" / "lifetimes_tcspc.csv"
FILTERS = RAW / "stern_volmer" / "plate_reader_filters.csv"
ABSORPTION = RAW / "quencher_absorption" / "absorption_extinction.csv"
REDOX = RAW / "quencher_cyclic_voltammetry" / "quencher_redox_potentials.csv"


def epsilon_wavelengths(path):
    """The wavelengths absorption_extinction.py actually wrote, read off its header.

    Restating the list here would make a second source for it, and the two would drift
    the first time a filter changes. The extraction script owns the grid; this reads it.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    return [int(h.split("_")[1]) for h in header if h.startswith("epsilon_")]


def src(path, column):
    return f"{path.relative_to(DATA).as_posix()}: {column}"


def identity(header, table, sheet, name, description):
    """A column taken from compound_identification.xlsx by id."""
    return Column(header, description, f"{IDENTIFICATION}: {sheet}/{name}",
                  lambda i: table[i][name])


def photocatalysts():
    tau0 = read_table(TAU0, "pc_id")
    # One row per batch; all batches of a photocatalyst share one filter pair, which
    # stern_volmer_intensities.py checks, so the last row read for an id stands for all.
    filters = read_table(FILTERS, "pc_old_id")

    for name, table in (("lifetimes", tau0), ("filters", filters)):
        if set(table) != set(NEW_ID):
            sys.exit(f"photocatalyst {name} disagree with {IDENTIFICATION}: "
                     f"{sorted(set(table) ^ set(NEW_ID))}")

    def col(header, table, path, name, description):
        return Column(header, description, src(path, name), field(table, name))

    blocks = [
        Block("Identity", [
            Column("new_id", "PC number used downstream and in the paper, 1-27.",
                   f"{IDENTIFICATION}: Photocatalysts/new_id", lambda i: NEW_ID[i]),
            Column("old_id", "PC number during the screen, used by the raw files.",
                   f"{IDENTIFICATION}: Photocatalysts/old_id", lambda i: i),
            identity("compound", PHOTOCATALYSTS, "Photocatalysts", "name", "Compound."),
        ]),
        Block("Stern-Volmer screen", [
            col("excitation / nm", filters, FILTERS, "excitation_nm",
                "Excitation filter centre in the Stern-Volmer screen, from the plate export "
                "(old PC1 and PC3: from the photocatalyst sheet)."),
            col("emission / nm", filters, FILTERS, "emission_nm",
                "Emission filter centre in the Stern-Volmer screen, from the plate export "
                "(old PC1 and PC3: from the photocatalyst sheet)."),
        ]),
        Block("Excited-state lifetime, TCSPC in MeCN under Ar", [
            col("tau0 / ns", tau0, TAU0, "tau0_ns",
                "Unquenched lifetime, single-exponential fit of pc_tcspc/PC<n>.txt."),
            col("tau0_err / ns", tau0, TAU0, "tau0_err_ns", "Standard error of that fit."),
            col("tau0_note", tau0, TAU0, "note", "Deviation from the standard fit window."),
        ]),
    ]
    note = """
        One row per photocatalyst in the screen, sorted by new_id. The raw files use old_id;
        the mapping is data/compound_identification.xlsx. new_id 22 (old 24) is kept here and dropped
        by the theory benchmark: its emission was too weak.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(NEW_ID, key=NEW_ID.get), blocks, note


def quenchers():
    absorption = read_table(ABSORPTION, "quencher_id")
    redox = read_table(REDOX, "quencher_id")

    for name, table in (("absorption", absorption), ("redox", redox)):
        stray = sorted(set(table) - set(QUENCHERS))
        if stray:
            sys.exit(f"{name} rows for quenchers not in {IDENTIFICATION}: {stray}")
    if set(redox) != set(QUENCHERS):
        sys.exit(f"redox table misses quenchers {sorted(set(QUENCHERS) - set(redox))}")

    def col(header, table, path, name, description):
        return Column(header, description, src(path, name), field(table, name))

    epsilon = [col(f"epsilon_{nm} / mM-1 cm-1", absorption, ABSORPTION,
                   f"epsilon_{nm}_mM-1cm-1", f"Molar extinction coefficient at {nm} nm.")
               for nm in epsilon_wavelengths(ABSORPTION)]
    epsilon.append(col("max_read_rsd / %", absorption, ABSORPTION, "max_read_rsd_percent",
                       "Largest relative spread between repeat reads across the 18 "
                       "wavelengths."))

    blocks = [
        Block("Identity", [
            Column("quencher_id", "Quencher number.",
                   f"{IDENTIFICATION}: Quenchers/quencher_id", lambda i: i),
            Column("quencher_name", "Label used throughout, Q<n>.",
                   f"{IDENTIFICATION}: Quenchers/quencher_id", lambda i: f"Q{i}"),
            identity("compound", QUENCHERS, "Quenchers", "name",
                     "Name derived from the SMILES (checked by name-to-structure conversion)."),
            identity("smiles", QUENCHERS, "Quenchers", "smiles", "Structure."),
        ]),
        Block("Redox potentials / V vs SCE", [
            col("E_Ox_vs_SCE_exp / V", redox, REDOX, "E_Ox_vs_SCE_V",
                "Oxidation potential. Where E_Ox_bound is set, a limit, not a value."),
            col("E_Ox_bound", redox, REDOX, "E_Ox_bound",
                "'>': not oxidized within the window; the true value lies above the one "
                "given. Blank: a value."),
            col("E_Red_vs_SCE_exp / V", redox, REDOX, "E_Red_vs_SCE_V",
                "Reduction potential. Where E_Red_bound is set, a limit, not a value."),
            col("E_Red_bound", redox, REDOX, "E_Red_bound",
                "'<': not reduced within the window; the true value lies below the one "
                "given. Blank: a value."),
            col("source", redox, REDOX, "source",
                "CV: measured in this lab. Otherwise the DOI of the paper the potentials "
                "were taken from, or 'estimated' where neither exists."),
            col("comment", redox, REDOX, "comment",
                "For a deposited voltammogram, which descriptor was read (onset, half-wave, "
                "or no wave inside the window); otherwise from the literature record."),
        ]),
        Block("Molar extinction coefficients in MeCN / mM-1 cm-1", epsilon),
        Block("Sample", [
            col("c_stock / mM", absorption, ABSORPTION, "c_stock_mM", "Stock concentration."),
            col("well_position", absorption, ABSORPTION, "well", "Well of the stock plate."),
        ]),
    ]
    note = """
        One row per quencher. The potentials are read from the voltammogram for the 19
        quenchers that have one, and taken from the literature for the others; both are in
        experimental_data_raw/quencher_cyclic_voltammetry/.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(QUENCHERS), blocks, note


def main():
    ids, blocks, note = photocatalysts()
    write_workbook(PC_OUT, "Photocatalysts", ids, blocks, note)
    print(f"{len(ids)} photocatalysts x {sum(len(b.columns) for b in blocks)} columns "
          f"-> {PC_OUT.name}")

    ids, blocks, note = quenchers()
    write_workbook(Q_OUT, "Quenchers", ids, blocks, note)
    print(f"{len(ids)} quenchers x {sum(len(b.columns) for b in blocks)} columns "
          f"-> {Q_OUT.name}")


if __name__ == "__main__":
    main()
