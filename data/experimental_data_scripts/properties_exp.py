# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl==3.1.5"]
# ///
"""
Measured properties of the photocatalysts and quenchers, one workbook per compound class.

Reads   ../compound_identification.xlsx   (ids, names, SMILES)
        ../experimental_data_raw/pc_tcspc/lifetimes_tcspc.csv
        ../experimental_data_raw/stern_volmer/plate_reader_filters.csv
        ../experimental_data_raw/quencher_absorption/absorption_extinction.csv
        ../experimental_data_raw/quencher_cyclic_voltammetry/quencher_redox_potentials.csv
Writes  ../experimental_data_processed/photocatalyst_properties_exp.xlsx
        ../experimental_data_processed/quencher_properties_exp.xlsx

    uv run data/experimental_data_scripts/properties_exp.py

Values copied from the raw-folder tables (layout: ../properties_workbook.py); redox
potentials rounded to 2 decimals.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from compound_identification import PHOTOCATALYSTS, QUENCHERS  # noqa: E402
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
DECIMALS = 2                 # for V


def epsilon_wavelengths(path):
    """Wavelengths of the epsilon columns in absorption_extinction.csv."""
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
    tau0 = read_table(TAU0, "pc_new_id")
    # One row per batch, one filter pair per photocatalyst; the last row per id is used.
    filters = read_table(FILTERS, "pc_new_id")

    for name, table in (("lifetimes", tau0), ("filters", filters)):
        if set(table) != set(PHOTOCATALYSTS):
            sys.exit(f"photocatalyst {name} disagree with {IDENTIFICATION}: "
                     f"{sorted(set(table) ^ set(PHOTOCATALYSTS))}")

    def col(header, table, path, name, description):
        return Column(header, description, src(path, name), field(table, name))

    blocks = [
        Block("Identity", [
            Column("new_id", "PC number, 1-27.",
                   f"{IDENTIFICATION}: Photocatalysts/new_id", lambda i: i),
            identity("compound", PHOTOCATALYSTS, "Photocatalysts", "name", "Compound."),
        ]),
        Block("Stern-Volmer screen", [
            col("excitation / nm", filters, FILTERS, "excitation_nm",
                "Excitation filter centre in the Stern-Volmer screen, from the plate export "
                "(PC1 and PC3: from the photocatalyst sheet)."),
            col("emission / nm", filters, FILTERS, "emission_nm",
                "Emission filter centre in the Stern-Volmer screen, from the plate export "
                "(PC1 and PC3: from the photocatalyst sheet)."),
        ]),
        Block("Excited-state lifetime, TCSPC in MeCN under Ar", [
            col("tau0 / ns", tau0, TAU0, "tau0_ns",
                "Unquenched lifetime, single-exponential fit of the TCSPC trace."),
            col("tau0_err / ns", tau0, TAU0, "tau0_err_ns", "Standard error of that fit."),
            col("tau0_note", tau0, TAU0, "note", "Deviation from the standard fit window."),
        ]),
    ]
    note = """
        One row per photocatalyst in the screen, sorted by new_id. PC22 is kept here and
        excluded by the theory benchmark (see SI).
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(PHOTOCATALYSTS), blocks, note


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

    def volts(header, name, description):
        """A redox-potential column, to DECIMALS."""
        value = field(redox, name)
        return Column(header, description, src(REDOX, name),
                      lambda i: None if value(i) is None else round(value(i), DECIMALS))

    epsilon =[col(f"epsilon_{nm} / mM-1 cm-1", absorption, ABSORPTION,
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
                     "Name derived from the SMILES."),
            identity("smiles", QUENCHERS, "Quenchers", "smiles", "Structure."),
        ]),
        Block("Redox potentials / V vs SCE", [
            volts("E_Ox_vs_SCE_exp / V", "E_Ox_vs_SCE_V",
                  "Oxidation potential, or a limit where E_Ox_bound is set."),
            col("E_Ox_bound", redox, REDOX, "E_Ox_bound",
                "'>': no oxidation within the window, the potential lies above the value "
                "given. Blank: a measured value."),
            volts("E_Red_vs_SCE_exp / V", "E_Red_vs_SCE_V",
                  "Reduction potential, or a limit where E_Red_bound is set."),
            col("E_Red_bound", redox, REDOX, "E_Red_bound",
                "'<': no reduction within the window, the potential lies below the value "
                "given. Blank: a measured value."),
            col("source", redox, REDOX, "source",
                "CV: measured in this lab. Otherwise the DOI of the literature source, "
                "or 'estimated'."),
            col("comment", redox, REDOX, "comment",
                "Free-text note from quencher_redox_measured.csv or "
                "quencher_redox_literature.csv."),
        ]),
        Block("Molar extinction coefficients in MeCN / mM-1 cm-1", epsilon),
        Block("Sample", [
            col("c_stock / mM", absorption, ABSORPTION, "c_stock_mM", "Stock concentration."),
            col("well_position", absorption, ABSORPTION, "well", "Well of the stock plate."),
        ]),
    ]
    note = """
        One row per quencher. The potentials are read by hand from the voltammograms or
        taken from the literature, and entered in quencher_redox_measured.csv or
        quencher_redox_literature.csv in experimental_data_raw/quencher_cyclic_voltammetry/.
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
