# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
"""
Experimental tables -> one property workbook per compound class.

Reads   ../experimental_data_raw/compound_records/compound_records_photocatalysts.csv
        ../experimental_data_raw/compound_records/compound_records_quenchers.csv
        ../experimental_data_raw/pc_tcspc/lifetimes_tau0.csv
        ../experimental_data_raw/quencher_absorption/absorption_extinction.csv
        ../experimental_data_raw/quencher_cyclic_voltammetry/cyclic_voltammetry_redox.csv
        ../experimental_data_processed/stern_volmer_fits.csv    (screen membership only)
Writes  ../experimental_data_processed/photocatalyst_properties_exp.xlsx
        ../experimental_data_processed/quencher_properties_exp.xlsx

Everything per compound that was measured or recorded, in one place. The per-measurement
tables stay in the raw folders beside the instrument files they were derived from; this
only selects from them and lays the result out in blocks (see ../properties_workbook.py).
No value is computed here.

Downstream, `dataset/build_dataset.py` reads these two workbooks and the Stern-Volmer fits,
and nothing else from the experimental side.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from properties_workbook import Block, Column, field, read_table, write_workbook  # noqa: E402

RAW = DATA / "experimental_data_raw"
OUT_DIR = DATA / "experimental_data_processed"
PC_OUT = OUT_DIR / "photocatalyst_properties_exp.xlsx"
Q_OUT = OUT_DIR / "quencher_properties_exp.xlsx"

PC_RECORDS = RAW / "compound_records" / "compound_records_photocatalysts.csv"
Q_RECORDS = RAW / "compound_records" / "compound_records_quenchers.csv"
TAU0 = RAW / "pc_tcspc" / "lifetimes_tau0.csv"
ABSORPTION = RAW / "quencher_absorption" / "absorption_extinction.csv"
CV = RAW / "quencher_cyclic_voltammetry" / "cyclic_voltammetry_redox.csv"
FITS = OUT_DIR / "stern_volmer_fits.csv"

EPSILON_NM = [380, 390, 400, 405, 415, 425, 450, 475, 490, 495, 500,
              510, 520, 580, 590, 600, 625, 645]


def src(path, column):
    return f"{path.relative_to(DATA).as_posix()}: {column}"


def photocatalysts():
    records = read_table(PC_RECORDS, "pc_id")
    tau0 = read_table(TAU0, "pc_id")
    screened = screened_photocatalysts()

    stray = sorted(set(tau0) - set(records))
    if stray:
        sys.exit(f"lifetimes for photocatalysts with no record: {stray}")

    def col(header, table, path, name, description):
        return Column(header, description, src(path, name), field(table, name))

    blocks = [
        Block("Identity", [
            col("pc_id", records, PC_RECORDS, "pc_id", "Photocatalyst number."),
            col("pc_name", records, PC_RECORDS, "pc_name", "Label used throughout, PC<n>."),
            col("compound", records, PC_RECORDS, "name", "Compound, as recorded."),
        ]),
        Block("Stern-Volmer screen", [
            Column("in_screen",
                   "1 if the photocatalyst has Stern-Volmer fits against the quencher set, "
                   "0 if not.",
                   src(FITS, "pc_id"), lambda i: int(i in screened)),
            col("excitation / nm", records, PC_RECORDS, "excitation_nm",
                "Excitation wavelength used in the screen."),
            col("emission / nm", records, PC_RECORDS, "emission_nm",
                "Emission wavelength read in the screen."),
        ]),
        Block("Excited-state lifetime, TCSPC in MeCN under Ar", [
            col("tau0 / ns", tau0, TAU0, "tau0_ns",
                "Unquenched lifetime: the longest of the usable measurements, because "
                "residual oxygen can only shorten it."),
            col("tau0_std / ns", tau0, TAU0, "tau0_std_ns",
                "Spread across repeats; with one measurement, that fit's standard error."),
            col("tau0_rsd", tau0, TAU0, "tau0_rsd", "tau0_std / tau0."),
            col("n_measurements", tau0, TAU0, "n_measurements",
                "Usable traces behind tau0."),
            col("n_traces_rejected", tau0, TAU0, "n_traces_rejected",
                "Traces dropped as duplicates, unfitted, or with too short a window."),
            col("tau0_consistent", tau0, TAU0, "consistent",
                "With repeats: 1 if they agree within 10 %, 0 if not. Blank for one "
                "measurement."),
            col("trace_used", tau0, TAU0, "trace_used", "The trace tau0 is taken from."),
            col("tau0_note", tau0, TAU0, "note", "How tau0 was selected, and what was rejected."),
        ]),
    ]
    note = """
        One row per photocatalyst with a compound record. PC7 was measured but is not in the
        screen; PC26 has no record beyond its number. PC24 is in the screen and excluded from
        the dataset in dataset/build_dataset.py, not here.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(records), blocks, note


def quenchers():
    records = read_table(Q_RECORDS, "quencher_id")
    absorption = read_table(ABSORPTION, "quencher_id")
    cv = read_table(CV, "quencher_id")

    for name, table in (("absorption", absorption), ("cyclic voltammetry", cv)):
        stray = sorted(set(table) - set(records))
        if stray:
            sys.exit(f"{name} rows for quenchers with no record: {stray}")

    def col(header, table, path, name, description):
        return Column(header, description, src(path, name), field(table, name))

    epsilon = [col(f"epsilon_{nm} / mM-1 cm-1", absorption, ABSORPTION,
                   f"epsilon_{nm}_mM-1cm-1", f"Molar extinction coefficient at {nm} nm.")
               for nm in EPSILON_NM]
    epsilon.append(col("max_read_rsd / %", absorption, ABSORPTION, "max_read_rsd_percent",
                       "Largest relative spread between repeat reads across the 18 "
                       "wavelengths."))

    blocks = [
        Block("Identity", [
            col("quencher_id", records, Q_RECORDS, "quencher_id", "Quencher number."),
            col("quencher_name", records, Q_RECORDS, "quencher_name",
                "Label used throughout, Q<n>."),
            col("compound", records, Q_RECORDS, "name", "Compound, as recorded."),
            col("smiles", records, Q_RECORDS, "smiles", "Structure."),
        ]),
        Block("Redox potentials, as recorded / V vs SCE", [
            col("E_Ox_vs_SCE_exp / V", records, Q_RECORDS, "E_Ox_vs_SCE_exp",
                "Oxidation potential. Where E_Ox_bound is set, a limit, not a value."),
            col("E_Ox_bound", records, Q_RECORDS, "E_Ox_bound",
                "'lower': not oxidized below this potential; the true value lies above it."),
            col("E_Red_vs_SCE_exp / V", records, Q_RECORDS, "E_Red_vs_SCE_exp",
                "Reduction potential. Where E_Red_bound is set, a limit, not a value."),
            col("E_Red_bound", records, Q_RECORDS, "E_Red_bound",
                "'upper': not reduced above this potential; the true value lies below it."),
            col("redox_source_type", records, Q_RECORDS, "redox_source_type",
                "measured (CV in this lab), literature, estimated, or unattributed."),
            col("redox_source_note", records, Q_RECORDS, "redox_source_note",
                "Free-text provenance from the lab record."),
            col("redox_doi", records, Q_RECORDS, "redox_doi", "Literature source, where given."),
        ]),
        Block("Cyclic voltammetry, re-read from the traces / V vs SCE", [
            col("E_Ox_CV / V", cv, CV, "E_ox_V",
                "Oxidation potential extracted from the deposited trace. A check on the "
                "recorded value, not a replacement for it."),
            col("E_Ox_CV_basis", cv, CV, "E_ox_basis", "onset, or half-wave where reversible."),
            col("E_Ox_CV_agrees", cv, CV, "ox_agrees",
                "1 if consistent with the recorded value within 0.15 V (or with its bound)."),
            col("E_Red_CV / V", cv, CV, "E_red_V",
                "Reduction potential extracted from the deposited trace."),
            col("E_Red_CV_basis", cv, CV, "E_red_basis", "onset, or half-wave where reversible."),
            col("E_Red_CV_agrees", cv, CV, "red_agrees",
                "1 if consistent with the recorded value within 0.15 V (or with its bound)."),
        ]),
        Block("Molar extinction coefficients in MeCN / mM-1 cm-1", epsilon),
        Block("Sample", [
            col("c_stock / mM", records, Q_RECORDS, "c_stock_mM", "Stock concentration."),
            col("deck_position", records, Q_RECORDS, "deck_position",
                "Deck slot of the stock plate."),
            col("well_position", records, Q_RECORDS, "well_position", "Well of the stock plate."),
        ]),
    ]
    note = """
        One row per quencher. Cyclic voltammetry traces are deposited for 19 quenchers; the
        re-read block is blank for the rest. The traces themselves, the per-sample detail and
        the reasons for any disagreement are in
        experimental_data_raw/quencher_cyclic_voltammetry/cyclic_voltammetry_redox.csv.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(records), blocks, note


def screened_photocatalysts():
    """Photocatalysts with Stern-Volmer fits. The fits have one row per pair, not per id."""
    if not FITS.exists():
        sys.exit(f"missing input: {FITS}\nrun stern_volmer_fits.py first")
    with FITS.open(encoding="utf-8", newline="") as fh:
        return {int(row["pc_id"]) for row in csv.DictReader(fh)}


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
