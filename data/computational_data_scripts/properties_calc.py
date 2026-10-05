# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
"""
Computational tables -> one property workbook per compound class.

Reads   ../computational_data_raw/photocatalysts/photocatalyst_dft.csv
        ../computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv
        ../computational_data_raw/quencher_adiabatic_DvTE/quencher_dvte.csv
        ../compound_identification.xlsx   (ids, names, SMILES)
Writes  ../computational_data_processed/photocatalyst_properties_calc.xlsx
        ../computational_data_processed/quencher_properties_calc.xlsx

    uv run data/computational_data_scripts/properties_calc.py

Collects the computed properties of each compound. The extraction scripts write their
full tables into the raw folders beside the jobs. These tables are in git, so this script
runs without the job deposit. It selects columns and lays them out in blocks
(../properties_workbook.py); no value is computed here. Compound names come from
compound_identification.xlsx, as in the experimental workbooks.
"""

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from compound_identification import NEW_ID, PHOTOCATALYSTS, QUENCHERS  # noqa: E402
from properties_workbook import Block, Column, field, read_table, write_workbook  # noqa: E402

RAW = DATA / "computational_data_raw"
OUT_DIR = DATA / "computational_data_processed"
PC_OUT = OUT_DIR / "photocatalyst_properties_calc.xlsx"
Q_OUT = OUT_DIR / "quencher_properties_calc.xlsx"

PC_DFT = RAW / "photocatalysts" / "photocatalyst_dft.csv"
Q_DFT = RAW / "quencher_adiabatic_DFT" / "quencher_dft.csv"
Q_DVTE = RAW / "quencher_adiabatic_DvTE" / "quencher_dvte.csv"

KCAL = "kcal mol-1"
DVTE_CUTOFFS = ["0.005", "0.008", "0.01", "0.015"]


IDENTIFICATION = "compound_identification.xlsx"

# Reorganization energies are lambda_<surface>_<geometry> in both workbooks.
LAMBDA_STATES = "Dplus and Dminus are the one-electron oxidized and reduced states."


def src(path, column):
    return f"{path.relative_to(DATA).as_posix()}: {column}"


def identity(header, table, sheet, name, description):
    """A column taken from compound_identification.xlsx by id."""
    return Column(header, description, f"{IDENTIFICATION}: {sheet}/{name}",
                  lambda i: table[i][name])


def photocatalysts():
    dft = read_table(PC_DFT, "pc_id")
    if set(dft) != set(NEW_ID):
        sys.exit(f"photocatalyst DFT disagree with {IDENTIFICATION}: "
                 f"{sorted(set(dft) ^ set(NEW_ID))}")

    def col(header, name, description, table=dft, path=PC_DFT):
        return Column(header, description, src(path, name), field(table, name))

    # lambda_<surface>_<geometry>, the same six names as the quencher workbook.
    lambdas = [("S0", "T1"), ("T1", "S0"), ("S0", "Dplus"), ("Dplus", "S0"),
               ("S0", "Dminus"), ("Dminus", "S0")]
    blocks = [
        Block("Identity", [
            Column("new_id", "PC number used downstream and in the paper, 1-27.",
                   f"{IDENTIFICATION}: Photocatalysts/new_id", lambda i: NEW_ID[i]),
            Column("old_id", "PC number during the screen, used by the raw files.",
                   src(PC_DFT, "pc_id"), lambda i: i),
            identity("compound", PHOTOCATALYSTS, "Photocatalysts", "name",
                     "Compound."),
        ]),
        Block(f"Triplet energy / {KCAL}", [
            col(f"E_T / {KCAL}", f"E_T / {KCAL}",
                "Adiabatic, Gibbs-corrected: ((E_T1 + G_T1) - (E_S0 + G_S0)) x 627.509."),
        ]),
        Block(f"Reorganization energies / {KCAL}", [
            col(f"lambda_{s}_{g} / {KCAL}", f"lambda_{s}_{g} / {KCAL}",
                f"{s} surface: E_{s} at the {g} geometry minus E_{s} at its own, electronic "
                f"energies x 627.509. {LAMBDA_STATES}")
            for s, g in lambdas
        ]),
        Block(f"Excited-state redox / {KCAL} and V vs SCE", [
            col(f"dG_ox_ES / {KCAL}", f"dG_ox_ES / {KCAL}",
                "PC* -> PC+: ((E_ox + G_ox) - (E_T1 + G_T1)) x 627.509."),
            col(f"dG_red_ES / {KCAL}", f"dG_red_ES / {KCAL}",
                "PC* -> PC-: ((E_red + G_red) - (E_T1 + G_T1)) x 627.509."),
            col("E_PCstar_PCminus / V", "E_PCstar_PCminus / V",
                "PC*/PC-, how strong an oxidant the triplet is: -dG_red_ES / 23.061 - 4.42."),
            col("E_PCplus_PCstar / V", "E_PCplus_PCstar / V",
                "PC+/PC*, how strong a reductant the triplet is: dG_ox_ES / 23.061 - 4.42."),
        ]),
    ]
    note = """
        One row per photocatalyst in the screen, sorted by new_id; the raw files use old_id.
        Every value is computed from the ORCA outputs. Total energies, Gibbs corrections and
        frequency checks are in computational_data_raw/photocatalysts/photocatalyst_dft.csv.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(dft, key=NEW_ID.get), blocks, note


def quenchers():
    dft = read_table(Q_DFT, "quencher_id")
    dvte = read_table(Q_DVTE, "quencher_id")

    def col(header, name, description, table=dft, path=Q_DFT):
        return Column(header, description, src(path, name), field(table, name))

    def dv(header, name, description):
        return col(header, name, description, dvte, Q_DVTE)

    blocks = [
        Block("Identity", [
            col("quencher_id", "quencher_id", "Quencher number."),
            Column("quencher_name", "Label used throughout, Q<n>.",
                   f"{IDENTIFICATION}: Quenchers/quencher_id", lambda i: f"Q{i}"),
            identity("compound", QUENCHERS, "Quenchers", "name",
                     "Name derived from the SMILES, as in the experimental workbook."),
            identity("smiles", QUENCHERS, "Quenchers", "smiles", "Structure."),
        ]),
        Block(f"Triplet state, adiabatic DFT / {KCAL}", [
            col(f"E_T / {KCAL}", "E_T",
                "Adiabatic triplet energy, Gibbs-corrected, lowest conformer of each state."),
            col(f"lambda_S0_T1 / {KCAL}", "lambda_S0_T1",
                "S0 surface: E_S0 at the T1 geometry minus E_S0 at its own."),
            col(f"lambda_T1_S0 / {KCAL}", "lambda_T1_S0",
                "T1 surface: E_T1 at the S0 geometry minus E_T1 at its own."),
        ]),
        Block("Oxidation, adiabatic DFT", [
            col(f"dG_Ox / {KCAL}", "dG_Ox", "Free energy of Q -> Q+."),
            col("E_Ox_vs_SCE_calc / V", "E_Ox_vs_SCE_calc", "dG_Ox / F - 4.42 V."),
            col(f"lambda_S0_Dplus / {KCAL}", "lambda_S0_Dplus",
                "S0 surface: E_S0 at the Dplus (radical cation) geometry minus E_S0 at its own."),
            col(f"lambda_Dplus_S0 / {KCAL}", "lambda_Dplus_S0",
                "Dplus surface: E_Dplus at the S0 geometry minus E_Dplus at its own."),
        ]),
        Block("Reduction, adiabatic DFT", [
            col(f"dG_Red / {KCAL}", "dG_Red", "Free energy released by Q -> Q-."),
            col("E_Red_vs_SCE_calc / V", "E_Red_vs_SCE_calc", "dG_Red / F - 4.42 V."),
            col(f"lambda_S0_Dminus / {KCAL}", "lambda_S0_Dminus",
                "S0 surface: E_S0 at the Dminus (radical anion) geometry minus E_S0 at its own."),
            col(f"lambda_Dminus_S0 / {KCAL}", "lambda_Dminus_S0",
                "Dminus surface: E_Dminus at the S0 geometry minus E_Dminus at its own."),
        ]),
        Block(f"Vertical triplet energy distribution, DvTE / {KCAL}", [
            dv("n_samples", "n_samples", "Sampled geometries with both single points."),
            dv("n_trajectories", "n_trajectories", "MD trajectories sampled."),
            dv(f"dvte_mu / {KCAL}", "dvte_mu", "Mean vertical singlet-triplet gap."),
            dv(f"dvte_std / {KCAL}", "dvte_std", "Standard deviation of the gaps."),
            *[dv(f"dvte_{p} / {KCAL}", f"dvte_{p}",
                 f"{p} quantile of a normal distribution with that mean and deviation.")
              for p in DVTE_CUTOFFS],
            *[dv(f"dvte_{p}_kde / {KCAL}", f"dvte_{p}_kde",
                 f"{p} quantile of a kernel density estimate of the samples (Scott's rule).")
              for p in DVTE_CUTOFFS],
        ]),
    ]
    note = """
        One row per quencher. DFT values are rounded to 2 decimals in kcal/mol and V.
        The energies behind each value are in the deposited ORCA outputs under
        computational_data_raw/quencher_adiabatic_DFT/Q<n>/<state>/, and every sampled gap in
        computational_data_raw/quencher_adiabatic_DvTE/quencher_dvte_samples.csv.gz.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    stray = sorted((set(dvte) | set(QUENCHERS)) ^ set(dft))
    if stray:
        sys.exit(f"quencher ids not shared by DFT, DvTE and {IDENTIFICATION}: {stray}")
    return sorted(dft), blocks, note


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
