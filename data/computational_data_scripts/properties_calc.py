# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
"""
Computational tables -> one property workbook per compound class.

Reads   ../computational_data_raw/photocatalysts/photocatalyst_dft.csv
        ../computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv
        ../computational_data_raw/quencher_adiabatic_DvTE/quencher_dvte.csv
        ../experimental_data_raw/compound_records/compound_records_*.csv   (names only)
Writes  ../computational_data_processed/photocatalyst_properties_calc.xlsx
        ../computational_data_processed/quencher_properties_calc.xlsx

Everything per compound that was computed, in one place. The extraction scripts write
their full tables -- total energies, Gibbs corrections, per-conformer and per-sample
detail -- into the raw folders beside the jobs they read. Those tables are in git even
though the jobs are not, so this runs without the deposit. It only selects from them and
lays the result out in blocks (see ../properties_workbook.py). No value is computed here.

The compound names come from the experimental records, so a row reads the same in the
computed and the measured workbook. PC30 and PC31 were computed but never screened and
have no record, so their name is blank.
"""

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from properties_workbook import Block, Column, field, read_table, write_workbook  # noqa: E402

RAW = DATA / "computational_data_raw"
OUT_DIR = DATA / "computational_data_processed"
PC_OUT = OUT_DIR / "photocatalyst_properties_calc.xlsx"
Q_OUT = OUT_DIR / "quencher_properties_calc.xlsx"

PC_DFT = RAW / "photocatalysts" / "photocatalyst_dft.csv"
Q_DFT = RAW / "quencher_adiabatic_DFT" / "quencher_dft.csv"
Q_DVTE = RAW / "quencher_adiabatic_DvTE" / "quencher_dvte.csv"
RECORDS = DATA / "experimental_data_raw" / "compound_records"
PC_RECORDS = RECORDS / "compound_records_photocatalysts.csv"
Q_RECORDS = RECORDS / "compound_records_quenchers.csv"

KCAL = "kcal mol-1"
DVTE_CUTOFFS = ["0.005", "0.008", "0.01", "0.015"]


def src(path, column):
    return f"{path.relative_to(DATA).as_posix()}: {column}"


def photocatalysts():
    dft = read_table(PC_DFT, "pc_id")
    records = read_table(PC_RECORDS, "pc_id")

    def col(header, name, description, table=dft, path=PC_DFT):
        return Column(header, description, src(path, name), field(table, name))

    lambdas = [
        ("S0", "T1", "triplet"), ("T1", "S0", "singlet"),
        ("S0", "ox", "cation"), ("ox", "S0", "neutral"),
        ("S0", "red", "anion"), ("red", "S0", "neutral"),
    ]
    blocks = [
        Block("Identity", [
            col("pc_id", "pc_id", "Photocatalyst number."),
            Column("pc_name", "Label used throughout, PC<n>.", "PC<pc_id>",
                   lambda i: f"PC{i}"),
            col("compound", "name", "Compound, from the experimental record. Blank for "
                "PC30 and PC31, which were never screened.", records, PC_RECORDS),
            col("n_atoms", "n_atoms", "Atoms in the S0 structure."),
            col("charge", "charge", "Charge of the ground state."),
        ]),
        Block(f"Triplet energy / {KCAL}", [
            col(f"E_T / {KCAL}", f"E_T / {KCAL}",
                "Adiabatic, Gibbs-corrected: (E_T1 + G_T1) - (E_S0 + G_S0)."),
        ]),
        Block(f"Reorganization energies / {KCAL}", [
            col(f"lambda_{a}_to_{b} / {KCAL}", f"lambda_{a}_to_{b} / {KCAL}",
                f"Relaxation from the {a} geometry to the {b} geometry on the {surface} "
                f"surface. Electronic energies only.")
            for a, b, surface in lambdas
        ]),
        Block("Redox potentials / V vs SCE", [
            col("E_PCplus_PC / V", "E_PCplus_PC / V", "PC+/PC, ground-state oxidation."),
            col("E_PC_PCminus / V", "E_PC_PCminus / V", "PC/PC-, ground-state reduction."),
            col("E_PCstar_PCminus / V", "E_PCstar_PCminus / V",
                "PC*/PC-: how strong an oxidant the triplet is. E_PC_PCminus + E_T."),
            col("E_PCplus_PCstar / V", "E_PCplus_PCstar / V",
                "PC+/PC*: how strong a reductant the triplet is. E_PCplus_PC - E_T."),
        ]),
        Block("Calculation checks", [
            *[col(f"n_imag_{s}", f"n_imag_{s}", f"Imaginary frequencies in the {s} optimization.")
              for s in ("S0", "T1", "ox", "red")],
            col("minimum_confirmed", "minimum_confirmed",
                "1 if none of the four optimizations has an imaginary frequency. Says "
                "nothing about missing jobs; n_jobs does."),
            col("n_jobs", "n_jobs", "Jobs found, of the ten the scheme needs."),
            col("note", "note", "What is missing or flagged."),
        ]),
    ]
    note = """
        One row per computed photocatalyst. Total energies and Gibbs corrections per job are in
        computational_data_raw/photocatalysts/photocatalyst_dft.csv.
        PC19 lacks four single points, so four of its six lambdas are blank; see its note.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    return sorted(dft), blocks, note


def quenchers():
    dft = read_table(Q_DFT, "quencher_id")
    dvte = read_table(Q_DVTE, "quencher_id")
    records = read_table(Q_RECORDS, "quencher_id")

    def col(header, name, description, table=dft, path=Q_DFT):
        return Column(header, description, src(path, name), field(table, name))

    def dv(header, name, description):
        return col(header, name, description, dvte, Q_DVTE)

    blocks = [
        Block("Identity", [
            col("quencher_id", "quencher_id", "Quencher number."),
            col("quencher_name", "quencher_name", "Label used throughout, Q<n>.",
                records, Q_RECORDS),
            col("compound", "name", "Compound, from the experimental record.",
                records, Q_RECORDS),
            col("smiles", "smiles", "Structure.", records, Q_RECORDS),
        ]),
        Block(f"Triplet state, adiabatic DFT / {KCAL}", [
            col(f"E_T / {KCAL}", "E_T",
                "Adiabatic triplet energy, Gibbs-corrected, lowest conformer of each state."),
            col(f"lambda_S_T / {KCAL}", "lambda_S_T",
                "Singlet surface: E_S0 at the T1 geometry minus E_S0 at the S0 geometry."),
            col(f"lambda_T_S / {KCAL}", "lambda_T_S",
                "Triplet surface: E_T1 at the S0 geometry minus E_T1 at the T1 geometry."),
        ]),
        Block("Oxidation, adiabatic DFT", [
            col(f"dG_Ox / {KCAL}", "dG_Ox", "Free energy of Q -> Q+."),
            col("E_Ox_vs_SCE_calc / V", "E_Ox_vs_SCE_calc", "dG_Ox / F - 4.442 V."),
            col(f"lambda_S0_D+ / {KCAL}", "lambda_S0_D+",
                "Neutral surface: E_S0 at the cation geometry minus E_S0."),
            col(f"lambda_D+_S0 / {KCAL}", "lambda_D+_S0",
                "Cation surface: E_cation at the S0 geometry minus E_cation."),
        ]),
        Block("Reduction, adiabatic DFT", [
            col(f"dG_Red / {KCAL}", "dG_Red", "Free energy released by Q -> Q-."),
            col("E_Red_vs_SCE_calc / V", "E_Red_vs_SCE_calc", "dG_Red / F - 4.442 V."),
            col(f"lambda_S0_D- / {KCAL}", "lambda_S0_D-",
                "Neutral surface: E_S0 at the anion geometry minus E_S0."),
            col(f"lambda_D-_S0 / {KCAL}", "lambda_D-_S0",
                "Anion surface: E_anion at the S0 geometry minus E_anion."),
        ]),
        Block("Conformers", [
            col(f"n_conformers_{s}", f"n_conformers_{s}",
                f"Conformers computed for {s}; the lowest in energy is used.")
            for s in ("S0", "T1", "ox", "red")
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
        One row per quencher. Roundings are the original pipeline's: 1 decimal in kcal/mol,
        2 in V. Per-conformer energies are in
        computational_data_raw/quencher_adiabatic_DFT/quencher_dft_conformers.csv, and every
        sampled gap in computational_data_raw/quencher_adiabatic_DvTE/quencher_dvte_samples.csv.gz.
        Read with pandas as: pd.read_excel(path, header=1), then drop the 'Unnamed' columns.
    """
    stray = sorted((set(dvte) | set(records)) ^ set(dft))
    if stray:
        sys.exit(f"quencher ids not shared by DFT, DvTE and the records: {stray}")
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
