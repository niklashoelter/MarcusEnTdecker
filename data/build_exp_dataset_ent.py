# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl==3.1.5"]
# ///
"""
Build the energy-transfer dataset: the pairs of exp_dataset on which quenching can be
attributed to energy transfer, with the calculated and predicted properties joined on.

Reads   exp_dataset.csv
        experimental_data_processed/quencher_properties_exp.xlsx
        computational_data_processed/photocatalyst_properties_calc.xlsx
        computational_data_processed/quencher_properties_calc.xlsx
        computational_data_processed/quencher_properties_ml.xlsx
Writes  exp_dataset_ent.csv     input to ../benchmark/
        exp_dataset_ent.xlsx    the same table in the property-workbook layout

    uv run data/build_exp_dataset_ent.py

Pairs: every non-quenching pair; a quenching pair only if every SET channel with a
measured quencher potential is more than 0.25 V uphill (a limit E_ox > x, E_red < x
counts only if the limit itself is past 0.25 V):

    oxidation of the quencher by PC*   dE_SET_ox  = E_ox(Q) - E(PC*/PC-)
    reduction of the quencher by PC*   dE_SET_red = E(PC+/PC*) - E_red(Q)

Quencher potentials measured, photocatalyst excited-state potentials DFT.
Photocatalyst E_T, lambda and radius (S0) from DFT; quencher E_T, lambda and radius from
DFT, E_T and lambda also from ML with uncertainty; quencher DvTE (vertical S-T gaps from
MD): mean, SD, p-quantiles of a normal distribution and of a KDE.
dG_EnT = E_T(Q) - E_T(PC). Missing values blank.
kcal/mol, V and A to 2 decimals; SET differences rounded before the 0.25 V comparison.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))

from build_exp_dataset import COLUMNS as EXP_COLUMNS, LAYOUT as EXP_LAYOUT, load, number  # noqa: E402
from properties_workbook import Block, Column, write_workbook  # noqa: E402

EXP_DATASET = DATA / "exp_dataset.csv"
Q_EXP = DATA / "experimental_data_processed" / "quencher_properties_exp.xlsx"
PC_CALC = DATA / "computational_data_processed" / "photocatalyst_properties_calc.xlsx"
Q_CALC = DATA / "computational_data_processed" / "quencher_properties_calc.xlsx"
Q_ML = DATA / "computational_data_processed" / "quencher_properties_ml.xlsx"
OUT_CSV = DATA / "exp_dataset_ent.csv"
OUT_XLSX = DATA / "exp_dataset_ent.xlsx"

SET_EXCLUSION_V = 0.25
DECIMALS = 2                                    # for kcal/mol, V and A
DVTE_P = ("0.005", "0.008", "0.01", "0.015")   # quantiles of the vertical gaps

# (block, [(column, description, source)]) for the columns added to exp_dataset's.
ADDED = [
    ("Electron transfer", [
        ("dE_SET_ox / V", "E_ox(Q) - E(PC*/PC-): how far oxidation of the quencher by PC* is "
         "uphill. Blank without a measured E_ox.",
         "quencher_properties_exp.xlsx: E_Ox_vs_SCE_exp; "
         "photocatalyst_properties_calc.xlsx: E_PCstar_PCminus"),
        ("dE_SET_ox_bound", "'>' where E_ox is a limit, so dE_SET_ox is a lower bound.",
         "quencher_properties_exp.xlsx: E_Ox_bound"),
        ("dE_SET_red / V", "E(PC+/PC*) - E_red(Q): how far reduction of the quencher by PC* is "
         "uphill. Blank without a measured E_red.",
         "quencher_properties_exp.xlsx: E_Red_vs_SCE_exp; "
         "photocatalyst_properties_calc.xlsx: E_PCplus_PCstar"),
        ("dE_SET_red_bound", "'>' where E_red is a limit, so dE_SET_red is a lower bound.",
         "quencher_properties_exp.xlsx: E_Red_bound"),
    ]),
    ("Photocatalyst, DFT", [
        ("pc_E_T_dft / kcal mol-1", "Adiabatic triplet energy.",
         "photocatalyst_properties_calc.xlsx: E_T"),
        ("pc_lambda_S0_T1_dft / kcal mol-1", "E_S0 at the T1 geometry minus E_S0.",
         "photocatalyst_properties_calc.xlsx: lambda_S0_T1"),
        ("pc_lambda_T1_S0_dft / kcal mol-1", "E_T1 at the S0 geometry minus E_T1.",
         "photocatalyst_properties_calc.xlsx: lambda_T1_S0"),
        ("pc_radius_dft / A", "Radius of the sphere with the van der Waals volume of the S0 "
         "geometry.", "photocatalyst_properties_calc.xlsx: radius_S0"),
    ]),
    ("Quencher, DFT", [
        ("q_E_T_dft / kcal mol-1", "Adiabatic triplet energy.",
         "quencher_properties_calc.xlsx: E_T"),
        ("q_lambda_S0_T1_dft / kcal mol-1", "E_S0 at the T1 geometry minus E_S0.",
         "quencher_properties_calc.xlsx: lambda_S0_T1"),
        ("q_lambda_T1_S0_dft / kcal mol-1", "E_T1 at the S0 geometry minus E_T1.",
         "quencher_properties_calc.xlsx: lambda_T1_S0"),
        ("q_radius_dft / A", "Radius of the sphere with the van der Waals volume of the S0 "
         "geometry.", "quencher_properties_calc.xlsx: radius_S0"),
    ]),
    ("Quencher, DvTE", [
        ("q_dvte_mu / kcal mol-1", "Mean of the sampled vertical singlet-triplet gaps.",
         "quencher_properties_calc.xlsx: dvte_mu"),
        ("q_dvte_std / kcal mol-1", "Standard deviation of the sampled vertical gaps.",
         "quencher_properties_calc.xlsx: dvte_std"),
        *[(f"q_dvte_{p} / kcal mol-1", f"p = {p} quantile of a normal distribution, mu + Phi^-1(p) * std.",
           f"quencher_properties_calc.xlsx: dvte_{p}") for p in DVTE_P],
        *[(f"q_dvte_{p}_kde / kcal mol-1", f"p = {p} quantile of a Gaussian KDE of the gaps.",
           f"quencher_properties_calc.xlsx: dvte_{p}_kde") for p in DVTE_P],
    ]),
    ("Quencher, ML", [
        ("q_E_T_ml / kcal mol-1", "Adiabatic triplet energy, predicted.",
         "quencher_properties_ml.xlsx: E_T"),
        ("q_lambda_S0_T1_ml / kcal mol-1", "lambda_S0_T1, predicted.",
         "quencher_properties_ml.xlsx: lambda_S0_T1"),
        ("q_lambda_T1_S0_ml / kcal mol-1", "lambda_T1_S0, predicted.",
         "quencher_properties_ml.xlsx: lambda_T1_S0"),
        ("q_E_T_ml_uncertainty / kcal mol-1", "The model's uncertainty on E_T.",
         "quencher_properties_ml.xlsx: E_T_uncertainty"),
        ("q_lambda_S0_T1_ml_uncertainty / kcal mol-1", "The model's uncertainty on lambda_S0_T1.",
         "quencher_properties_ml.xlsx: lambda_S0_T1_uncertainty"),
        ("q_lambda_T1_S0_ml_uncertainty / kcal mol-1", "The model's uncertainty on lambda_T1_S0.",
         "quencher_properties_ml.xlsx: lambda_T1_S0_uncertainty"),
    ]),
    ("Driving force", [
        ("dG_EnT_dft / kcal mol-1", "q_E_T_dft - pc_E_T_dft.", "computed here"),
        ("dG_EnT_ml / kcal mol-1", "q_E_T_ml - pc_E_T_dft.", "computed here"),
    ]),
]
LAYOUT = EXP_LAYOUT + ADDED
COLUMNS = [name for _, columns in LAYOUT for name, _, _ in columns]

BOUNDS = {"E_Ox_bound": ">", "E_Red_bound": "<"}


def difference(a, b):
    return a - b if None not in (a, b) else None


def rounded(x):
    """A value in kcal/mol or V, to DECIMALS; None stays None."""
    return None if x is None else round(x, DECIMALS)


def cell(x):
    """rounded(x) as a table cell, blank if absent."""
    x = rounded(x)
    return "" if x is None else x


def set_ruled_out(channels):
    """True if at least one channel is measured and every measured one is past the rule."""
    measured = [value for value in channels if value is not None]
    return bool(measured) and all(value > SET_EXCLUSION_V for value in measured)


def main():
    if not EXP_DATASET.exists():
        sys.exit(f"missing input: {EXP_DATASET}\nrun build_exp_dataset.py first")
    q_exp = load(Q_EXP, "quencher_id")
    pc_calc = load(PC_CALC, "new_id")
    q_calc = load(Q_CALC, "quencher_id")
    q_ml = load(Q_ML, "quencher_id")

    for q in q_exp.values():
        for field, allowed in BOUNDS.items():
            if str(q.get(field, "")).strip() not in ("", allowed):
                sys.exit(f"quencher {q['quencher_id']}: unexpected {field} {q[field]!r}")

    rows, dropped = [], 0
    with EXP_DATASET.open(encoding="utf-8", newline="") as fh:
        for pair in csv.DictReader(fh):
            pc, qc = pc_calc.get(int(pair["pc_new_id"])), q_calc.get(int(pair["quencher_id"]))
            qe, qm = q_exp.get(int(pair["quencher_id"])), q_ml.get(int(pair["quencher_id"]))

            de_ox = rounded(difference(number(qe, "E_Ox_vs_SCE_exp / V"),
                                       number(pc, "E_PCstar_PCminus / V")))
            de_red = rounded(difference(number(pc, "E_PCplus_PCstar / V"),
                                        number(qe, "E_Red_vs_SCE_exp / V")))
            if pair["quenches"] == "1" and not set_ruled_out([de_ox, de_red]):
                dropped += 1
                continue

            pc_e_t = number(pc, "E_T / kcal mol-1")
            q_e_t_dft, q_e_t_ml = number(qc, "E_T / kcal mol-1"), number(qm, "E_T / kcal mol-1")
            rows.append({
                **{name: pair[name] for name in EXP_COLUMNS},
                "dE_SET_ox / V": cell(de_ox),
                "dE_SET_ox_bound": ">" if de_ox is not None and qe.get("E_Ox_bound") else "",
                "dE_SET_red / V": cell(de_red),
                "dE_SET_red_bound": ">" if de_red is not None and qe.get("E_Red_bound") else "",
                "pc_E_T_dft / kcal mol-1": cell(pc_e_t),
                "pc_lambda_S0_T1_dft / kcal mol-1": cell(number(pc, "lambda_S0_T1 / kcal mol-1")),
                "pc_lambda_T1_S0_dft / kcal mol-1": cell(number(pc, "lambda_T1_S0 / kcal mol-1")),
                "pc_radius_dft / A": cell(number(pc, "radius_S0 / A")),
                "q_E_T_dft / kcal mol-1": cell(q_e_t_dft),
                "q_lambda_S0_T1_dft / kcal mol-1": cell(number(qc, "lambda_S0_T1 / kcal mol-1")),
                "q_lambda_T1_S0_dft / kcal mol-1": cell(number(qc, "lambda_T1_S0 / kcal mol-1")),
                "q_radius_dft / A": cell(number(qc, "radius_S0 / A")),
                "q_dvte_mu / kcal mol-1": cell(number(qc, "dvte_mu / kcal mol-1")),
                "q_dvte_std / kcal mol-1": cell(number(qc, "dvte_std / kcal mol-1")),
                **{f"q_dvte_{p}{kind} / kcal mol-1": cell(number(qc, f"dvte_{p}{kind} / kcal mol-1"))
                   for p in DVTE_P for kind in ("", "_kde")},
                "q_E_T_ml / kcal mol-1": cell(q_e_t_ml),
                "q_lambda_S0_T1_ml / kcal mol-1": cell(number(qm, "lambda_S0_T1 / kcal mol-1")),
                "q_lambda_T1_S0_ml / kcal mol-1": cell(number(qm, "lambda_T1_S0 / kcal mol-1")),
                **{f"q_{name}_ml_uncertainty / kcal mol-1": cell(number(qm, f"{name}_uncertainty / kcal mol-1"))
                   for name in ("E_T", "lambda_S0_T1", "lambda_T1_S0")},
                "dG_EnT_dft / kcal mol-1": cell(difference(q_e_t_dft, pc_e_t)),
                "dG_EnT_ml / kcal mol-1": cell(difference(q_e_t_ml, pc_e_t)),
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    blocks = [Block(title, [Column(name, description, source,
                                   lambda i, name=name: _cell(rows[i][name]))
                            for name, description, source in columns])
              for title, columns in LAYOUT]
    write_workbook(OUT_XLSX, "Pairs", list(range(len(rows))), blocks, """
        The pairs of exp_dataset on which quenching can be attributed to energy transfer.
        Built by data/build_exp_dataset_ent.py; exp_dataset_ent.csv holds the same table.
        Every pair that does not quench is kept. A pair that quenches is kept only if every
        channel of single-electron transfer with a measured quencher potential is more
        than 0.25 V uphill.
    """)

    quenching = sum(r["quenches"] == "1" for r in rows)
    print(f"{len(rows)} pairs -> {OUT_CSV.name}, {OUT_XLSX.name}")
    print(f"  quenches {quenching}, does not quench {len(rows) - quenching}")
    print(f"  {dropped} quenching pairs dropped: electron transfer not ruled out")
    for name in COLUMNS:
        missing = sum(1 for r in rows if r[name] == "")
        if missing and not name.endswith("_bound"):
            print(f"  {name}: blank for {missing} pairs")


def _cell(value):
    """A CSV value as the workbook cell: numbers as numbers, blanks as empty."""
    if value in ("", None):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        number_value = float(value)
    except ValueError:
        return value
    return int(number_value) if str(value).lstrip("-").isdigit() else number_value


if __name__ == "__main__":
    main()
