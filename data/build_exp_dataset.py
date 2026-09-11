# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
"""
The processed data -> one row per photocatalyst-quencher pair, with exactly the columns
the analysis downstream reads.

Reads   experimental_data_processed/stern_volmer_fits.csv
        experimental_data_processed/photocatalyst_properties_exp.xlsx
        experimental_data_processed/quencher_properties_exp.xlsx
        computational_data_processed/photocatalyst_properties_calc.xlsx
        computational_data_processed/quencher_properties_calc.xlsx
Writes  exp_dataset.csv     what ../benchmark/ reads
        exp_dataset.xlsx    the same table in the property-workbook layout, with a
                            Column key sheet

    uv run data/build_exp_dataset.py

THE RATE CONSTANT
    k_q = K_SV / tau_0, K_SV per pair from the Stern-Volmer fits, tau_0 per
    photocatalyst from its TCSPC trace. Both measured here.

THE LABEL IS SET ON K_SV, NOT ON k_q
    Both labels are made in experimental_data_scripts/stern_volmer_fits.py and copied here
    unchanged. `quenches`: K_SV > 10, or K_SV <= 10 with a slope significant across the
    points the fit used. `quenches_strict`: the published criterion, K_SV >= 25 and
    r2 >= 0.90. K_SV is the observable; dividing by a photocatalyst-specific tau_0 would
    move the detection floor with the photocatalyst, which is why no cutoff is set on k_q.

    A pair that does not quench is left-censored, not missing: `censored = 1`, its fitted
    K_SV kept, and `k_q_floor = 10 / tau_0` the smallest rate constant it could have shown.

THE DRIVING FORCES
    Quencher-centric, negative when exergonic, F = 23.060548 kcal mol-1 V-1:

        dG_EnT     = E_T(Q) - E_T(PC)
        dG_SET     = the more accessible of
                     F * (E_Ox(Q) - E(PC*/PC-))     Q oxidized, PC* the oxidant
                     F * (E(PC+/PC*) - E_Red(Q))    Q reduced,  PC* the reductant

    dG_SET is computed from the measured quencher potentials (`_exp`) and from the
    computed ones (`_calc`), never merged. A measured potential recorded as a limit
    (`>` or `<` in the quencher workbook) makes that channel's driving force a lower
    bound -- SET at least this endergonic -- and `dG_SET_exp_bound = lower` says so.

PC24 is in the screen but not in the dataset: it was discarded from the screen, a
scientific decision recorded in EXCLUDED_PHOTOCATALYSTS, not a quality filter.
"""

import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))

from openpyxl import load_workbook  # noqa: E402

from properties_workbook import Block, Column, write_workbook  # noqa: E402

EXPERIMENTAL = DATA / "experimental_data_processed"
COMPUTATIONAL = DATA / "computational_data_processed"
OUT_CSV = DATA / "exp_dataset.csv"
OUT_XLSX = DATA / "exp_dataset.xlsx"

# Row 1 of every property workbook names the column blocks; the headers are on row 2.
WORKBOOK_HEADER_ROW = 2

# The K_SV a pair has to clear to count as detected, and so the censoring limit. Equal to
# K_SV_LABEL in stern_volmer_fits.py.
K_SV_FLOOR = 10.0

SIGFIGS = 6
F_KCAL_PER_V = 23.060548

EXCLUDED_PHOTOCATALYSTS = {
    24: "discarded from the screen; marked 'discard' in the photocatalyst record",
}

KCAL = "kcal mol-1"

# (block, [(column, description, source)]). The order is the order of the table.
LAYOUT = [
    ("Identity", [
        ("pc_id", "Photocatalyst number.", "photocatalyst_properties_exp.xlsx"),
        ("pc_name", "PC<n>.", "photocatalyst_properties_exp.xlsx"),
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
        ("quenches", "1: K_SV > 10, or K_SV <= 10 with a significant slope. The label.",
         "stern_volmer_fits.csv: quenches"),
        ("quenches_strict", "1: K_SV >= 25 and r2 >= 0.90, the published criterion.",
         "stern_volmer_fits.csv: quenching_observed"),
        ("censored", "1 where quenches = 0: K_SV lies somewhere below the floor.",
         "computed here"),
    ]),
    ("Photocatalyst, DFT", [
        (f"pc_E_T_dft / {KCAL}", "Adiabatic triplet energy.", "photocatalyst_properties_calc.xlsx"),
        (f"pc_lambda_S0_to_T1_dft / {KCAL}", "Relaxation on the triplet surface.",
         "photocatalyst_properties_calc.xlsx"),
        (f"pc_lambda_T1_to_S0_dft / {KCAL}", "Relaxation on the singlet surface.",
         "photocatalyst_properties_calc.xlsx"),
    ]),
    ("Quencher, DFT and MD", [
        (f"q_lambda_S_T_dft / {KCAL}", "Relaxation on the singlet surface.",
         "quencher_properties_calc.xlsx"),
        (f"q_lambda_T_S_dft / {KCAL}", "Relaxation on the triplet surface.",
         "quencher_properties_calc.xlsx"),
        (f"q_dvte_mu_md / {KCAL}", "Mean vertical triplet energy over the MD samples.",
         "quencher_properties_calc.xlsx"),
        (f"q_dvte_std_md / {KCAL}", "Its standard deviation.", "quencher_properties_calc.xlsx"),
    ]),
    ("Driving forces", [
        (f"dG_EnT / {KCAL}", "E_T(Q) - E_T(PC), both DFT. Negative: exergonic.", "computed here"),
        (f"dG_SET_exp / {KCAL}", "More accessible SET channel, measured quencher potentials.",
         "computed here"),
        ("dG_SET_exp_bound", "'lower': a bounded potential; the true value is more endergonic.",
         "computed here"),
        (f"dG_SET_calc / {KCAL}", "More accessible SET channel, computed quencher potentials.",
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


def better_channel(ox, red):
    """The more accessible SET channel, (value, 'ox'|'red'), or (None, '') if neither."""
    available = [(v, name) for v, name in ((ox, "ox"), (red, "red")) if v is not None]
    return min(available) if available else (None, "")


def main():
    fits_path = EXPERIMENTAL / "stern_volmer_fits.csv"
    if not fits_path.exists():
        sys.exit(f"missing input: {fits_path}\nrun the data stage first")

    pc_exp = load(EXPERIMENTAL / "photocatalyst_properties_exp.xlsx", "pc_id")
    q_exp = load(EXPERIMENTAL / "quencher_properties_exp.xlsx", "quencher_id")
    pc_calc = load(COMPUTATIONAL / "photocatalyst_properties_calc.xlsx", "pc_id")
    q_calc = load(COMPUTATIONAL / "quencher_properties_calc.xlsx", "quencher_id")

    rows, excluded = [], 0
    with fits_path.open(encoding="utf-8", newline="") as fh:
        for fit in csv.DictReader(fh):
            pc_id, q_id = int(fit["pc_id"]), int(fit["quencher_id"])
            if pc_id in EXCLUDED_PHOTOCATALYSTS:
                excluded += 1
                continue
            pce, qe = pc_exp.get(pc_id), q_exp.get(q_id)
            pcc, qc = pc_calc.get(pc_id), q_calc.get(q_id)

            tau0 = number(pce, "tau0 / ns")
            k_sv = number(fit, "K_SV")
            # tau0 is in ns; k_q comes out in M-1 s-1.
            k_q = k_sv / (tau0 * 1e-9) if (k_sv is not None and tau0) else None
            floor = K_SV_FLOOR / (tau0 * 1e-9) if tau0 else None
            quenches = fit["quenches"] == "1"

            pc_e_t = number(pcc, f"E_T / {KCAL}")
            q_e_t = number(qc, f"E_T / {KCAL}")
            pc_star_ox = number(pcc, "E_PCstar_PCminus / V")    # PC* as the oxidant
            pc_star_red = number(pcc, "E_PCplus_PCstar / V")    # PC* as the reductant

            def set_forces(e_ox, e_red):
                ox = F_KCAL_PER_V * (e_ox - pc_star_ox) if None not in (e_ox, pc_star_ox) else None
                red = (F_KCAL_PER_V * (pc_star_red - e_red)
                       if None not in (e_red, pc_star_red) else None)
                return ox, red

            ox_exp, red_exp = set_forces(number(qe, "E_Ox_vs_SCE_exp / V"),
                                         number(qe, "E_Red_vs_SCE_exp / V"))
            set_exp, channel = better_channel(ox_exp, red_exp)
            # Either bound becomes a lower bound on the driving force: the quencher was not
            # oxidized (or reduced) within the window, so SET is at least this endergonic.
            bound_column = {"ox": "E_Ox_bound", "red": "E_Red_bound"}.get(channel)
            set_bound = "lower" if bound_column and (qe or {}).get(bound_column) else ""
            set_calc, _ = better_channel(*set_forces(number(qc, "E_Ox_vs_SCE_calc / V"),
                                                     number(qc, "E_Red_vs_SCE_calc / V")))

            rows.append({
                "pc_id": pc_id,
                "pc_name": (pce or {}).get("pc_name", ""),
                "quencher_id": q_id,
                "quencher_name": (qe or {}).get("quencher_name", ""),
                "K_SV / M-1": sig(k_sv),
                "tau0 / ns": sig(tau0),
                "k_q / M-1 s-1": sig(k_q),
                "k_q_floor / M-1 s-1": sig(floor),
                "quenches": int(quenches),
                "quenches_strict": int(fit["quenching_observed"] == "True"),
                "censored": int(not quenches),
                f"pc_E_T_dft / {KCAL}": sig(pc_e_t),
                f"pc_lambda_S0_to_T1_dft / {KCAL}": sig(number(pcc, f"lambda_S0_to_T1 / {KCAL}")),
                f"pc_lambda_T1_to_S0_dft / {KCAL}": sig(number(pcc, f"lambda_T1_to_S0 / {KCAL}")),
                f"q_lambda_S_T_dft / {KCAL}": sig(number(qc, f"lambda_S_T / {KCAL}")),
                f"q_lambda_T_S_dft / {KCAL}": sig(number(qc, f"lambda_T_S / {KCAL}")),
                f"q_dvte_mu_md / {KCAL}": sig(number(qc, f"dvte_mu / {KCAL}")),
                f"q_dvte_std_md / {KCAL}": sig(number(qc, f"dvte_std / {KCAL}")),
                f"dG_EnT / {KCAL}": sig(q_e_t - pc_e_t) if None not in (q_e_t, pc_e_t) else "",
                f"dG_SET_exp / {KCAL}": sig(set_exp),
                "dG_SET_exp_bound": set_bound,
                f"dG_SET_calc / {KCAL}": sig(set_calc),
            })

    rows.sort(key=lambda row: (row["pc_id"], row["quencher_id"]))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    blocks = [Block(title, [Column(name, description, source,
                                   lambda i, name=name: rows[i][name] if rows[i][name] != "" else None)
                            for name, description, source in columns])
              for title, columns in LAYOUT]
    write_workbook(OUT_XLSX, "Pairs", list(range(len(rows))), blocks, """
        One row per photocatalyst-quencher pair, sorted by pc_id then quencher_id. Built by
        data/build_exp_dataset.py; exp_dataset.csv is the same table.
        PC24 is excluded: discarded from the screen.
    """)

    n_pc = len({r["pc_id"] for r in rows})
    n_q = len({r["quencher_id"] for r in rows})
    print(f"{len(rows)} pairs ({n_pc} photocatalysts x {n_q} quenchers) -> "
          f"{OUT_CSV.name}, {OUT_XLSX.name}")
    for pc_id, reason in EXCLUDED_PHOTOCATALYSTS.items():
        print(f"  excluded PC{pc_id} ({excluded} pairs): {reason}")
    print(f"  quenches {sum(r['quenches'] for r in rows)}, "
          f"quenches_strict {sum(r['quenches_strict'] for r in rows)}, "
          f"censored {sum(r['censored'] for r in rows)}")
    for name in COLUMNS:
        missing = sum(1 for r in rows if r[name] == "")
        if missing and name != "dG_SET_exp_bound":
            print(f"  {name}: blank for {missing} pairs")


if __name__ == "__main__":
    main()
