# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Quencher ORCA outputs -> adiabatic triplet energy, redox potentials, reorganization energies.

Reads   ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/<state>/
Writes  ../computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv

48 quenchers, four states each -- S0, T1, ox, red -- one optimisation and one set of
single points per state, in the photocatalysts' layout.

WHAT IS IN THE DEPOSIT
    Only the calculation each published number came from. The delivery held several
    conformers per state, and a few states twice from parallel pipelines; what is kept is
    the conformer that won on energy, with no conformer or pipeline number in the names.
    So nothing chooses between candidates at read time -- there are none left.

    Two folders are composites, because that is where the published value comes from:
    Q17's S0 takes its thermal correction and Q18's S0 its reduced-state vertical single
    point from a second run of the same conformer. In those two, `opt_output.out` is not
    the run `sp_output.out` came from.

PROVENANCE
    Port of Quencher/Adiabatic DFT Pipeline 5/extract_parameters_T1.py, which produced the
    published values. Its roundings are kept deliberately -- 1 decimal in kcal/mol, 2 in V
    -- since they define those numbers; this is the one script that does not follow the
    six-significant-figure rule.

    Its `_sim` reorganization energies, taken from the conformer closest in geometry to
    the lowest-energy S0 rather than the lowest in energy, are not computed: nothing reads
    them, and their published values could not be reproduced from the geometries.
"""

import csv
import re
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "computational_data_raw" / "quencher_adiabatic_DFT"
OUT_DIR = RAW
OUT = OUT_DIR / "quencher_dft.csv"

H_TO_KCAL = 627.509          # Hartree -> kcal/mol
F_KCAL_PER_V = 23.061        # kcal mol-1 V-1, as used by the original conversion
# V offset in E = dG/F - 4.42, the absolute SCE potential. The original used 4.442 here but
# 4.42 for the photocatalysts (photocatalyst_dft.py); harmonized to 4.42 so that both sides
# of a SET driving force are on the same reference.
SHE_TO_SCE = 4.42

STATES = ("S0", "T1", "ox", "red")

# which vertical single points exist for each state
VERTICALS = {
    "S0": ("spin_change", "ox", "red"),
    "T1": ("spin_change",),
    "ox": ("red",),
    "red": ("ox",),
}

COLUMNS = [
    "quencher_id",
    "E_T", "lambda_S_T", "lambda_T_S",
    "dG_Ox", "E_Ox_vs_SCE_calc", "lambda_S0_D+", "lambda_D+_S0",
    "dG_Red", "E_Red_vs_SCE_calc", "lambda_S0_D-", "lambda_D-_S0",
]


def sub(a, b):
    """Difference of two energies, or None when either is missing."""
    return None if (a is None or b is None) else a - b


def to_kcal(hartree):
    """Hartree difference -> kcal/mol, rounded as the original did."""
    return None if hartree is None else round(hartree * H_TO_KCAL, 1)


def to_volt(kcal):
    """kcal/mol -> V vs SCE, rounded as the original did."""
    return None if kcal is None else round((kcal / F_KCAL_PER_V) - SHE_TO_SCE, 2)


def final_single_point(path):
    """Last 'FINAL SINGLE POINT ENERGY' in an ORCA output, in Hartree."""
    if not path.exists():
        return None
    value = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            try:
                value = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    return value


def thermal_correction(path):
    """Last 'G-E(el)' in an ORCA output, in Hartree.

    The line reads:  G-E(el)   ...   0.13127235 Eh   82.37 kcal/mol
    """
    if not path.exists():
        return None
    value = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "G-E(el)" in line:
            fields = line.split()
            if "Eh" in fields:
                try:
                    value = float(fields[fields.index("Eh") - 1])
                except (ValueError, IndexError):
                    pass
    return value


def read_states(quencher_dir):
    """state -> {energies...} for one quencher. A state with no jobs is absent."""
    out = {}
    for state in STATES:
        state_dir = quencher_dir / state
        if not state_dir.is_dir():
            continue
        entry = {
            "E_singlepoint": final_single_point(state_dir / "sp_output.out"),
            "E_correction": thermal_correction(state_dir / "opt_output.out"),
        }
        for kind in VERTICALS[state]:
            entry[f"E_vert_{kind}"] = final_single_point(
                state_dir / f"sp_vert_{kind}_output.out")
        sp, corr = entry["E_singlepoint"], entry["E_correction"]
        entry["E_combined"] = None if (sp is None or corr is None) else sp + corr
        if entry["E_combined"] is not None:
            out[state] = entry
    return out


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input directory: {RAW}")

    quencher_dirs = sorted(
        (d for d in RAW.iterdir() if d.is_dir() and re.fullmatch(r"Q\d+", d.name)),
        key=lambda d: int(d.name[1:]))
    if not quencher_dirs:
        sys.exit(f"no Q<n> directories in {RAW}")

    rows, incomplete = [], []

    for qdir in quencher_dirs:
        qid = int(qdir.name[1:])
        states = read_states(qdir)

        row = {"quencher_id": qid}
        s0, t1 = states.get("S0"), states.get("T1")
        ox, red = states.get("ox"), states.get("red")

        if s0 and t1:
            row["E_T"] = to_kcal(sub(t1["E_combined"], s0["E_combined"]))
            row["lambda_S_T"] = to_kcal(sub(t1["E_vert_spin_change"], s0["E_singlepoint"]))
            row["lambda_T_S"] = to_kcal(sub(s0["E_vert_spin_change"], t1["E_singlepoint"]))

        if s0 and ox:
            row["dG_Ox"] = to_kcal(sub(ox["E_combined"], s0["E_combined"]))
            row["E_Ox_vs_SCE_calc"] = to_volt(row["dG_Ox"])
            row["lambda_S0_D+"] = to_kcal(sub(ox["E_vert_red"], s0["E_singlepoint"]))
            row["lambda_D+_S0"] = to_kcal(sub(s0["E_vert_ox"], ox["E_singlepoint"]))

        if s0 and red:
            dg = to_kcal(sub(red["E_combined"], s0["E_combined"]))
            row["dG_Red"] = None if dg is None else -1.0 * dg
            row["E_Red_vs_SCE_calc"] = to_volt(row["dG_Red"])
            row["lambda_S0_D-"] = to_kcal(sub(red["E_vert_ox"], s0["E_singlepoint"]))
            row["lambda_D-_S0"] = to_kcal(sub(s0["E_vert_red"], red["E_singlepoint"]))

        missing = [c for c in COLUMNS
                   if c.startswith(("E_", "lambda", "dG")) and row.get(c) is None]
        if missing:
            incomplete.append((qid, missing))
        rows.append({c: ("" if row.get(c) is None else row.get(c)) for c in COLUMNS})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} quenchers -> {OUT.name}")
    if incomplete:
        print(f"  quenchers with missing values: {len(incomplete)}")
        for qid, missing in incomplete:
            print(f"    Q{qid}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
