# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Adiabatic triplet energies, redox potentials and reorganization energies of the quenchers.

Reads   ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/<state>/
Writes  ../computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv

    uv run data/computational_data_scripts/quencher_dft.py

Each of the 48 quenchers has four states, S0, T1, ox and red. A state folder holds the
optimisation with frequencies (`opt_output.out`, thermal correction G-E(el)), the single
point at that geometry (`sp_output.out`) and the vertical single points of the other
states at that geometry (`sp_vert_<kind>_output.out`). Each state is the lowest-energy
conformer. In Q17/S0 and Q18/S0, `opt_output.out` and `sp_output.out` come from two runs
of the same conformer.

    E_combined   E(sp) + G-E(el)(opt)
    E_T          E_combined(T1) - E_combined(S0)
    dG_Ox        E_combined(ox) - E_combined(S0)
    dG_Red       -(E_combined(red) - E_combined(S0))
    E vs SCE     dG / 23.061 - 4.42
    Hartree to kcal/mol: x 627.509

Reorganization energies are named lambda_<surface>_<geometry>: lambda_S0_T1 is E_S0 at the
T1 geometry minus E_S0 at its own. Dplus and Dminus are the one-electron oxidized and
reduced states, the `ox` and `red` folders.

Every value is computed from the full-precision energies, potentials from the unrounded
free energies, and rounded to 2 decimals in kcal/mol and V when written.
"""

import csv
import re
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "computational_data_raw" / "quencher_adiabatic_DFT"
OUT_DIR = RAW
OUT = OUT_DIR / "quencher_dft.csv"

H_TO_KCAL = 627.509          # Hartree -> kcal/mol
F_KCAL_PER_V = 23.061        # Faraday constant, kcal mol-1 V-1
# Absolute SCE potential in V, E = dG/F - 4.42. Same value as in photocatalyst_dft.py.
SHE_TO_SCE = 4.42
DECIMALS = 2                 # for kcal/mol and V

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
    "E_T", "lambda_S0_T1", "lambda_T1_S0",
    "dG_Ox", "E_Ox_vs_SCE_calc", "lambda_S0_Dplus", "lambda_Dplus_S0",
    "dG_Red", "E_Red_vs_SCE_calc", "lambda_S0_Dminus", "lambda_Dminus_S0",
]


def sub(a, b):
    """Difference of two energies, or None when either is missing."""
    return None if (a is None or b is None) else a - b


def to_kcal(hartree):
    """Hartree difference -> kcal/mol."""
    return None if hartree is None else hartree * H_TO_KCAL


def to_volt(kcal):
    """kcal/mol -> V vs SCE."""
    return None if kcal is None else (kcal / F_KCAL_PER_V) - SHE_TO_SCE


def rounded(x):
    """A value in kcal/mol or V, to DECIMALS; blank if absent."""
    return "" if x is None else round(x, DECIMALS)


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
            row["lambda_S0_T1"] = to_kcal(sub(t1["E_vert_spin_change"], s0["E_singlepoint"]))
            row["lambda_T1_S0"] = to_kcal(sub(s0["E_vert_spin_change"], t1["E_singlepoint"]))

        if s0 and ox:
            row["dG_Ox"] = to_kcal(sub(ox["E_combined"], s0["E_combined"]))
            row["E_Ox_vs_SCE_calc"] = to_volt(row["dG_Ox"])
            row["lambda_S0_Dplus"] = to_kcal(sub(ox["E_vert_red"], s0["E_singlepoint"]))
            row["lambda_Dplus_S0"] = to_kcal(sub(s0["E_vert_ox"], ox["E_singlepoint"]))

        if s0 and red:
            dg = to_kcal(sub(red["E_combined"], s0["E_combined"]))
            row["dG_Red"] = None if dg is None else -1.0 * dg
            row["E_Red_vs_SCE_calc"] = to_volt(row["dG_Red"])
            row["lambda_S0_Dminus"] = to_kcal(sub(red["E_vert_ox"], s0["E_singlepoint"]))
            row["lambda_Dminus_S0"] = to_kcal(sub(s0["E_vert_red"], red["E_singlepoint"]))

        missing = [c for c in COLUMNS
                   if c.startswith(("E_", "lambda", "dG")) and row.get(c) is None]
        if missing:
            incomplete.append((qid, missing))
        rows.append({"quencher_id": qid,
                     **{c: rounded(row.get(c)) for c in COLUMNS if c != "quencher_id"}})

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
