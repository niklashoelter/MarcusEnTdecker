# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.2"]
# ///
"""
Adiabatic triplet energies, redox potentials, reorganization energies and molecular radii
of the quenchers.

Reads   ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/<state>/
        ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/S0/opt_geometry.xyz
Writes  ../computational_data_processed/quencher_dft.csv

    uv run data/computational_data_scripts/quencher_dft.py

States S0, T1, ox, red (lowest-energy conformer each). Per state: `opt_output.out`
(optimisation + frequencies, G-E(el)), `sp_output.out` (single point) and
`sp_vert_<kind>_output.out` (other states at this geometry). In Q17/S0, `opt_output.out` and
`sp_output.out` are two runs of the same conformer; Q18/S0 `sp_vert_red_output.out` is a
second run.

    E_combined   E(sp) + G-E(el)(opt)
    E_T          E_combined(T1) - E_combined(S0)
    dG_Ox        E_combined(ox) - E_combined(S0)
    dG_Red       E_combined(red) - E_combined(S0)
    E_Ox vs SCE   dG_Ox / 23.061 - 4.42
    E_Red vs SCE  -dG_Red / 23.061 - 4.42
    Hartree to kcal/mol: x 627.509

lambda_<surface>_<geometry>, e.g. lambda_S0_T1 = E_S0 at the T1 geometry - E_S0. Dplus/Dminus:
one-electron oxidized/reduced states (`ox`/`red`). radius_S0: radius in A of the sphere with
the van der Waals volume of the S0 geometry (../molecular_size.py).

Computed from full-precision energies; kcal/mol, V and A rounded to 2 decimals when written.
"""

import csv
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "computational_data_raw" / "quencher_adiabatic_DFT"
OUT_DIR = DATA / "computational_data_processed"
OUT = OUT_DIR / "quencher_dft.csv"

H_TO_KCAL = 627.509          # Hartree -> kcal/mol
F_KCAL_PER_V = 23.061        # Faraday constant, kcal mol-1 V-1
# Absolute SCE potential in V, E = dG/F - 4.42.
SHE_TO_SCE = 4.42
DECIMALS = 2                 # for kcal/mol, V and A

sys.path.insert(0, str(DATA))
from molecular_size import radius  # noqa: E402

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
    "radius_S0",
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
    """A value in kcal/mol, V or A, to DECIMALS; blank if absent."""
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
            row["dG_Red"] = to_kcal(sub(red["E_combined"], s0["E_combined"]))
            row["E_Red_vs_SCE_calc"] = to_volt(None if row["dG_Red"] is None else -row["dG_Red"])
            row["lambda_S0_Dminus"] = to_kcal(sub(red["E_vert_ox"], s0["E_singlepoint"]))
            row["lambda_Dminus_S0"] = to_kcal(sub(s0["E_vert_red"], red["E_singlepoint"]))

        row["radius_S0"] = radius(qdir / "S0" / "opt_geometry.xyz")

        missing = [c for c in COLUMNS
                   if c.startswith(("E_", "lambda", "dG", "radius")) and row.get(c) is None]
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
