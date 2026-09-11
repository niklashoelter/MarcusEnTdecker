# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Quencher ORCA outputs -> adiabatic triplet energy, redox potentials, reorganization energies.

Reads   ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/<state>/
Writes  ../computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv             one row per quencher
        ../computational_data_raw/quencher_adiabatic_DFT/quencher_dft_conformers.csv  one row per conformer

48 quenchers, four states each: S0, T1, ox, red.

PROVENANCE
    The per-quencher arithmetic is a port of
    Quencher/Adiabatic DFT Pipeline 5/extract_parameters_T1.py, which produced the
    published values. That script began from a spreadsheet of per-conformer energies; the
    step that built that spreadsheet from the ORCA outputs was never in the tree, so it is
    implemented here and written out as the conformer table.

    The original's roundings are preserved deliberately, because they define the published
    numbers: 1 decimal in kcal/mol, 2 decimals in V. This is the one place in the data
    package that does not follow the six-significant-figure rule.

    The original also produced `_sim` variants of every reorganization energy, using the
    conformer geometrically closest to the lowest-energy S0 structure rather than the
    lowest-energy one. Those columns are not computed here: nothing in the project reads
    them, neither assembled dataset carries them, and their published values could not be
    reproduced from the geometries.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "computational_data_raw" / "quencher_adiabatic_DFT"
OUT_DIR = RAW
OUT = OUT_DIR / "quencher_dft.csv"
OUT_CONF = OUT_DIR / "quencher_dft_conformers.csv"

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

# Where a rerun, not the earliest calculation, supplies a quantity. Established by testing
# which run reproduces the published table, not by a general rule -- see the note column of
# computational_data_raw/quencher_adiabatic_DFT/provenance.csv. Q17's
# S0 is the only case: its published dG_Red needs calc3's thermal correction, while every
# other published Q17 quantity needs calc1's single points.
RERUN_FIELD_SOURCE = {
    (17, "S0", "E_correction"): "calc3_conf1",
}

CONF_COLUMNS = [
    "quencher_id", "state", "conformer", "runs", "lowest_conformer",
    "E_singlepoint", "E_correction", "E_combined",
    "E_vert_spin_change", "E_vert_ox", "E_vert_red",
]

COLUMNS = [
    "quencher_id",
    "E_T", "lambda_S_T", "lambda_T_S",
    "dG_Ox", "E_Ox_vs_SCE_calc", "lambda_S0_D+", "lambda_D+_S0",
    "dG_Red", "E_Red_vs_SCE_calc", "lambda_S0_D-", "lambda_D-_S0",
    "n_conformers_S0", "n_conformers_T1", "n_conformers_ox", "n_conformers_red",
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
    so the Hartree number is the field before 'Eh'.
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


def group_reruns(prefixes):
    """conformer -> [job prefixes, earliest calculation first].

    A few quenchers were computed more than once and their jobs carry a calculation prefix
    as well as a conformer number: Q17's S0 holds `calc1_conf1` and `calc3_conf1`, Q18's
    holds `calc1_conf1` and `calc5_conf1`. These are one conformer computed twice, not two
    conformers, and must not compete against each other for lowest energy.
    """
    groups = defaultdict(list)
    for prefix in prefixes:
        m = re.fullmatch(r"calc(\d+)_(conf\d+)", prefix)
        key = m.group(2) if m else prefix
        order = int(m.group(1)) if m else -1
        groups[key].append((order, prefix))
    return {conf: [p for _, p in sorted(runs)] for conf, runs in groups.items()}


def read_conformers(quencher_dir, quencher_id):
    """state -> {conformer: {energies...}} for one quencher.

    A rerun fills gaps rather than replacing: the earliest calculation supplies the values
    and any quantity it lacks is taken from the next run that has it. Q18 is the case that
    fixes this -- its calc1 never ran the reduced-state vertical single point, which
    lambda_D-_S0 needs, while every other published quantity for it follows calc1.
    """
    out = {}
    for state in STATES:
        state_dir = quencher_dir / state
        if not state_dir.is_dir():
            continue
        prefixes = sorted(f.name[:-len("_opt_output.out")]
                          for f in state_dir.glob("*_opt_output.out"))
        entries = {}
        for conf, runs in group_reruns(prefixes).items():
            entry = {name: None for name in
                     ("E_singlepoint", "E_correction",
                      "E_vert_spin_change", "E_vert_ox", "E_vert_red")}
            entry["runs"] = " ".join(runs)

            for prefix in runs:            # earliest first; later runs only fill gaps
                values = {
                    "E_singlepoint": final_single_point(state_dir / f"{prefix}_sp_output.out"),
                    "E_correction": thermal_correction(state_dir / f"{prefix}_opt_output.out"),
                }
                for kind in VERTICALS[state]:
                    values[f"E_vert_{kind}"] = final_single_point(
                        state_dir / f"{prefix}_sp_vert_{kind}_output.out")
                for name, value in values.items():
                    forced = RERUN_FIELD_SOURCE.get((quencher_id, state, name))
                    if forced is not None and forced != prefix:
                        continue           # this quantity is taken from a named rerun only
                    if (entry[name] is None or forced == prefix) and value is not None:
                        entry[name] = value

            sp, corr = entry["E_singlepoint"], entry["E_correction"]
            entry["E_combined"] = None if (sp is None or corr is None) else sp + corr
            entries[conf] = entry
        if entries:
            out[state] = entries
    return out


def lowest(entries):
    """Conformer with the smallest E_combined, ignoring incomplete ones."""
    usable = {k: e for k, e in entries.items()
              if e["E_combined"] is not None and e["E_singlepoint"] and e["E_correction"]}
    return min(usable, key=lambda k: usable[k]["E_combined"]) if usable else None


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input directory: {RAW}")

    quencher_dirs = sorted(
        (d for d in RAW.iterdir() if d.is_dir() and re.fullmatch(r"Q\d+", d.name)),
        key=lambda d: int(d.name[1:]))
    if not quencher_dirs:
        sys.exit(f"no Q<n> directories in {RAW}")

    rows, conf_rows, incomplete = [], [], []

    for qdir in quencher_dirs:
        qid = int(qdir.name[1:])
        states = read_conformers(qdir, qid)
        mins = {s: lowest(e) for s, e in states.items()}

        for state, entries in states.items():
            for conf, entry in entries.items():
                conf_rows.append({
                    "quencher_id": qid, "state": state, "conformer": conf,
                    "runs": entry["runs"],
                    "lowest_conformer": int(mins.get(state) == conf),
                    **{c: entry[c] for c in
                       ("E_singlepoint", "E_correction", "E_combined",
                        "E_vert_spin_change", "E_vert_ox", "E_vert_red")},
                })

        row = {"quencher_id": qid}
        row.update({f"n_conformers_{s}": len(states.get(s, {})) for s in STATES})

        s0 = states.get("S0", {}).get(mins.get("S0"))
        t1 = states.get("T1", {}).get(mins.get("T1"))
        ox = states.get("ox", {}).get(mins.get("ox"))
        red = states.get("red", {}).get(mins.get("red"))

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
    with OUT_CONF.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONF_COLUMNS)
        writer.writeheader()
        writer.writerows([{c: ("" if r.get(c) is None else r.get(c)) for c in CONF_COLUMNS}
                          for r in conf_rows])

    print(f"{len(rows)} quenchers, {len(conf_rows)} conformers -> {OUT.name}, {OUT_CONF.name}")
    reruns = sum(1 for r in conf_rows if " " in str(r["runs"]))
    print(f"  conformers computed more than once: {reruns}")
    if incomplete:
        print(f"  quenchers with missing values: {len(incomplete)}")
        for qid, missing in incomplete:
            print(f"    Q{qid}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
