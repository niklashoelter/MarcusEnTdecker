# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Curate the photocatalyst ORCA jobs into the raw-data layout.

    uv run data/computational_data_scripts/photocatalyst_curate.py <source_dir>

`source_dir` is the delivered job tree: PC_<n>/<job>/output.out, 31 photocatalysts,
587 outputs, 895 MB. It is not in this repository -- this script is the record of how
the deposit under `../computational_data_raw/photocatalysts/` was made from it.

WHAT IS KEPT AND WHY
    Eleven jobs per photocatalyst, and not the other nine. The delivered tree carries
    several parallel attempts at the same quantity -- `_uks`, `_m06`, `_lanl`,
    `_slightlyold`, `_depr`, `_tddft` -- and picking between them by name would be a
    guess. It is not a guess: every energy in the analysis spreadsheet
    (`Photocatalysts/computed_PCs.xlsx`) was matched against every candidate job, and
    all 31 photocatalysts x 10 quantities resolved to the plain, non-suffixed job to
    better than 1e-6 Eh. The suffixed runs are alternative levels of theory that no
    published number came from, so they are omitted rather than deposited as noise.

    `conf` is kept even though no energy comes from it: it is the GOAT/XTB2 conformer
    search that produced the starting geometry, and without it the optimisations begin
    from nowhere.

WHAT IS ADDED
    The delivered files are ORCA outputs only -- no inputs, no geometries. Both are
    recoverable from the output itself: ORCA echoes its input file verbatim, and the
    last CARTESIAN COORDINATES block of an optimisation is the optimised geometry. Both
    are written out beside each job so the deposit matches the quencher tree, where the
    `.inp` and `.xyz` files were kept from the start. They are extracted, not authored.

Writes `../computational_data_raw/photocatalysts/` and a `manifest.csv` recording
every source directory, kept or omitted, with its termination status.
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "computational_data_raw" / "photocatalysts"

# The directories this script owns and may therefore delete on a re-run.
RE_PC = re.compile(r"PC_\d+")

# source job directory -> (state directory, file stem)
#
# The naming is the quencher tree's: a state directory holds its own optimisation and
# the single points taken *at that geometry*. The delivered `S_geometry_T_energy` is
# "the T1 energy at the S0 geometry", which is `S0/sp_vert_T1`; `ox_geometry_S_energy`
# is the S0 energy at the oxidised geometry, `ox/sp_vert_S0`. The same quantity is then
# said the same way in both halves of the dataset.
LAYOUT = {
    "conf": ("conf", "goat"),
    "singlet_opt": ("S0", "opt"),
    "S_geometry_T_energy": ("S0", "sp_vert_T1"),
    "S_geometry_ox_energy": ("S0", "sp_vert_ox"),
    "S_geometry_red_energy": ("S0", "sp_vert_red"),
    "triplet_opt": ("T1", "opt"),
    "T_geometry_S_energy": ("T1", "sp_vert_S0"),
    "ox_opt": ("ox", "opt"),
    "ox_geometry_S_energy": ("ox", "sp_vert_S0"),
    "red_opt": ("red", "opt"),
    "red_geometry_S_energy": ("red", "sp_vert_S0"),
}

# Why each omitted family is omitted, recorded per row rather than left to inference.
OMIT_REASON = {
    "_uks": "unrestricted repeat of the same level; no published number came from it",
    "_m06": "M06-2X/def2-TZVP, a different and non-relativistic level of theory",
    "_lanl": "wB97X-D4/def2-TZVP without ZORA, a different level of theory",
    "_slightlyold": "superseded earlier run",
    "_depr": "deprecated attempt",
    "_tddft": "TD-DFT triplet, not the adiabatic triplet the analysis uses",
}

INPUT_HEADER = re.compile(r"^\s*INPUT FILE\s*$")
INPUT_LINE = re.compile(r"^\|\s*\d+>\s?(.*)$")
COORD_HEADER = "CARTESIAN COORDINATES (ANGSTROEM)"
COORD_LINE = re.compile(r"^\s*([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
CHARGE_MULT = re.compile(r"\*\s*xyz(?:file)?\s+(-?\d+)\s+(\d+)")


def read_output(path):
    """One pass over an ORCA output: the echoed input, the last geometry, the status."""
    inp = []
    in_input = False
    coords = []
    current = []
    collecting = False
    terminated = 0

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not inp and INPUT_HEADER.match(line):
                in_input = True
                continue
            if in_input:
                match = INPUT_LINE.match(line)
                if match:
                    inp.append(match.group(1).rstrip())
                elif inp:
                    in_input = False

            if COORD_HEADER in line:
                collecting = True
                current = []
                continue
            if collecting:
                match = COORD_LINE.match(line)
                if match:
                    current.append(match.groups())
                elif current:
                    coords = current
                    collecting = False
                elif line.strip() and not line.lstrip().startswith("-"):
                    collecting = False

            if "ORCA TERMINATED NORMALLY" in line:
                terminated += 1

    if collecting and current:
        coords = current

    charge = mult = ""
    for line in inp:
        match = CHARGE_MULT.search(line)
        if match:
            charge, mult = match.groups()
            break

    return inp, coords, terminated, charge, mult


def write_xyz(path, coords, comment):
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{len(coords)}\n{comment}\n")
        for element, x, y, z in coords:
            fh.write(f"{element:<3}{float(x):>15.8f}{float(y):>15.8f}{float(z):>15.8f}\n")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: photocatalyst_curate.py <source_dir>")
    source = Path(sys.argv[1]).resolve()
    if not source.is_dir():
        sys.exit(f"not a directory: {source}")

    pc_dirs = sorted(
        (d for d in source.iterdir() if d.is_dir() and d.name.startswith("PC_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not pc_dirs:
        sys.exit(f"no PC_<n> directories under {source}")

    # Clear only the PC_<n> trees this script owns, and the manifest it writes.
    #
    # NOT the whole directory. `computational_data_raw/photocatalysts/` also holds
    # files that predate this script and that nothing else has a copy of -- the
    # directory is outside git, so anything removed here is gone. An earlier version
    # rmtree'd the lot, which would have destroyed them on the next run.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for existing in OUT_ROOT.iterdir():
        if existing.is_dir() and RE_PC.fullmatch(existing.name):
            shutil.rmtree(existing)
    (OUT_ROOT / "manifest.csv").unlink(missing_ok=True)

    kept_out = sorted(p.name for p in OUT_ROOT.iterdir() if p.is_file())
    if kept_out:
        print(f"  leaving in place, not written by this script: {', '.join(kept_out)}")

    manifest = []
    kept = omitted = failed = 0

    for pc_dir in pc_dirs:
        pc = pc_dir.name
        print(f"  {pc} ...", flush=True)
        for job_dir in sorted(pc_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            job = job_dir.name
            outputs = sorted(job_dir.glob("*.out"))

            if job not in LAYOUT:
                reason = next((r for s, r in OMIT_REASON.items() if s in job),
                              "not used by the analysis")
                omitted += 1
                manifest.append({
                    "pc_id": pc, "source_job": job, "kept": 0, "destination": "",
                    "terminated_normally": "", "n_atoms": "", "charge": "", "mult": "",
                    "note": reason,
                })
                continue

            state, stem = LAYOUT[job]
            primary = job_dir / "output.out"
            if not primary.exists():
                manifest.append({
                    "pc_id": pc, "source_job": job, "kept": 0, "destination": "",
                    "terminated_normally": "", "n_atoms": "", "charge": "", "mult": "",
                    "note": "no output.out in the source",
                })
                continue

            inp, coords, terminated, charge, mult = read_output(primary)
            if not terminated:
                failed += 1

            dest = OUT_ROOT / pc / state
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary, dest / f"{stem}_output.out")
            if inp:
                (dest / f"{stem}_input.inp").write_text(
                    "\n".join(inp) + "\n", encoding="utf-8", newline="\n")
            if coords:
                write_xyz(dest / f"{stem}_geometry.xyz", coords, f"{pc} {state} {stem}")

            # Scratch outputs that sit beside the real one (`output_tmp.out`, the
            # `input_atom<n>.out` basis-fitting runs) are not part of the job.
            extras = [p.name for p in outputs if p.name != "output.out"]

            kept += 1
            manifest.append({
                "pc_id": pc, "source_job": job, "kept": 1,
                "destination": f"{pc}/{state}/{stem}_output.out",
                "terminated_normally": terminated, "n_atoms": len(coords),
                "charge": charge, "mult": mult,
                "note": ("also in the source, not copied: " + ", ".join(extras)) if extras else "",
            })

    manifest_path = OUT_ROOT / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    size = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file())
    print(f"\n{len(pc_dirs)} photocatalysts")
    print(f"  kept     {kept} jobs -> computational_data_raw/photocatalysts  ({size / 1e6:.0f} MB)")
    print(f"  omitted  {omitted} jobs, each with a reason in manifest.csv")
    if failed:
        print(f"  WARNING  {failed} kept jobs did not terminate normally")

    expected = set(LAYOUT)
    for pc_dir in pc_dirs:
        missing = sorted(expected - {d.name for d in pc_dir.iterdir() if d.is_dir()})
        if missing:
            print(f"  {pc_dir.name}: no {', '.join(missing)}")


if __name__ == "__main__":
    main()
