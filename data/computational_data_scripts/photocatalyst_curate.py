# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Curate the photocatalyst ORCA jobs into the raw-data layout.

    uv run data/computational_data_scripts/photocatalyst_curate.py <source_dir>

`source_dir` is the delivered job tree: PC_<n>/<job>/output.out, 31 photocatalysts,
587 outputs, 895 MB, not in this repository. This script is the record of how the deposit
under `../computational_data_raw/photocatalysts/` was made from it, and writes that tree.

WHAT IS KEPT
    Eleven jobs per photocatalyst, not the other nine. The delivery carries parallel
    attempts at the same quantity, and choosing by name would be a guess. It is not one:
    every energy in `Photocatalysts/computed_PCs.xlsx` was matched against every candidate,
    and all 31 photocatalysts x 10 quantities resolved to the plain job to better than
    1e-6 Eh. No published number came from the suffixed runs: `_uks` (unrestricted repeat),
    `_m06` (M06-2X/def2-TZVP), `_lanl` (no ZORA), `_tddft` (TD-DFT triplet), `_slightlyold`
    and `_depr` (superseded).

    `conf` is kept although no energy comes from it: it is the GOAT/XTB2 conformer search
    that produced the starting geometry.

WHAT IS ADDED
    The delivered files are outputs only, but ORCA echoes its input verbatim and the last
    CARTESIAN COORDINATES block of an optimisation is the optimised geometry. So every job
    ends up complete -- `_input.inp`, `_geometry.xyz`, `_output.out` -- and readable on its
    own without parsing a 3 MB output. The two extracted files sit beside the output they
    came from, so nothing here is authored.

Old PC7, PC26, PC30 and PC31 are in the delivery but crossed out in the photocatalyst
sheet, and are not deposited (../compound_identification.py).
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "computational_data_raw" / "photocatalysts"

# The directories this script owns and may therefore delete on a re-run.
RE_PC = re.compile(r"PC_\d+")

# Delivered, but crossed out in the photocatalyst sheet and so not part of the data.
sys.path.insert(0, str(ROOT))
from compound_identification import CROSSED_OUT  # noqa: E402

EXCLUDED = set(CROSSED_OUT)

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

INPUT_HEADER = re.compile(r"^\s*INPUT FILE\s*$")
INPUT_LINE = re.compile(r"^\|\s*\d+>\s?(.*)$")
COORD_HEADER = "CARTESIAN COORDINATES (ANGSTROEM)"
COORD_LINE = re.compile(r"^\s*([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


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

    return inp, coords, terminated


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
        (d for d in source.iterdir() if d.is_dir() and d.name.startswith("PC_")
         and int(d.name.split("_")[1]) not in EXCLUDED),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not pc_dirs:
        sys.exit(f"no PC_<n> directories under {source}")

    # Clear only the PC_<n> trees this script owns.
    #
    # NOT the whole directory. `computational_data_raw/photocatalysts/` also holds
    # files that predate this script and that nothing else has a copy of -- the
    # directory is outside git, so anything removed here is gone. An earlier version
    # rmtree'd the lot, which would have destroyed them on the next run.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for existing in OUT_ROOT.iterdir():
        if existing.is_dir() and RE_PC.fullmatch(existing.name):
            shutil.rmtree(existing)

    kept_out = sorted(p.name for p in OUT_ROOT.iterdir() if p.is_file())
    if kept_out:
        print(f"  leaving in place, not written by this script: {', '.join(kept_out)}")

    kept = omitted = failed = 0

    for pc_dir in pc_dirs:
        pc = pc_dir.name
        print(f"  {pc} ...", flush=True)
        for job_dir in sorted(pc_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            job = job_dir.name

            if job not in LAYOUT:
                omitted += 1
                continue

            state, stem = LAYOUT[job]
            primary = job_dir / "output.out"
            if not primary.exists():
                print(f"  {pc}/{job}: no output.out in the source")
                continue

            inp, coords, terminated = read_output(primary)
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
            # Scratch outputs beside the real one (`output_tmp.out`, the
            # `input_atom<n>.out` basis-fitting runs) are not part of the job.
            kept += 1

    size = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file())
    print(f"\n{len(pc_dirs)} photocatalysts")
    print(f"  kept     {kept} jobs -> computational_data_raw/photocatalysts  ({size / 1e6:.0f} MB)")
    print(f"  omitted  {omitted} jobs, the suffixed runs listed under WHAT IS KEPT")
    if failed:
        print(f"  WARNING  {failed} kept jobs did not terminate normally")

    expected = set(LAYOUT)
    for pc_dir in pc_dirs:
        missing = sorted(expected - {d.name for d in pc_dir.iterdir() if d.is_dir()})
        if missing:
            print(f"  {pc_dir.name}: no {', '.join(missing)}")


if __name__ == "__main__":
    main()
