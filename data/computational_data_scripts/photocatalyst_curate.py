# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Curate the photocatalyst ORCA jobs into the raw-data layout.

    uv run data/computational_data_scripts/photocatalyst_curate.py <source_dir>

Reads   <source_dir>/PC_<old id>/<job>/output.out   (job tree, not in this repository)
        ../compound_identification.xlsx           (old -> new id)
Writes  ../computational_data_raw/photocatalysts_new_ids/PC<new id>/<state>/
            <stem>_output.out, <stem>_input.inp (echoed input),
            <stem>_geometry.xyz (last CARTESIAN COORDINATES block)

Only the jobs in LAYOUT are copied (suffixed runs are omitted); `conf` is the GOAT/XTB2
conformer search. Photocatalysts without a new id are skipped.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "computational_data_raw" / "photocatalysts_new_ids"

# Output directories cleared on a re-run.
RE_PC = re.compile(r"PC\d+")

sys.path.insert(0, str(ROOT))
from compound_identification import NEW_ID  # noqa: E402

# source job directory -> (state directory, file stem); a state directory holds its
# optimisation and the vertical single points at that geometry.
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
         and int(d.name.split("_")[1]) in NEW_ID),
        key=lambda d: NEW_ID[int(d.name.split("_")[1])],
    )
    if not pc_dirs:
        sys.exit(f"no PC_<n> directories under {source}")

    # Remove only the PC<n> trees.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for existing in OUT_ROOT.iterdir():
        if existing.is_dir() and RE_PC.fullmatch(existing.name):
            shutil.rmtree(existing)

    kept_out = sorted(p.name for p in OUT_ROOT.iterdir() if p.is_file())
    if kept_out:
        print(f"  leaving in place, not written by this script: {', '.join(kept_out)}")

    kept = omitted = failed = 0

    for pc_dir in pc_dirs:
        pc = f"PC{NEW_ID[int(pc_dir.name.split('_')[1])]}"
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
            # Scratch outputs (`output_tmp.out`, `input_atom<n>.out`) are not copied.
            kept += 1

    size = sum(f.stat().st_size for f in OUT_ROOT.rglob("*") if f.is_file())
    print(f"\n{len(pc_dirs)} photocatalysts")
    print(f"  kept     {kept} jobs -> {OUT_ROOT.relative_to(ROOT).as_posix()}  "
          f"({size / 1e6:.0f} MB)")
    print(f"  omitted  {omitted} jobs not in LAYOUT (suffixed runs)")
    if failed:
        print(f"  WARNING  {failed} kept jobs did not terminate normally")

    expected = set(LAYOUT)
    for pc_dir in pc_dirs:
        missing = sorted(expected - {d.name for d in pc_dir.iterdir() if d.is_dir()})
        if missing:
            print(f"  PC{NEW_ID[int(pc_dir.name.split('_')[1])]}: no {', '.join(missing)}")


if __name__ == "__main__":
    main()
