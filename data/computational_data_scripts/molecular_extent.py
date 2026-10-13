# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2"]
# ///
"""
Molecular extent radii of the photocatalysts and quenchers.

Reads   ../computational_data_raw/photocatalysts/PC_<n>/S0/opt_geometry.xyz
        ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/S0/opt_geometry.xyz
        ../compound_identification.xlsx   (photocatalyst and quencher ids)
Writes  ../computational_data_processed/molecular_extent.csv

    uv run data/computational_data_scripts/molecular_extent.py

The extent radius of a molecule is half its largest atom-atom extent including the van
der Waals radii of the two atoms,

    r_extent = max_ij (d_ij + R_i + R_j) / 2

with d_ij the distance of the nuclei in the optimised S0 geometry and R the Bondi radii of
../molecular_size.py. It is an alternative to the volume-equivalent radius radius_S0 of
photocatalyst_dft.py and quencher_dft.py, for which the benchmark compares the contact
distance r_PC + r_Q.

Photocatalysts are numbered by old id; those excluded from the data are skipped
(../compound_identification.py). The radius is rounded to 2 decimals in A when written.
"""

import csv
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "computational_data_raw"
OUT = DATA / "computational_data_processed" / "molecular_extent.csv"
DECIMALS = 2                    # A

sys.path.insert(0, str(DATA))
from compound_identification import IN_DATA, QUENCHERS  # noqa: E402
from molecular_size import BONDI, read_xyz  # noqa: E402

COLUMNS = ["kind", "id", "n_atoms", "r_extent / A"]


def extent_radius(elements, xyz) -> float:
    """max_ij (d_ij + R_i + R_j) / 2 in A, with Bondi radii R."""
    radii = np.array([BONDI[e] for e in elements])
    distance = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    return float(((distance + radii[:, None] + radii[None, :]) / 2.0).max())


def geometries():
    """(kind, id, path of the S0 geometry) of every photocatalyst in the data and every
    quencher."""
    for old_id in sorted(IN_DATA):
        yield "photocatalyst", old_id, RAW / "photocatalysts" / f"PC_{old_id}" / "S0" / "opt_geometry.xyz"
    for quencher_id in sorted(QUENCHERS):
        yield ("quencher", quencher_id,
               RAW / "quencher_adiabatic_DFT" / f"Q{quencher_id}" / "S0" / "opt_geometry.xyz")


def main():
    rows, missing = [], []
    for kind, number, path in geometries():
        if not path.exists():
            missing.append(str(path.relative_to(DATA)))
            continue
        elements, xyz = read_xyz(path)
        rows.append({"kind": kind, "id": number, "n_atoms": len(elements),
                     "r_extent / A": f"{extent_radius(elements, xyz):.{DECIMALS}f}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for kind in ("photocatalyst", "quencher"):
        values = [float(r["r_extent / A"]) for r in rows if r["kind"] == kind]
        print(f"{kind:14s} {len(values):3d}  r_extent {min(values):.2f}-{max(values):.2f} A")
    for path in missing:
        print(f"  no geometry: {path}")
    print(f"-> {OUT.relative_to(DATA)}")


if __name__ == "__main__":
    main()
