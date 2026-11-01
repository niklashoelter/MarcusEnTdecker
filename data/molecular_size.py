"""
Molecular radius from a DFT geometry, for the contact distance r_PC + r_Q of a pair.

V = volume of the union of atomic spheres with Bondi radii (BONDI; Ir and Ru set by hand)
on a cubic grid of spacing GRID; r = (3V / 4 pi)^(1/3).
"""

from pathlib import Path

import numpy as np

BONDI = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "S": 1.80,
         "Se": 1.90, "Br": 1.85, "Ir": 2.00, "Ru": 2.00}    # A
GRID = 0.1                                                  # A


def read_xyz(path: Path):
    """(elements, coordinates in A) of an .xyz file."""
    lines = path.read_text().splitlines()
    rows = [line.split() for line in lines[2:2 + int(lines[0])]]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def vdw_volume(elements, xyz, grid=GRID) -> float:
    """Volume in A3 of the union of the atomic van der Waals spheres, on a cubic grid."""
    radii = np.array([BONDI[e] for e in elements])
    low = (xyz - radii[:, None]).min(axis=0) - grid
    high = (xyz + radii[:, None]).max(axis=0) + grid
    shape = np.ceil((high - low) / grid).astype(int) + 1
    axes = [low[k] + grid * np.arange(shape[k]) for k in range(3)]
    occupied = np.zeros(shape, bool)
    for centre, r in zip(xyz, radii):
        box = [slice(max(0, int((centre[k] - r - low[k]) / grid)),
                     min(shape[k], int((centre[k] + r - low[k]) / grid) + 2)) for k in range(3)]
        gx, gy, gz = np.meshgrid(*(axes[k][box[k]] for k in range(3)), indexing="ij")
        occupied[box[0], box[1], box[2]] |= ((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2
                                             + (gz - centre[2]) ** 2 <= r * r)
    return int(occupied.sum()) * grid ** 3


def sphere_radius(volume) -> float:
    """Radius of the sphere of the given volume, (3V / 4 pi)^(1/3)."""
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def radius(path: Path) -> float | None:
    """Volume-equivalent sphere radius in A of the geometry in an .xyz file; None if the
    file is absent."""
    if not path.exists():
        return None
    return sphere_radius(vdw_volume(*read_xyz(path)))
