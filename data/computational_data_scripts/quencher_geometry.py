# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "rdkit==2025.9.4"]
# ///
"""
Change of the quencher geometries from the S0 to the T1 minimum.

Reads   ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/S0/opt_geometry.xyz
        ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/T1/opt_geometry.xyz
        ../compound_identification.xlsx   (quencher ids and SMILES)
Writes  ../computational_data_processed/quencher_geometry_S0_T1.csv

    uv run data/computational_data_scripts/quencher_geometry.py

The bonds are those of the S0 geometry (RDKit connectivity from the coordinates); the
bond orders come from the SMILES (`note` says where that fails, and every bond then
counts as single). Atoms are labelled by element and 0-based index in the .xyz file.

    max_stretch_pct       largest relative change of an S0 bond length in T1, in %
                          (signed; the bond with the largest absolute change), with the
                          bond and its lengths in A in stretch_bond
    dtors_double_deg      largest change about a non-aromatic double bond of the
                          deviation from coplanarity, in degrees, bond in double_bond
    dtors_any_heavy_deg   the same about any bond between two heavy atoms that each have
                          another heavy neighbour, bond in any_bond
    rmsd_heavy            heavy-atom RMSD between S0 and T1 after Kabsch alignment, in A

Deviation from coplanarity about a bond i-j: for every pair of substituents p of i and
s of j, the dihedral p-i-j-s folded to 0-90 degrees (min(|t|, 180 - |t|)); the median
over the pairs, at the S0 and at the T1 geometry. The change is the absolute difference
of the two medians. For dtors_double_deg every substituent counts, for
dtors_any_heavy_deg only heavy ones, so that methyl rotors do not enter. The labels give
the two medians, S0 -> T1.

Percentages and angles are rounded to 1 decimal, the RMSD to 3 decimals in A.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "computational_data_raw" / "quencher_adiabatic_DFT"
OUT = DATA / "computational_data_processed" / "quencher_geometry_S0_T1.csv"

sys.path.insert(0, str(DATA))
from compound_identification import QUENCHERS  # noqa: E402
from molecular_size import read_xyz  # noqa: E402

RDLogger.DisableLog("rdApp.warning")    # "more than one matching pattern" of the template match

COLUMNS = ["id", "max_stretch_pct", "stretch_bond", "dtors_double_deg", "double_bond",
           "dtors_any_heavy_deg", "any_bond", "rmsd_heavy", "note"]


def dihedral(p0, p1, p2, p3) -> float:
    """Dihedral angle p0-p1-p2-p3 in degrees, -180 to 180."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def kabsch_rmsd(P, Q) -> float:
    """RMSD in A of two coordinate sets after optimal superposition."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def coplanarity(x, i, j, left, right) -> float:
    """Median over substituent pairs of the dihedral about i-j, folded to 0-90 degrees."""
    folded = [min(abs(t), 180.0 - abs(t))
              for t in (dihedral(x[p], x[i], x[j], x[s]) for p in left for s in right)]
    return float(np.median(folded))


def bonded(elements, x0, smiles):
    """(molecule with the S0 bonds, note): bond orders from the SMILES where possible."""
    block = f"{len(elements)}\n\n" + "\n".join(
        f"{e} {a:.6f} {b:.6f} {c:.6f}" for e, (a, b, c) in zip(elements, x0))
    m = Chem.MolFromXYZBlock(block)
    rdDetermineBonds.DetermineConnectivity(m)
    try:
        m = AllChem.AssignBondOrdersFromTemplate(Chem.AddHs(Chem.MolFromSmiles(smiles)), m)
        Chem.Kekulize(m, clearAromaticFlags=False)
    except Exception as error:  # noqa: BLE001  (any failure leaves the bonds single)
        return m, f"bond orders not assigned ({type(error).__name__})"
    return m, ""


def geometry_change(elements, x0, x1, smiles) -> dict:
    """The columns of COLUMNS except id for one quencher."""
    m, note = bonded(elements, x0, smiles)

    stretch, stretch_bond = 0.0, ""
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d0, d1 = np.linalg.norm(x0[i] - x0[j]), np.linalg.norm(x1[i] - x1[j])
        if abs(d1 / d0 - 1) > abs(stretch):
            stretch = d1 / d0 - 1
            stretch_bond = f"{elements[i]}{i}-{elements[j]}{j} {d0:.2f}->{d1:.2f}"

    double, double_bond, heavy, heavy_bond = 0.0, "", 0.0, ""
    for b in m.GetBonds():
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if a.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
            continue
        i, j = a.GetIdx(), c.GetIdx()
        left = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != j]
        right = [n.GetIdx() for n in c.GetNeighbors() if n.GetIdx() != i]
        if not left or not right:
            continue
        left_heavy = [n for n in left if m.GetAtomWithIdx(n).GetAtomicNum() > 1]
        right_heavy = [n for n in right if m.GetAtomWithIdx(n).GetAtomicNum() > 1]
        is_double = b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
        if is_double:
            t0, t1 = coplanarity(x0, i, j, left, right), coplanarity(x1, i, j, left, right)
            if abs(t1 - t0) > double:
                double = abs(t1 - t0)
                double_bond = f"{elements[i]}{i}={elements[j]}{j} {t0:.0f}->{t1:.0f}"
        if left_heavy and right_heavy:
            t0 = coplanarity(x0, i, j, left_heavy, right_heavy)
            t1 = coplanarity(x1, i, j, left_heavy, right_heavy)
            if abs(t1 - t0) > heavy:
                heavy = abs(t1 - t0)
                heavy_bond = (f"{elements[i]}{i}{'=' if is_double else '-'}{elements[j]}{j}"
                              f"{' ring' if b.IsInRing() else ''} {t0:.0f}->{t1:.0f}")

    heavy_atoms = [k for k, e in enumerate(elements) if e != "H"]
    return {"max_stretch_pct": round(100 * stretch, 1), "stretch_bond": stretch_bond,
            "dtors_double_deg": round(double, 1), "double_bond": double_bond,
            "dtors_any_heavy_deg": round(heavy, 1), "any_bond": heavy_bond,
            "rmsd_heavy": round(kabsch_rmsd(x0[heavy_atoms], x1[heavy_atoms]), 3), "note": note}


def main():
    rows, missing = [], []
    for quencher_id, record in QUENCHERS.items():
        paths = [RAW / f"Q{quencher_id}" / state / "opt_geometry.xyz" for state in ("S0", "T1")]
        if not all(p.exists() for p in paths):
            missing += [str(p.relative_to(DATA)) for p in paths if not p.exists()]
            continue
        (e0, x0), (e1, x1) = (read_xyz(p) for p in paths)
        if e0 != e1:
            raise ValueError(f"Q{quencher_id}: S0 and T1 geometries list different atoms")
        rows.append({"id": f"Q{quencher_id}", **geometry_change(e0, x0, x1, record["smiles"])})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} quenchers; max |stretch| {max(abs(r['max_stretch_pct']) for r in rows):.1f} %, "
          f"max torsion change {max(r['dtors_any_heavy_deg'] for r in rows):.1f} deg")
    for path in missing:
        print(f"  no geometry: {path}")
    print(f"-> {OUT.relative_to(DATA)}")


if __name__ == "__main__":
    main()
