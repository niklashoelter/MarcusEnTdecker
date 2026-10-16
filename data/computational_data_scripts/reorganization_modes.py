# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "scipy==1.18.1"]
# ///
"""
Reorganization energies of the photocatalysts and quenchers resolved into normal modes.

Reads   ../computational_data_raw/photocatalysts_new_ids/PC<n>/{S0,T1}/opt_geometry.xyz, opt_output.out
        ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/{S0,T1}/opt_geometry.xyz, opt_output.out
Writes  ../computational_data_processed/reorganization_modes.csv

    uv run data/computational_data_scripts/reorganization_modes.py

For a molecule on surface X (S0 or T1) with the other minimum at Y, the displacement Y - X
is projected onto the normal modes of X from its frequency calculation, Q_j = L_j . dx,
and each mode carries lambda_j = omega_j^2 Q_j^2 / 2. Surface S0 gives lambda_S0_T1 (the S0
surface at the T1 geometry), surface T1 gives lambda_T1_S0. Two displacements are used:

    internal    the change of the redundant internal coordinates (bonds, angles and
                dihedrals, dihedral changes wrapped to (-pi, pi]) mapped to mass-weighted
                Cartesian displacements, dx = pinv(B M^-1/2) dq, with B the Wilson matrix
                at X (numerical derivatives)
    cartesian   the mass-weighted Cartesian difference after a mass-weighted Kabsch
                superposition of Y on X

Of the symmetry-equivalent versions of Y (as given, its mirror image, and permutations of
equal elements that keep the bond graph), the one with the smallest internal sum is used.

Per molecule, surface and displacement the table gives the sum of the lambda_j and, with
the modes split at 3 k_B T at 298.15 K (621.7 cm-1), the fraction of the sum in the modes
below the cut (classical) and the lambda-weighted mean frequency of the modes at or above it
(quantum). The sum is rounded to 2 decimals in kcal/mol when written, the fraction to 6
decimals and the frequency to 2 decimals in cm-1. Photocatalysts are numbered by new id.
"""

import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "computational_data_raw"
OUT = DATA / "computational_data_processed" / "reorganization_modes.csv"

AMU, AVOGADRO, C_CM = 1.66053907e-27, 6.02214076e23, 2.99792458e10
R_KCAL, TEMPERATURE = 1.9872e-3, 298.15          # kcal mol-1 K-1, K
KCAL_PER_CM = 1.0 / 349.755                      # kcal/mol per cm-1
CUT_CM = 3.0 * R_KCAL * TEMPERATURE / KCAL_PER_CM  # 3 k_B T in cm-1

MASS = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
        "S": 32.06, "Se": 78.971, "Br": 79.904, "Ru": 101.07, "Ir": 192.217}
COVALENT = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05,
            "Se": 1.20, "Br": 1.20, "Ru": 1.46, "Ir": 1.41}             # A
METALS = {"Ru", "Ir"}
BOND_SCALE, METAL_BOND_SCALE = 1.25, 1.35        # x sum of covalent radii

COLUMNS = ["kind", "id", "surface", "displacement", "n_modes", "n_imaginary",
           "lambda / kcal mol-1", "classical_fraction", "quantum_frequency / cm-1"]


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    elements, xyz = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        elements.append(re.sub(r"\d", "", parts[0]).capitalize())
        xyz.append([float(v) for v in parts[1:4]])
    return elements, np.array(xyz)


def read_modes(path):
    """Frequencies in cm-1 and the normal-mode matrix of the last frequency block."""
    text = Path(path).read_text(errors="ignore")
    i = text.rfind("VIBRATIONAL FREQUENCIES")
    if i < 0:
        raise ValueError(f"no frequencies in {path}")
    j = text.find("NORMAL MODES", i)
    k = text.find("IR SPECTRUM", j)
    freqs = [float(m) for m in re.findall(r"^\s*\d+:\s+(-?[\d.]+) cm\*\*-1", text[i:j], re.MULTILINE)]
    n3 = len(freqs)
    modes = np.zeros((n3, n3))
    columns = None
    for line in text[j:k].splitlines():
        parts = line.split()
        if not parts:
            continue
        if all(re.fullmatch(r"\d+", t) for t in parts) and len(parts) <= 6:
            columns = [int(t) for t in parts]
            continue
        if columns is not None and re.fullmatch(r"\d+", parts[0]) and len(parts) == len(columns) + 1:
            modes[int(parts[0]), columns] = [float(v) for v in parts[1:]]
    return np.array(freqs), modes


def bond_cutoffs(elements):
    r = np.array([COVALENT[e] for e in elements])
    metal = np.array([e in METALS for e in elements])
    scale = np.where(metal[:, None] | metal[None], METAL_BOND_SCALE, BOND_SCALE)
    return scale * (r[:, None] + r[None])


def bonds_of(elements, X, Y):
    """Bonds present in either geometry, joined into one connected graph."""
    n = len(elements)
    cut = bond_cutoffs(elements)
    bonds = set()
    for G in (X, Y):
        d = np.linalg.norm(G[:, None] - G[None], axis=-1)
        i, j = np.where(np.triu(d < cut, 1))
        bonds |= set(zip(i.tolist(), j.tolist()))
    adjacent = {a: set() for a in range(n)}
    for a, b in bonds:
        adjacent[a].add(b)
        adjacent[b].add(a)
    while True:
        seen, stack = {0}, [0]
        while stack:
            a = stack.pop()
            for b in adjacent[a]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        if len(seen) == n:
            break
        rest = [a for a in range(n) if a not in seen]
        d = np.linalg.norm(X[list(seen)][:, None] - X[rest][None], axis=-1)
        a, b = np.unravel_index(d.argmin(), d.shape)
        a, b = list(seen)[a], rest[b]
        bonds.add((min(a, b), max(a, b)))
        adjacent[a].add(b)
        adjacent[b].add(a)
    return sorted(bonds), adjacent


def internals(adjacent, bonds, X):
    """Angles (near-linear ones left out) and dihedrals along every bond."""
    angles, dihedrals = [], []
    for j in adjacent:
        neighbours = sorted(adjacent[j])
        for a in range(len(neighbours)):
            for c in range(a + 1, len(neighbours)):
                i, k = neighbours[a], neighbours[c]
                v1, v2 = X[i] - X[j], X[k] - X[j]
                if abs(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)) < 0.995:
                    angles.append((i, j, k))
    for j, k in bonds:
        for i in adjacent[j]:
            if i == k:
                continue
            for m in adjacent[k]:
                if m in (j, i):
                    continue
                dihedrals.append((i, j, k, m))
    return angles, dihedrals


def q_values(G, bonds, angles, dihedrals):
    bi = np.asarray(bonds, int).reshape(-1, 2)
    b = np.linalg.norm(G[bi[:, 0]] - G[bi[:, 1]], axis=1)
    if len(angles):
        ai = np.asarray(angles, int)
        v1, v2 = G[ai[:, 0]] - G[ai[:, 1]], G[ai[:, 2]] - G[ai[:, 1]]
        c = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
        a = np.arccos(np.clip(c, -1, 1))
    else:
        a = np.zeros(0)
    if len(dihedrals):
        di = np.asarray(dihedrals, int)
        b0 = G[di[:, 0]] - G[di[:, 1]]
        b1 = G[di[:, 2]] - G[di[:, 1]]
        b2 = G[di[:, 3]] - G[di[:, 2]]
        b1n = b1 / np.linalg.norm(b1, axis=1)[:, None]
        v = b0 - (b0 * b1n).sum(1)[:, None] * b1n
        w = b2 - (b2 * b1n).sum(1)[:, None] * b1n
        d = np.arctan2((np.cross(b1n, v) * w).sum(1), (v * w).sum(1))
    else:
        d = np.zeros(0)
    return b, a, d


def wilson_b(G, bonds, angles, dihedrals, h=1e-5):
    """Numerical Wilson B matrix (internal coordinates x 3N) at G."""
    q0 = np.concatenate(q_values(G, bonds, angles, dihedrals))
    n3 = G.size
    B = np.zeros((q0.size, n3))
    flat = G.ravel()
    n_bonds, n_angles = len(bonds), len(angles)
    for c in range(n3):
        gp, gm = flat.copy(), flat.copy()
        gp[c] += h
        gm[c] -= h
        dq = (np.concatenate(q_values(gp.reshape(-1, 3), bonds, angles, dihedrals))
              - np.concatenate(q_values(gm.reshape(-1, 3), bonds, angles, dihedrals)))
        dq[n_bonds + n_angles:] = (dq[n_bonds + n_angles:] + np.pi) % (2 * np.pi) - np.pi
        B[:, c] = dq / (2 * h)
    return B


def kabsch(P, Q, w):
    """P superposed on Q with weights w."""
    w = w / w.sum()
    pc, qc = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    A, Bm = P - pc, Q - qc
    U, _, Vt = np.linalg.svd((w[:, None] * A).T @ Bm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ A.T).T + qc


def graph(elements, G):
    d = np.linalg.norm(G[:, None] - G[None], axis=-1)
    return np.triu(d < bond_cutoffs(elements), 1)


def candidates(elements, X, Y, m):
    """Y, its mirror image, and permutations of equal elements (assignment after Kabsch)
    of either that keep the bond graph of X."""
    out = [("identity", Y), ("mirror", Y * np.array([1.0, 1.0, -1.0]))]
    gX = graph(elements, X)
    el = np.array(elements)
    for label, Z0 in list(out):
        Z = kabsch(Z0, X, m)
        for _ in range(10):
            perm = np.arange(len(elements))
            for e in np.unique(el):
                idx = np.flatnonzero(el == e)
                cost = ((X[idx, None, :] - Z[None, idx, :]) ** 2).sum(-1)
                rows, cols = linear_sum_assignment(cost)
                perm[idx[rows]] = idx[cols]
            Z = kabsch(Z[perm], X, m)
            Z0 = Z0[perm]
        if (graph(elements, Z0) == gX).all() and not any(np.allclose(Z0, c) for _, c in out):
            out.append((label + "+perm", Z0))
    return out


def project_one(elements, X, Y, m, state_dir):
    """(frequencies, lambda_j internal, lambda_j cartesian, number of imaginary modes)."""
    freqs, L = read_modes(state_dir / "opt_output.out")
    if L.shape[0] != 3 * len(elements):
        raise ValueError(f"mode matrix of {state_dir} does not match the geometry")
    sm = np.repeat(np.sqrt(m), 3)
    Lmw = L * sm[:, None]
    norms = np.linalg.norm(Lmw, axis=0)
    Lmw[:, norms > 0] /= norms[norms > 0]
    vibrational = freqs > 1.0
    n_imaginary = int((freqs < -1.0).sum())
    omega = 2 * np.pi * C_CM * np.abs(freqs)
    to_kcal = (np.sqrt(AMU) * 1e-10) ** 2 * AVOGADRO / 4184.0

    bonds, adjacent = bonds_of(elements, X, Y)
    angles, dihedrals = internals(adjacent, bonds, X)
    qx, qy = q_values(X, bonds, angles, dihedrals), q_values(Y, bonds, angles, dihedrals)
    dq = np.concatenate([qy[0] - qx[0], qy[1] - qx[1],
                         (qy[2] - qx[2] + np.pi) % (2 * np.pi) - np.pi])
    Bmw = wilson_b(X, bonds, angles, dihedrals) / sm[None, :]
    Qi = Lmw.T @ (np.linalg.pinv(Bmw, rcond=1e-6) @ dq)
    lam_internal = 0.5 * omega ** 2 * Qi ** 2 * to_kcal

    Qc = Lmw.T @ (sm * (kabsch(Y, X, m) - X).ravel())
    lam_cartesian = 0.5 * omega ** 2 * Qc ** 2 * to_kcal
    return freqs[vibrational], lam_internal[vibrational], lam_cartesian[vibrational], n_imaginary


def project(state_dir, other_dir):
    elements, X = read_xyz(state_dir / "opt_geometry.xyz")
    elements_other, Y0 = read_xyz(other_dir / "opt_geometry.xyz")
    if elements != elements_other:
        raise ValueError(f"atom order of {state_dir} and {other_dir} differs")
    m = np.array([MASS[e] for e in elements])
    best = None
    for _, Y in candidates(elements, X, Y0, m):
        result = project_one(elements, X, Y, m, state_dir)
        if best is None or result[1].sum() < best[1].sum():
            best = result
    return best


def split(freqs, lam):
    """(sum, classical fraction, lambda-weighted quantum frequency) at CUT_CM."""
    quantum = freqs >= CUT_CM
    return (lam.sum(), lam[~quantum].sum() / lam.sum(),
            (lam[quantum] * freqs[quantum]).sum() / max(lam[quantum].sum(), 1e-12))


def molecules():
    """(kind, id, folder) of every photocatalyst and quencher with DFT jobs."""
    jobs = [("photocatalyst", int(d.name[2:]), d)
            for d in (RAW / "photocatalysts_new_ids").glob("PC*")
            if d.is_dir() and d.name[2:].isdigit()]
    jobs += [("quencher", int(d.name[1:]), d)
             for d in (RAW / "quencher_adiabatic_DFT").glob("Q*") if d.is_dir()]
    return sorted(jobs, key=lambda job: (job[0], job[1]))


def main():
    rows = []
    for kind, number, folder in molecules():
        for surface, other in (("S0", "T1"), ("T1", "S0")):
            freqs, lam_internal, lam_cartesian, n_imaginary = project(folder / surface, folder / other)
            for displacement, lam in (("internal", lam_internal), ("cartesian", lam_cartesian)):
                total, fraction, frequency = split(freqs, lam)
                rows.append({"kind": kind, "id": number, "surface": surface,
                             "displacement": displacement, "n_modes": len(freqs),
                             "n_imaginary": n_imaginary,
                             "lambda / kcal mol-1": f"{total:.2f}",
                             "classical_fraction": f"{fraction:.6f}",
                             "quantum_frequency / cm-1": f"{frequency:.2f}"})
            print(f"{kind:14s} {number:3d} {surface}  sum internal {lam_internal.sum():6.2f}  "
                  f"cartesian {lam_cartesian.sum():6.2f} kcal/mol", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {OUT.relative_to(DATA)}")


if __name__ == "__main__":
    main()
