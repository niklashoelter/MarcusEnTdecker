# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Organophotocatalyst triplet energies and reorganization energies.

Reads   ../computational_data_raw/organophotocatalysts/OPC<n>/{S0,T1}/*_output.out
Writes  ../computational_data_raw/organophotocatalysts/organophotocatalyst_dft.csv

    uv run data/computational_data_scripts/organophotocatalyst_dft.py

A separate library of organic photocatalysts. It has no measured pairs and is not read by
build_exp_dataset_ent.py or the benchmark. The level of theory is wB97X-D4 / def2-TZVP
(optimisation and frequencies) // def2-TZVPP, def2/J, CPCM(acetonitrile), without ZORA.

Each state folder holds the optimisation with frequencies (`opt_output.out`), a compound job
whose last single point is in def2-TZVPP, and the vertical single point of the other state at
that geometry (`sp_vert_<state>_output.out`). The definitions are those of the photocatalysts
(photocatalyst_dft.py):

    E_T          ((E_T1 + G_T1) - (E_S0 + G_S0)) * 627.509
    lambda_X_Y   (E_X at the Y geometry - E_X) * 627.509     electronic only

There are no oxidized or reduced states, so no excited-state potentials.

Values in kcal/mol are computed from the full-precision energies and rounded to 2 decimals
when written. Energies in Hartree are copied as ORCA prints them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "computational_data_raw" / "organophotocatalysts"
OUT = RAW / "organophotocatalyst_dft.csv"

HARTREE_KCAL = 627.509
DECIMALS = 2                    # for kcal/mol

NAMES = {
    1: "thioxanthone",
    2: "2-isopropylthioxanthone",
    3: "2-chlorothioxanthone",
    4: "2-methoxythioxanthone",
    5: "xanthone",
    6: "acetophenone",
    7: "benzophenone",
    8: "4CzIPN",
}

# state directory, file stem -> the key it is stored under
JOBS = {
    "E_S0": ("S0", "opt"),
    "E_T1": ("T1", "opt"),
    "E_T1_at_S0geom": ("S0", "sp_vert_T1"),
    "E_S0_at_T1geom": ("T1", "sp_vert_S0"),
}

OPTIMIZATIONS = ("E_S0", "E_T1")

COLUMNS = [
    "opc_id", "name", "n_atoms", "charge",
    "E_T / kcal mol-1", "lambda_S0_T1 / kcal mol-1", "lambda_T1_S0 / kcal mol-1",
    "E_S0 / Ha", "E_T1 / Ha", "E_T1_at_S0geom / Ha", "E_S0_at_T1geom / Ha",
    "G_corr_S0 / Ha", "G_corr_T1 / Ha",
    "n_imag_S0", "n_imag_T1", "minimum_confirmed", "n_jobs", "note",
]


def scan(path):
    """Electronic energy (string as printed), Gibbs correction, imaginary modes, charge.

    The optimizations are compound jobs that end with a single point in the larger TZVPP
    basis on the converged geometry, so the last FINAL SINGLE POINT ENERGY is used. The
    Gibbs correction comes from the frequency step in the smaller basis.
    """
    energy = gibbs = None
    n_imag = 0
    charge = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = line.split()[-1]
            elif line.startswith("G-E(el)"):
                gibbs = line.split()[2]
            elif "***imaginary mode***" in line:
                n_imag += 1
            elif charge is None and "Total Charge" in line:
                charge = int(line.split()[-1])
    return energy, gibbs, n_imag, charge


def count_atoms(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return int(fh.readline().strip())


def blank(x):
    return "" if x is None else x


def rounded(x):
    """A value in kcal/mol, to DECIMALS; blank if absent."""
    return "" if x is None else round(x, DECIMALS)


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input: {RAW}\nthe raw job files are not in git; fetch the deposit")

    opc_dirs = sorted((p for p in RAW.glob("OPC*") if p.is_dir()), key=lambda p: int(p.name[3:]))
    if not opc_dirs:
        sys.exit(f"no OPC<n> directories under {RAW}")
    unknown = [p.name for p in opc_dirs if int(p.name[3:]) not in NAMES]
    if unknown:
        sys.exit(f"no name for {', '.join(unknown)}")

    rows = []
    for opc_dir in opc_dirs:
        opc_id = int(opc_dir.name[3:])
        printed, gibbs, imag, charge = {}, {}, {}, None
        n_jobs = 0
        for key, (state, stem) in JOBS.items():
            path = opc_dir / state / f"{stem}_output.out"
            if not path.exists():
                continue
            e, g, n, c = scan(path)
            printed[key], gibbs[key], imag[key] = e, g, n
            if key == "E_S0":
                charge = c
            n_jobs += 1
        energy = {k: float(v) for k, v in printed.items() if v is not None}
        free = {k: energy[k] + float(gibbs[k]) for k in OPTIMIZATIONS
                if k in energy and gibbs.get(k) is not None}

        def lam(at_other_geom, own):
            """Relaxation on one surface, in kcal/mol, or None if a job is absent."""
            if at_other_geom not in energy or own not in energy:
                return None
            return (energy[at_other_geom] - energy[own]) * HARTREE_KCAL

        e_t = ((free["E_T1"] - free["E_S0"]) * HARTREE_KCAL
               if "E_T1" in free and "E_S0" in free else None)

        missing = [k for k in JOBS if k not in energy]
        n_imag_total = sum(imag.get(k, 0) for k in OPTIMIZATIONS)
        notes = []
        if missing:
            notes.append("no " + ", ".join(missing))
        if n_imag_total:
            where = [k[2:] for k in OPTIMIZATIONS if imag.get(k)]
            notes.append(f"imaginary frequencies in {', '.join(where)}")

        rows.append({
            "opc_id": opc_id,
            "name": NAMES[opc_id],
            "n_atoms": count_atoms(opc_dir / "S0" / "opt_geometry.xyz") or "",
            "charge": blank(charge),
            "E_T / kcal mol-1": rounded(e_t),
            "lambda_S0_T1 / kcal mol-1": rounded(lam("E_S0_at_T1geom", "E_S0")),
            "lambda_T1_S0 / kcal mol-1": rounded(lam("E_T1_at_S0geom", "E_T1")),
            **{f"{k} / Ha": blank(printed.get(k)) for k in JOBS},
            **{f"G_corr_{k[2:]} / Ha": blank(gibbs.get(k)) for k in OPTIMIZATIONS},
            **{f"n_imag_{k[2:]}": imag.get(k, "") for k in OPTIMIZATIONS},
            "minimum_confirmed": 0 if n_imag_total else 1,
            "n_jobs": n_jobs,
            "note": "; ".join(notes),
        })

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    e_t = [r["E_T / kcal mol-1"] for r in rows if r["E_T / kcal mol-1"] != ""]
    print(f"{len(rows)} organophotocatalysts -> {OUT.relative_to(ROOT.parent)}")
    print(f"  E_T on {len(e_t)}: {min(e_t):.1f} to {max(e_t):.1f} kcal/mol")
    for r in rows:
        if r["n_jobs"] < len(JOBS) or not r["minimum_confirmed"]:
            print(f"  OPC{r['opc_id']}: {r['n_jobs']}/{len(JOBS)} jobs -- {r['note']}")


if __name__ == "__main__":
    main()
