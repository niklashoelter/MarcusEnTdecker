# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""
Photocatalyst triplet energies, reorganization energies, excited-state potentials and
molecular radii.

Reads   ../computational_data_raw/photocatalysts/PC_<n>/{S0,T1,ox,red}/*_output.out
        ../computational_data_raw/photocatalysts/PC_<n>/S0/opt_geometry.xyz
        ../compound_identification.xlsx   (photocatalyst ids)
Writes  ../computational_data_raw/photocatalysts/photocatalyst_dft.csv

    uv run data/computational_data_scripts/photocatalyst_dft.py

Each state folder holds the optimisation with frequencies (`opt_output.out`) and the
vertical single points of the other states at that geometry (`sp_vert_<state>_output.out`).
E is the electronic energy, G the Gibbs correction G-E(el):

    E_T                ((E_T1 + G_T1) - (E_S0 + G_S0)) * 627.509
    lambda_X_Y         (E_X at the Y geometry - E_X) * 627.509     electronic only
    dG_ox_ES           ((E_ox + G_ox) - (E_T1 + G_T1)) * 627.509   PC* -> PC+
    dG_red_ES          ((E_red + G_red) - (E_T1 + G_T1)) * 627.509  PC* -> PC-
    E_PCstar_PCminus   -dG_red_ES / 23.061 - 4.42
    E_PCplus_PCstar     dG_ox_ES / 23.061 - 4.42

Reorganization energies are named lambda_<surface>_<geometry>: `lambda_S0_T1` is the
singlet surface at the T1 geometry, `lambda_T1_S0` the triplet surface at the S0 geometry.
Dplus and Dminus are the one-electron oxidized and reduced states, the `ox` and `red`
folders. Potentials are named by their couple: E(PC*/PC-) measures how strong an oxidant
the triplet is.

radius_S0 is the radius in A of the sphere with the van der Waals volume of the optimised
S0 geometry (../molecular_size.py). The benchmark takes r_PC + r_Q as the contact distance
of a pair.

Photocatalysts are numbered by old id. Those excluded from the data are skipped
(../compound_identification.py).

Values in kcal/mol and V are computed from the full-precision energies and rounded to 2
decimals when written, the radius to 2 decimals in A. Energies in Hartree are copied as
ORCA prints them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "computational_data_raw" / "photocatalysts"
OUT = RAW / "photocatalyst_dft.csv"

HARTREE_KCAL = 627.509
FARADAY_KCAL_V = 23.061
SCE_ABSOLUTE_V = 4.42
DECIMALS = 2                    # for kcal/mol, V and A

# Old ids of the photocatalysts in the data.
sys.path.insert(0, str(ROOT))
from compound_identification import IN_DATA  # noqa: E402
from molecular_size import radius  # noqa: E402

# state directory, file stem -> the key it is stored under
JOBS = {
    "E_S0": ("S0", "opt"),
    "E_T1": ("T1", "opt"),
    "E_ox": ("ox", "opt"),
    "E_red": ("red", "opt"),
    "E_T1_at_S0geom": ("S0", "sp_vert_T1"),
    "E_ox_at_S0geom": ("S0", "sp_vert_ox"),
    "E_red_at_S0geom": ("S0", "sp_vert_red"),
    "E_S0_at_T1geom": ("T1", "sp_vert_S0"),
    "E_S0_at_oxgeom": ("ox", "sp_vert_S0"),
    "E_S0_at_redgeom": ("red", "sp_vert_S0"),
}

OPTIMIZATIONS = ("E_S0", "E_T1", "E_ox", "E_red")

COLUMNS = [
    "pc_id", "n_atoms", "charge",
    "E_T / kcal mol-1",
    "lambda_S0_T1 / kcal mol-1", "lambda_T1_S0 / kcal mol-1",
    "lambda_S0_Dplus / kcal mol-1", "lambda_Dplus_S0 / kcal mol-1",
    "lambda_S0_Dminus / kcal mol-1", "lambda_Dminus_S0 / kcal mol-1",
    "dG_ox_ES / kcal mol-1", "dG_red_ES / kcal mol-1",
    "E_PCstar_PCminus / V", "E_PCplus_PCstar / V",
    "radius_S0 / A",
    "E_S0 / Ha", "E_T1 / Ha", "E_ox / Ha", "E_red / Ha",
    "E_T1_at_S0geom / Ha", "E_S0_at_T1geom / Ha",
    "E_ox_at_S0geom / Ha", "E_S0_at_oxgeom / Ha",
    "E_red_at_S0geom / Ha", "E_S0_at_redgeom / Ha",
    "G_corr_S0 / Ha", "G_corr_T1 / Ha", "G_corr_ox / Ha", "G_corr_red / Ha",
    "n_imag_S0", "n_imag_T1", "n_imag_ox", "n_imag_red",
    "minimum_confirmed", "n_jobs", "note",
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
    """A value in kcal/mol, V or A, to DECIMALS; blank if absent."""
    return "" if x is None else round(x, DECIMALS)


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input: {RAW}\nthe raw job files are not in git; fetch the deposit")

    pc_dirs = sorted((p for p in RAW.glob("PC_*")
                      if int(p.name.split("_")[1]) in IN_DATA),
                     key=lambda p: int(p.name.split("_")[1]))
    if not pc_dirs:
        sys.exit(f"no PC_<n> directories under {RAW}")

    rows = []
    for pc_dir in pc_dirs:
        pc_id = int(pc_dir.name.split("_")[1])
        printed, gibbs, imag, charge = {}, {}, {}, None
        n_jobs = 0
        for key, (state, stem) in JOBS.items():
            path = pc_dir / state / f"{stem}_output.out"
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

        def dG(final, initial, factor=HARTREE_KCAL):
            if final not in free or initial not in free:
                return None
            return (free[final] - free[initial]) * factor

        e_t = dG("E_T1", "E_S0")
        dg_ox, dg_red = dG("E_ox", "E_T1"), dG("E_red", "E_T1")
        e_star_minus = -dg_red / FARADAY_KCAL_V - SCE_ABSOLUTE_V if dg_red is not None else None
        e_plus_star = dg_ox / FARADAY_KCAL_V - SCE_ABSOLUTE_V if dg_ox is not None else None

        missing = [k for k in JOBS if k not in energy]
        n_imag_total = sum(imag.get(k, 0) for k in OPTIMIZATIONS)
        notes = []
        if missing:
            notes.append("no " + ", ".join(missing))
        if n_imag_total:
            where = [k[2:] for k in OPTIMIZATIONS if imag.get(k)]
            notes.append(f"imaginary frequencies in {', '.join(where)}")

        rows.append({
            "pc_id": pc_id,
            "n_atoms": count_atoms(pc_dir / "S0" / "opt_geometry.xyz") or "",
            "charge": blank(charge),
            "E_T / kcal mol-1": rounded(e_t),
            "lambda_S0_T1 / kcal mol-1": rounded(lam("E_S0_at_T1geom", "E_S0")),
            "lambda_T1_S0 / kcal mol-1": rounded(lam("E_T1_at_S0geom", "E_T1")),
            "lambda_S0_Dplus / kcal mol-1": rounded(lam("E_S0_at_oxgeom", "E_S0")),
            "lambda_Dplus_S0 / kcal mol-1": rounded(lam("E_ox_at_S0geom", "E_ox")),
            "lambda_S0_Dminus / kcal mol-1": rounded(lam("E_S0_at_redgeom", "E_S0")),
            "lambda_Dminus_S0 / kcal mol-1": rounded(lam("E_red_at_S0geom", "E_red")),
            "dG_ox_ES / kcal mol-1": rounded(dg_ox),
            "dG_red_ES / kcal mol-1": rounded(dg_red),
            "E_PCstar_PCminus / V": rounded(e_star_minus),
            "E_PCplus_PCstar / V": rounded(e_plus_star),
            "radius_S0 / A": rounded(radius(pc_dir / "S0" / "opt_geometry.xyz")),
            **{f"{k} / Ha": blank(printed.get(k)) for k in JOBS},
            **{f"G_corr_{k[2:]} / Ha": blank(gibbs.get(k)) for k in OPTIMIZATIONS},
            **{f"n_imag_{k[2:]}": imag.get(k, "") for k in OPTIMIZATIONS},
            "minimum_confirmed": 0 if n_imag_total else 1,
            "n_jobs": n_jobs,
            "note": "; ".join(notes),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    e_t = [r["E_T / kcal mol-1"] for r in rows if r["E_T / kcal mol-1"] != ""]
    print(f"{len(rows)} photocatalysts -> {OUT.relative_to(ROOT.parent)}")
    print(f"  E_T on {len(e_t)}: {min(e_t):.1f} to {max(e_t):.1f} kcal/mol")
    for r in rows:
        if r["n_jobs"] < len(JOBS) or not r["minimum_confirmed"]:
            print(f"  PC_{r['pc_id']}: {r['n_jobs']}/{len(JOBS)} jobs -- {r['note']}")


if __name__ == "__main__":
    main()
