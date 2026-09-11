# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Photocatalyst triplet energies, reorganization energies and excited-state potentials.

Reads   ../computational_data_raw/photocatalysts/PC_<n>/{S0,T1,ox,red}/*_output.out
Writes  ../computational_data_raw/photocatalysts/photocatalyst_dft.csv

    uv run data/computational_data_scripts/photocatalyst_dft.py

Every number is parsed out of an ORCA output in the deposit; no spreadsheet is read. The
arithmetic is Leon Schlosser's, as it stands in the photocatalyst property file he
delivered (spectroscopy_data_processed/dataset/properties_photocatalysts_calc.xlsx), so
that this table reproduces it:

    E_T                ((E_T1 + G_T1) - (E_S0 + G_S0)) * 627.503
    lambda_X_to_Y      (E_Y at the X geometry - E_Y) * 627.509     electronic only
    dG_ox_ES           ((E_ox + G_ox) - (E_T1 + G_T1)) * 627.509   PC* -> PC+
    dG_red_ES          ((E_red + G_red) - (E_T1 + G_T1)) * 627.509  PC* -> PC-
    E_PCstar_PCminus   -dG_red_ES / 23.061 - 4.42
    E_PCplus_PCstar     dG_ox_ES / 23.061 - 4.42

The two Hartree conversions differ (627.503 for E_T, 627.509 for everything else) because
they do in the source. They are kept rather than unified: the point of this table is to
reproduce the published values, and unifying them moves E_T by 6e-4 kcal/mol.

AGREEMENT WITH THE DELIVERED FILE
    The six lambdas agree to 1e-13 kcal/mol, the two dG to 6e-8 (the source computed them
    in Excel from energies pasted to 15 significant digits), E_T to 5e-8 -- for every
    photocatalyst the file has, except four E_T values. PC10, PC13, PC25 and PC28 differ
    by +2.23, -0.70, +0.61 and -0.55 kcal/mol: the delivered file took E_T from a
    spreadsheet column that was not updated for those four, and no job in the delivery
    reproduces its numbers. The values here come from the jobs. PC19 is blank in the
    delivered file; here it has what its six jobs allow (four single points were never
    run).

NAMING
    `lambda_S0_to_T1` is the relaxation on the *triplet* surface starting from the singlet
    geometry (the delivered file's `lamba_T-S`); `lambda_T1_to_S0` is the relaxation on
    the *singlet* surface starting from the triplet geometry (its `lambda_S-T`). The
    ox/red pairs follow the same pattern. Potentials are named by their couple:
    E(PC*/PC-) is how strong an oxidant the triplet is. The delivered file labels it
    `E_ox_ES`, and E(PC+/PC*) `E_red_ES`.

Photocatalysts are numbered by old id here. Old PC7, PC26, PC30 and PC31 were computed but
are crossed out in the photocatalyst sheet and are not part of the data
(../photocatalyst_ids.py).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "computational_data_raw" / "photocatalysts"
OUT = RAW / "photocatalyst_dft.csv"

HARTREE_KCAL_E_T = 627.503      # as used for E_T in the source
HARTREE_KCAL = 627.509          # as used for every other energy difference
FARADAY_KCAL_V = 23.061
SCE_ABSOLUTE_V = 4.42

# Only the photocatalysts in the data; the crossed-out ones are skipped. Old ids.
sys.path.insert(0, str(ROOT))
from photocatalyst_ids import IN_DATA  # noqa: E402

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
    "lambda_S0_to_T1 / kcal mol-1", "lambda_T1_to_S0 / kcal mol-1",
    "lambda_S0_to_ox / kcal mol-1", "lambda_ox_to_S0 / kcal mol-1",
    "lambda_S0_to_red / kcal mol-1", "lambda_red_to_S0 / kcal mol-1",
    "dG_ox_ES / kcal mol-1", "dG_red_ES / kcal mol-1",
    "E_PCstar_PCminus / V", "E_PCplus_PCstar / V",
    "E_S0 / Ha", "E_T1 / Ha", "E_ox / Ha", "E_red / Ha",
    "E_T1_at_S0geom / Ha", "E_S0_at_T1geom / Ha",
    "E_ox_at_S0geom / Ha", "E_S0_at_oxgeom / Ha",
    "E_red_at_S0geom / Ha", "E_S0_at_redgeom / Ha",
    "G_corr_S0 / Ha", "G_corr_T1 / Ha", "G_corr_ox / Ha", "G_corr_red / Ha",
    "n_imag_S0", "n_imag_T1", "n_imag_ox", "n_imag_red",
    "minimum_confirmed", "n_jobs", "note",
]


def scan(path):
    """Electronic energy (as printed), Gibbs correction, imaginary modes, charge.

    The last FINAL SINGLE POINT ENERGY is taken, not the first: the optimizations are
    compound jobs that re-run a single point in the larger TZVPP basis on the converged
    geometry, and that larger-basis energy is the one every difference below is built
    from. The Gibbs correction comes from the freq step in the smaller basis, which is
    where the frequencies were computed -- mixing the two is intentional, and it is what
    the published numbers did. The energy is kept as ORCA printed it, so the table
    carries it at full precision.
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

        e_t = dG("E_T1", "E_S0", HARTREE_KCAL_E_T)
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
            "E_T / kcal mol-1": blank(e_t),
            "lambda_S0_to_T1 / kcal mol-1": blank(lam("E_T1_at_S0geom", "E_T1")),
            "lambda_T1_to_S0 / kcal mol-1": blank(lam("E_S0_at_T1geom", "E_S0")),
            "lambda_S0_to_ox / kcal mol-1": blank(lam("E_ox_at_S0geom", "E_ox")),
            "lambda_ox_to_S0 / kcal mol-1": blank(lam("E_S0_at_oxgeom", "E_S0")),
            "lambda_S0_to_red / kcal mol-1": blank(lam("E_red_at_S0geom", "E_red")),
            "lambda_red_to_S0 / kcal mol-1": blank(lam("E_S0_at_redgeom", "E_S0")),
            "dG_ox_ES / kcal mol-1": blank(dg_ox),
            "dG_red_ES / kcal mol-1": blank(dg_red),
            "E_PCstar_PCminus / V": blank(e_star_minus),
            "E_PCplus_PCstar / V": blank(e_plus_star),
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
