# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Photocatalyst triplet energies, reorganization energies and redox potentials from DFT.

Reads   ../computational_data_raw/photocatalysts/PC_<n>/{S0,T1,ox,red}/*_output.out
Writes  ../computational_data_processed/photocatalyst_dft.csv

    uv run data/computational_data_scripts/photocatalyst_dft.py

Nothing is read from outside `data/`, and no spreadsheet is consulted. Every number in
the output is parsed out of an ORCA output file that is in the deposit beside it.

THE FOUR-POINT SCHEME
    Each property is a difference between two energies, and which geometry each is
    evaluated at is the whole content of the calculation. Four optimizations -- S0, T1,
    the oxidized and the reduced species -- and six single points that cross them over.

        E_T           adiabatic triplet energy, Gibbs-corrected:
                      (E_T1 + G_T1) - (E_S0 + G_S0)
        lambda_X_to_Y the energy released relaxing from the X geometry to the Y
                      geometry on the Y surface -- electronic only, no Gibbs term,
                      because a reorganization energy is a difference on one potential
                      energy surface and the vibrational corrections do not belong in it

    The naming is deliberately geometry-explicit. `lambda_S0_to_T1` is the relaxation on
    the *triplet* surface starting from the singlet geometry; `lambda_T1_to_S0` is the
    relaxation on the *singlet* surface starting from the triplet geometry. They are
    different numbers and they enter Marcus theory in different directions, so a name
    that does not say which surface it lives on is a name that will be misread.

POTENTIALS ARE NAMED BY THEIR COUPLE
    `E_PCplus_PC_V` is the PC+/PC couple, the ground-state oxidation. `E_PCstar_PCminus_V`
    is the PC*/PC- couple -- how strong an oxidant the triplet is. Naming a potential
    "oxidation potential" invites the reader to attach it to the wrong species, because
    the downstream driving-force expressions are quencher-centric and invert: the
    potential at which PC* *oxidizes a substrate* is the reduction potential of the
    PC*/PC- couple. The source spreadsheet has these two labels crossed for exactly this
    reason, which is why this table does not use them.

THE ABSOLUTE REFERENCE
    A computed free energy of ionization is on an absolute scale and has to be shifted
    onto the SCE scale the experiments are on. That shift is a chosen constant, not a
    computed one, and it moves every potential here rigidly -- so it is named once,
    below, rather than buried in an expression.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "computational_data_raw" / "photocatalysts"
OUT = ROOT / "computational_data_processed" / "photocatalyst_dft.csv"

HARTREE_KCAL = 627.509474
HARTREE_EV = 27.211386

# Absolute potential of the saturated calomel electrode, in volts. Recovered from the
# source analysis and adopted so that this table stays on the same scale as the
# potentials already published from it; it reproduces them to 2e-5 V across all 31
# photocatalysts. It is a convention, and a different choice shifts every potential in
# this file by the same amount without changing any difference between them.
SCE_ABSOLUTE_V = 4.422

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
    "E_PCplus_PC / V", "E_PC_PCminus / V",
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
    """Electronic energy, Gibbs correction and imaginary-mode count from one output.

    The last FINAL SINGLE POINT ENERGY is taken, not the first: the optimizations are
    compound jobs that re-run a single point in the larger TZVPP basis on the converged
    geometry, and that larger-basis energy is the one every difference below is built
    from. The Gibbs correction comes from the freq step in the smaller basis, which is
    where the frequencies were computed -- mixing the two is intentional and standard,
    and it is what the published numbers did.
    """
    energy = gibbs = None
    n_imag = 0
    charge = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[-1])
            elif line.startswith("G-E(el)"):
                gibbs = float(line.split()[2])
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


def sig(x, n=8):
    if x is None:
        return ""
    return 0.0 if float(x) == 0 else float(f"{float(x):.{n}g}")


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input: {RAW}\n"
                 f"the raw job files are not in git -- see {RAW.parent}/README.md")

    pc_dirs = sorted(RAW.glob("PC_*"), key=lambda p: int(p.name.split("_")[1]))
    if not pc_dirs:
        sys.exit(f"no PC_<n> directories under {RAW}")

    rows = []
    for pc_dir in pc_dirs:
        pc_id = int(pc_dir.name.split("_")[1])
        energy, gibbs, imag, charge = {}, {}, {}, None
        n_jobs = 0
        for key, (state, stem) in JOBS.items():
            path = pc_dir / state / f"{stem}_output.out"
            if not path.exists():
                continue
            e, g, n, c = scan(path)
            energy[key], gibbs[key], imag[key] = e, g, n
            if key == "E_S0":
                charge = c
            n_jobs += 1

        def diff_kcal(a, b):
            """a - b in kcal/mol, or None if either job is absent."""
            if energy.get(a) is None or energy.get(b) is None:
                return None
            return (energy[a] - energy[b]) * HARTREE_KCAL

        def delta_G_eV(state):
            """Free energy of the ionization S0 -> state, in eV."""
            if energy.get(state) is None or energy.get("E_S0") is None:
                return None
            if gibbs.get(state) is None or gibbs.get("E_S0") is None:
                return None
            return ((energy[state] + gibbs[state])
                    - (energy["E_S0"] + gibbs["E_S0"])) * HARTREE_EV

        e_t_kcal = None
        if all(energy.get(k) is not None and gibbs.get(k) is not None
               for k in ("E_T1", "E_S0")):
            e_t_kcal = ((energy["E_T1"] + gibbs["E_T1"])
                        - (energy["E_S0"] + gibbs["E_S0"])) * HARTREE_KCAL

        # Oxidation costs free energy, so its potential is +dG; reduction releases it,
        # so its potential is -dG. Both are then shifted onto the SCE scale.
        dG_ox, dG_red = delta_G_eV("E_ox"), delta_G_eV("E_red")
        e_ox_v = dG_ox - SCE_ABSOLUTE_V if dG_ox is not None else None
        e_red_v = -dG_red - SCE_ABSOLUTE_V if dG_red is not None else None

        # The excited state is a stronger oxidant and a stronger reductant than the
        # ground state, by exactly the energy stored in it.
        e_t_volt = e_t_kcal / 23.060548 if e_t_kcal is not None else None
        e_star_red_v = (e_red_v + e_t_volt) if None not in (e_red_v, e_t_volt) else None
        e_star_ox_v = (e_ox_v - e_t_volt) if None not in (e_ox_v, e_t_volt) else None

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
            "charge": "" if charge is None else charge,
            "E_T / kcal mol-1": sig(e_t_kcal),
            "lambda_S0_to_T1 / kcal mol-1": sig(diff_kcal("E_T1_at_S0geom", "E_T1")),
            "lambda_T1_to_S0 / kcal mol-1": sig(diff_kcal("E_S0_at_T1geom", "E_S0")),
            "lambda_S0_to_ox / kcal mol-1": sig(diff_kcal("E_ox_at_S0geom", "E_ox")),
            "lambda_ox_to_S0 / kcal mol-1": sig(diff_kcal("E_S0_at_oxgeom", "E_S0")),
            "lambda_S0_to_red / kcal mol-1": sig(diff_kcal("E_red_at_S0geom", "E_red")),
            "lambda_red_to_S0 / kcal mol-1": sig(diff_kcal("E_S0_at_redgeom", "E_S0")),
            "E_PCplus_PC / V": sig(e_ox_v, 6),
            "E_PC_PCminus / V": sig(e_red_v, 6),
            "E_PCstar_PCminus / V": sig(e_star_red_v, 6),
            "E_PCplus_PCstar / V": sig(e_star_ox_v, 6),
            "E_S0 / Ha": sig(energy.get("E_S0"), 14),
            "E_T1 / Ha": sig(energy.get("E_T1"), 14),
            "E_ox / Ha": sig(energy.get("E_ox"), 14),
            "E_red / Ha": sig(energy.get("E_red"), 14),
            "E_T1_at_S0geom / Ha": sig(energy.get("E_T1_at_S0geom"), 14),
            "E_S0_at_T1geom / Ha": sig(energy.get("E_S0_at_T1geom"), 14),
            "E_ox_at_S0geom / Ha": sig(energy.get("E_ox_at_S0geom"), 14),
            "E_S0_at_oxgeom / Ha": sig(energy.get("E_S0_at_oxgeom"), 14),
            "E_red_at_S0geom / Ha": sig(energy.get("E_red_at_S0geom"), 14),
            "E_S0_at_redgeom / Ha": sig(energy.get("E_S0_at_redgeom"), 14),
            "G_corr_S0 / Ha": sig(gibbs.get("E_S0"), 10),
            "G_corr_T1 / Ha": sig(gibbs.get("E_T1"), 10),
            "G_corr_ox / Ha": sig(gibbs.get("E_ox"), 10),
            "G_corr_red / Ha": sig(gibbs.get("E_red"), 10),
            "n_imag_S0": imag.get("E_S0", ""),
            "n_imag_T1": imag.get("E_T1", ""),
            "n_imag_ox": imag.get("E_ox", ""),
            "n_imag_red": imag.get("E_red", ""),
            "minimum_confirmed": 0 if n_imag_total else 1,
            "n_jobs": n_jobs,
            "note": "; ".join(notes),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    complete = [r for r in rows if r["E_T / kcal mol-1"] != ""]
    e_t = [r["E_T / kcal mol-1"] for r in complete]
    print(f"{len(rows)} photocatalysts -> {OUT.relative_to(ROOT.parent)}")
    print(f"  E_T on {len(e_t)}: {min(e_t):.1f} to {max(e_t):.1f} kcal/mol")

    incomplete = [r for r in rows if r["n_jobs"] < len(JOBS)]
    for r in incomplete:
        print(f"  PC_{r['pc_id']}: {r['n_jobs']}/{len(JOBS)} jobs -- {r['note']}")

    unconfirmed = [r for r in rows if not r["minimum_confirmed"]]
    for r in unconfirmed:
        print(f"  PC_{r['pc_id']}: not a minimum -- {r['note']}")


if __name__ == "__main__":
    main()
