# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Processed tables -> one row per photocatalyst-quencher pair.

Reads   ../data/experimental_data_processed/stern_volmer_fits.csv
        ../data/experimental_data_processed/lifetimes_tau0.csv
        ../data/experimental_data_processed/compound_records_quenchers.csv
        ../data/experimental_data_processed/compound_records_photocatalysts.csv
        ../data/experimental_data_processed/absorption_extinction.csv
        ../data/computational_data_processed/quencher_dft.csv
        ../data/computational_data_processed/quencher_dvte.csv
Writes  pairs.csv

Nothing is read from outside ../data/. No spreadsheet is consulted at run time and no
file is copied in as an import side effect. Every column below can be followed back to
the instrument file or job output it came from by walking the chain one step at a time.

THE CENTRAL QUANTITY

    k_q = K_SV / tau_0

K_SV per pair from the Stern-Volmer fits, tau_0 per photocatalyst from the TCSPC
traces. Both measured here.

WHY k_q AND NOT K_SV
    tau_0 spans 86x across the screen, from 51 ns to 6439 ns. K_SV = k_q * tau_0, so a
    fixed cutoff on K_SV demands an 86x larger rate constant from the short-lived
    photocatalysts than from the long-lived ones -- it would label the photocatalyst,
    not the pair. Marcus and Rehm-Weller predict k_q, so k_q is what the ground truth
    is defined on.

CENSORING IS CARRIED, NOT DROPPED
    Roughly half the screen falls below the detection floor. Those pairs are one-sided
    observations, not missing data, and every one of them is in this table with
    `censored = 1` and its fitted K_SV intact, negative slopes included. `k_q_floor`
    gives the smallest rate constant that pair could have shown, `25 / tau_0`, which is
    the number a survival analysis needs and which varies by photocatalyst precisely
    because tau_0 does.

    Dropping them would bias every fit in the paper towards the pairs that happened to
    quench.

BOUNDS ARE CARRIED TOO
    Nineteen quenchers have a redox potential recorded as a limit rather than a value.
    `E_Ox_bound` / `E_Red_bound` say so. Anything that rules out the single-electron
    transfer channel by comparing potentials has to look at those columns, because
    treating `> 2.0` as `2.0` makes oxidation look reachable when the measurement only
    established that it is not reachable below 2.0 V.

WHAT IS NOT HERE YET
    Every photocatalyst-side computed quantity: E_T(PC), its reorganization energies,
    and its excited-state redox potentials. The DFT job files for the photocatalysts
    are not in the repository, and this package will not inherit those numbers from a
    spreadsheet it cannot regenerate.

    That means the pair-level driving forces -- dG_EnT, dG_SET_ox, dG_SET_red -- cannot
    be computed yet, and no column for them is written. They are not missing values to
    be filled in later by a merge; they are quantities this table is not yet entitled
    to state. Everything on the quencher side is present, so they become a one-line
    addition once the photocatalyst job files arrive.
"""

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTAL = ROOT / "data" / "experimental_data_processed"
COMPUTATIONAL = ROOT / "data" / "computational_data_processed"
OUT = Path(__file__).resolve().parent / "pairs.csv"

# A pair counts as quenching above this rate constant. 1e7 M-1 s-1 is the historical
# K_SV >= 10 threshold at a representative tau_0 of 1 microsecond.
K_Q_THRESHOLD = 1.0e7

# The Stern-Volmer detection floor, in K_SV. Fixed in K_SV, photocatalyst-specific once
# divided by tau_0 -- which is the whole reason the floor has to be carried per row.
K_SV_FLOOR = 25.0

SIGFIGS = 6

# Wavelengths at which an extinction coefficient was measured, for matching a quencher's
# absorbance to the wavelength its photocatalyst was excited at.
EPSILON_NM = [380, 390, 400, 405, 415, 425, 450, 475, 490, 495, 500,
              510, 520, 580, 590, 600, 625, 645]

COLUMNS = [
    # identity
    "pc_id", "pc_name", "pc_compound", "quencher_id", "quencher_name", "quencher_compound",
    "quencher_smiles",
    # the measurement
    "K_SV / M-1", "r2", "n_points_fitted", "n_points_measured", "fit_method", "fit_note",
    "tau0 / ns", "tau0_rsd", "tau0_consistent",
    # the rate
    "k_q / M-1 s-1", "log10_k_q", "quenches", "censored", "k_q_floor / M-1 s-1",
    # quencher, experimental
    "q_E_Ox_vs_SCE_exp / V", "q_E_Ox_bound", "q_E_Red_vs_SCE_exp / V", "q_E_Red_bound",
    "q_epsilon_at_pc_excitation / mM-1 cm-1", "pc_excitation / nm",
    # quencher, computed
    "q_E_T_dft / kcal mol-1", "q_lambda_S_T_dft / kcal mol-1", "q_lambda_T_S_dft / kcal mol-1",
    "q_E_Ox_vs_SCE_calc / V", "q_E_Red_vs_SCE_calc / V",
    "q_dvte_mu_md / kcal mol-1", "q_dvte_std_md / kcal mol-1",
]


def sig(x, n=SIGFIGS):
    if x in (None, ""):
        return ""
    return 0.0 if float(x) == 0 else float(f"{float(x):.{n}g}")


def load(path, key):
    """Read a processed CSV into {key: row}. Missing file is fatal, not skipped."""
    if not path.exists():
        sys.exit(f"missing input: {path}\nrun the data stage first: uv run reproduce.py data")
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(row[key]): row for row in csv.DictReader(fh)}


def number(row, field):
    """A numeric field, or None when absent or blank."""
    if row is None:
        return None
    value = row.get(field, "")
    return float(value) if value not in ("", None) else None


def nearest_epsilon(absorption_row, excitation_nm):
    """Extinction coefficient at the wavelength closest to the excitation wavelength.

    Excitation was at 380, 405, 425 or 450 nm and every one of those was measured
    directly, so `nearest` is an exact match in practice. It is written as a search
    rather than a lookup so that a photocatalyst excited somewhere else still resolves,
    and `epsilon_offset_nm` would be the thing to add if that ever happens.
    """
    if absorption_row is None or excitation_nm is None:
        return None
    nm = min(EPSILON_NM, key=lambda w: abs(w - excitation_nm))
    return number(absorption_row, f"epsilon_{nm}_mM-1cm-1")


def main():
    fits_path = EXPERIMENTAL / "stern_volmer_fits.csv"
    if not fits_path.exists():
        sys.exit(f"missing input: {fits_path}\nrun the data stage first")

    tau0 = load(EXPERIMENTAL / "lifetimes_tau0.csv", "pc_id")
    quenchers = load(EXPERIMENTAL / "compound_records_quenchers.csv", "quencher_id")
    photocatalysts = load(EXPERIMENTAL / "compound_records_photocatalysts.csv", "pc_id")
    absorption = load(EXPERIMENTAL / "absorption_extinction.csv", "quencher_id")
    dft = load(COMPUTATIONAL / "quencher_dft.csv", "quencher_id")
    dvte = load(COMPUTATIONAL / "quencher_dvte.csv", "quencher_id")

    rows, no_tau0, quench_count, censored_count = [], set(), 0, 0

    with fits_path.open(encoding="utf-8", newline="") as fh:
        for fit in csv.DictReader(fh):
            pc_id, q_id = int(fit["pc_id"]), int(fit["quencher_id"])

            lifetime = number(tau0.get(pc_id), "tau0_ns")
            if lifetime is None:
                no_tau0.add(pc_id)

            k_sv = number(fit, "K_SV")
            # tau_0 is in ns; k_q comes out in M-1 s-1.
            k_q = (k_sv / (lifetime * 1e-9)) if (k_sv is not None and lifetime) else None
            floor = (K_SV_FLOOR / (lifetime * 1e-9)) if lifetime else None

            quenches = (k_q is not None and k_q > K_Q_THRESHOLD)
            # `quenching_observed` is the Stern-Volmer detection criterion, K_SV >= 25
            # and r2 >= 0.90. A pair that fails it is left-censored: the true rate is
            # somewhere below the floor, not zero and not missing.
            censored = fit["quenching_observed"] != "True"
            quench_count += quenches
            censored_count += censored

            pc = photocatalysts.get(pc_id)
            q = quenchers.get(q_id)
            excitation = number(pc, "excitation_nm")

            rows.append({
                "pc_id": pc_id,
                "pc_name": (pc or {}).get("pc_name", ""),
                "pc_compound": (pc or {}).get("name", ""),
                "quencher_id": q_id,
                "quencher_name": (q or {}).get("quencher_name", ""),
                "quencher_compound": (q or {}).get("name", ""),
                "quencher_smiles": (q or {}).get("smiles", ""),

                "K_SV / M-1": sig(k_sv),
                "r2": sig(number(fit, "r2")),
                "n_points_fitted": fit["num_points"],
                "n_points_measured": fit["n_measured"],
                "fit_method": fit["fit_method"],
                "fit_note": fit["note"],
                "tau0 / ns": sig(lifetime),
                "tau0_rsd": sig(number(tau0.get(pc_id), "tau0_rsd")),
                "tau0_consistent": (tau0.get(pc_id) or {}).get("consistent", ""),

                "k_q / M-1 s-1": sig(k_q),
                "log10_k_q": sig(math.log10(k_q)) if (k_q and k_q > 0) else "",
                "quenches": int(quenches),
                "censored": int(censored),
                "k_q_floor / M-1 s-1": sig(floor),

                "q_E_Ox_vs_SCE_exp / V": sig(number(q, "E_Ox_vs_SCE_exp")),
                "q_E_Ox_bound": (q or {}).get("E_Ox_bound", ""),
                "q_E_Red_vs_SCE_exp / V": sig(number(q, "E_Red_vs_SCE_exp")),
                "q_E_Red_bound": (q or {}).get("E_Red_bound", ""),
                "q_epsilon_at_pc_excitation / mM-1 cm-1":
                    sig(nearest_epsilon(absorption.get(q_id), excitation)),
                "pc_excitation / nm": sig(excitation),

                "q_E_T_dft / kcal mol-1": sig(number(dft.get(q_id), "E_T")),
                "q_lambda_S_T_dft / kcal mol-1": sig(number(dft.get(q_id), "lambda_S_T")),
                "q_lambda_T_S_dft / kcal mol-1": sig(number(dft.get(q_id), "lambda_T_S")),
                "q_E_Ox_vs_SCE_calc / V": sig(number(dft.get(q_id), "E_Ox_vs_SCE_calc")),
                "q_E_Red_vs_SCE_calc / V": sig(number(dft.get(q_id), "E_Red_vs_SCE_calc")),
                "q_dvte_mu_md / kcal mol-1": sig(number(dvte.get(q_id), "dvte_mu")),
                "q_dvte_std_md / kcal mol-1": sig(number(dvte.get(q_id), "dvte_std")),
            })

    rows.sort(key=lambda row: (row["pc_id"], row["quencher_id"]))
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    pcs = {row["pc_id"] for row in rows}
    qs = {row["quencher_id"] for row in rows}
    print(f"{len(rows)} pairs -> {OUT.name}   ({len(pcs)} photocatalysts x {len(qs)} quenchers)")
    if len(rows) != len(pcs) * len(qs):
        print(f"  WARNING: not a complete grid, expected {len(pcs) * len(qs)}")
    if no_tau0:
        print(f"  WARNING: no lifetime for photocatalysts {sorted(no_tau0)} — k_q is empty there")

    rates = [row["k_q / M-1 s-1"] for row in rows if row["k_q / M-1 s-1"] != ""]
    print(f"  k_q computed for {len(rates)} pairs; "
          f"{quench_count} above {K_Q_THRESHOLD:.0e}, {len(rates) - quench_count} below")
    print(f"  left-censored (below the detection floor): {censored_count}")
    if rates:
        floors = [row["k_q_floor / M-1 s-1"] for row in rows if row["k_q_floor / M-1 s-1"] != ""]
        print(f"  detection floor ranges {min(floors):.2e} to {max(floors):.2e} M-1 s-1 "
              f"({max(floors) / min(floors):.0f}x, because tau_0 does)")

    for label, field in (("quencher SMILES", "quencher_smiles"),
                         ("quencher DFT E_T", "q_E_T_dft / kcal mol-1"),
                         ("quencher DvTE", "q_dvte_mu_md / kcal mol-1"),
                         ("experimental E_Ox", "q_E_Ox_vs_SCE_exp / V")):
        have = sum(1 for row in rows if row[field] != "")
        print(f"  {label}: {have}/{len(rows)}")


if __name__ == "__main__":
    main()
