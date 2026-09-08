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
        ../data/computational_data_processed/photocatalyst_dft.csv
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

THE DRIVING FORCES
    Both channels a triplet photocatalyst can quench through, per pair, all
    quencher-centric and all negative when exergonic:

        dG_EnT     = E_T(Q) - E_T(PC)
        dG_SET_ox  = F * (E_Ox(Q) - E(PC*/PC-))     Q oxidized, PC* is the oxidant
        dG_SET_red = F * (E(PC+/PC*) - E_Red(Q))    Q reduced,  PC* is the reductant

    with F = 23.0605 kcal mol-1 V-1. `dG_SET` is whichever of the two is more
    accessible, and `dG_SET_channel` says which, because "the SET driving force" is
    meaningless without knowing which way the electron went.

    Potentials are named by their couple rather than by direction. E(PC*/PC-) is the
    potential at which PC* *oxidizes* a substrate -- the reduction potential of the
    excited state. Naming it "the excited-state oxidation potential" is how the sign
    gets flipped.

    Both SET columns are computed twice, from the measured quencher potentials
    (`_exp`) and from the computed ones (`_calc`), and the two are never merged. They
    are different quantities with different provenance, and which one an analysis used
    is part of what it has to report.

BOUNDED POTENTIALS PROPAGATE INTO BOUNDED DRIVING FORCES
    Nineteen quenchers have a redox potential recorded as a limit. Both limits point
    the same way once they reach a driving force: an oxidation recorded as `> 2.0` puts
    a floor under dG_SET_ox, and a reduction recorded as `< -2.10` puts a floor under
    dG_SET_red. So `dG_SET_ox_bound = lower` means SET is *at least* this endergonic --
    the true value can only be worse. That is a usable statement, and it is the only
    honest one; reading the number at face value would understate how inaccessible the
    channel is, in the direction that wrongly excludes pairs from the energy-transfer
    set.

WHAT IS STILL NOT HERE
    The photocatalyst reorganization energies are carried but no combined lambda is
    formed. Marcus theory needs the pair's lambda, and how the two partners' terms
    combine is a modelling decision that belongs in benchmark/, not in the table.
"""

import collections
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

# Faraday constant in the units the driving forces are reported in.
F_KCAL_PER_V = 23.060548

# Photocatalysts measured but excluded from the dataset, with the reason.
#
# This is a scientific decision taken outside this script, recorded here so that the
# 1296 rows the Stern-Volmer fits would otherwise produce and the 1248 written below
# differ by something a reader can see and disagree with. It is NOT a data-quality
# filter and no threshold produces it: PC_24 is unremarkable on both fit quality and
# lifetime provenance, and three photocatalysts that are kept have worse median r2.
EXCLUDED_PHOTOCATALYSTS = {
    24: "discarded from the screen; marked 'discard' in the photocatalyst record",
}

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
    "label_class", "threshold_resolvable",
    # quencher, experimental
    "q_E_Ox_vs_SCE_exp / V", "q_E_Ox_bound", "q_E_Red_vs_SCE_exp / V", "q_E_Red_bound",
    "q_redox_source_type", "q_redox_doi",
    "q_epsilon_at_pc_excitation / mM-1 cm-1", "pc_excitation / nm",
    # quencher, computed
    "q_E_T_dft / kcal mol-1", "q_lambda_S_T_dft / kcal mol-1", "q_lambda_T_S_dft / kcal mol-1",
    "q_E_Ox_vs_SCE_calc / V", "q_E_Red_vs_SCE_calc / V",
    "q_dvte_mu_md / kcal mol-1", "q_dvte_std_md / kcal mol-1",
    # photocatalyst, computed
    "pc_E_T_dft / kcal mol-1",
    "pc_lambda_S0_to_T1_dft / kcal mol-1", "pc_lambda_T1_to_S0_dft / kcal mol-1",
    "pc_E_PCplus_PC / V", "pc_E_PC_PCminus / V",
    "pc_E_PCstar_PCminus / V", "pc_E_PCplus_PCstar / V",
    # driving forces
    "dG_EnT / kcal mol-1",
    "dG_SET_ox_exp / kcal mol-1", "dG_SET_ox_exp_bound",
    "dG_SET_red_exp / kcal mol-1", "dG_SET_red_exp_bound",
    "dG_SET_exp / kcal mol-1", "dG_SET_exp_bound", "dG_SET_exp_channel",
    "dG_SET_ox_calc / kcal mol-1", "dG_SET_red_calc / kcal mol-1",
    "dG_SET_calc / kcal mol-1", "dG_SET_calc_channel",
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


def label_class(quenches, censored, floor):
    """'positive', 'negative' or 'indeterminate' at K_Q_THRESHOLD.

    `quenches` compares k_q against a single global threshold. `k_q_floor` is where
    this photocatalyst's detection limit actually sits, 25/tau_0, and for a
    short-lived photocatalyst that floor lies well ABOVE the global threshold -- 33x
    above it for PC10. The two disagree, and which one is right depends on the pair:

      positive       measured above the threshold AND the Stern-Volmer detection
                     criterion was met, so quenching was actually observed
      negative       measured below the threshold AND this photocatalyst's floor is
                     below the threshold too, so quenching at that rate would have
                     been seen had it occurred
      indeterminate  everything else -- either labelled a quencher on a measurement
                     that never passed detection, or labelled a non-quencher by a
                     photocatalyst that could not have detected the threshold anyway

    An indeterminate row is not a bad measurement. It is a measurement that does not
    answer this particular question, and treating it as a negative asserts something
    the experiment did not establish.
    """
    if floor is None:
        return ""
    if quenches and not censored:
        return "positive"
    if not quenches and floor <= K_Q_THRESHOLD:
        return "negative"
    return "indeterminate"


def better_set_channel(ox, red):
    """The more accessible of the two SET channels: (value, 'ox'|'red'), or (None, '').

    "More accessible" is the less positive driving force. If only one channel has the
    potentials it needs, that one is returned -- a missing channel is not evidence that
    the other one is the preferred route, so the column says which one it is reporting.
    """
    available = [(value, name) for value, name in ((ox, "ox"), (red, "red"))
                 if value is not None]
    if not available:
        return None, ""
    return min(available)


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
    pc_dft = load(COMPUTATIONAL / "photocatalyst_dft.csv", "pc_id")

    rows, no_tau0, quench_count, censored_count = [], set(), 0, 0
    excluded_rows = 0

    with fits_path.open(encoding="utf-8", newline="") as fh:
        for fit in csv.DictReader(fh):
            pc_id, q_id = int(fit["pc_id"]), int(fit["quencher_id"])
            if pc_id in EXCLUDED_PHOTOCATALYSTS:
                excluded_rows += 1
                continue

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

            pcc = pc_dft.get(pc_id)
            pc_e_t = number(pcc, "E_T / kcal mol-1")
            pc_star_ox = number(pcc, "E_PCstar_PCminus / V")   # PC* as the oxidant
            pc_star_red = number(pcc, "E_PCplus_PCstar / V")   # PC* as the reductant

            q_e_t = number(dft.get(q_id), "E_T")
            dG_ent = (q_e_t - pc_e_t) if None not in (q_e_t, pc_e_t) else None

            def set_driving_forces(e_ox, e_red):
                """(dG_SET_ox, dG_SET_red) from one pair of quencher potentials."""
                ox = (F_KCAL_PER_V * (e_ox - pc_star_ox)
                      if None not in (e_ox, pc_star_ox) else None)
                red = (F_KCAL_PER_V * (pc_star_red - e_red)
                       if None not in (e_red, pc_star_red) else None)
                return ox, red

            dG_ox_exp, dG_red_exp = set_driving_forces(
                number(q, "E_Ox_vs_SCE_exp"), number(q, "E_Red_vs_SCE_exp"))
            dG_ox_calc, dG_red_calc = set_driving_forces(
                number(dft.get(q_id), "E_Ox_vs_SCE_calc"),
                number(dft.get(q_id), "E_Red_vs_SCE_calc"))

            # A bound on either potential becomes a lower bound on the driving force:
            # the quencher was not oxidized below the recorded value, so SET is at
            # least this endergonic. Both directions land on `lower` for that reason.
            ox_bound = "lower" if (q or {}).get("E_Ox_bound") else ""
            red_bound = "lower" if (q or {}).get("E_Red_bound") else ""

            dG_set_exp, exp_channel = better_set_channel(dG_ox_exp, dG_red_exp)
            dG_set_calc, calc_channel = better_set_channel(dG_ox_calc, dG_red_calc)
            set_bound = {"ox": ox_bound, "red": red_bound}.get(exp_channel, "")

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
                "label_class": label_class(quenches, censored, floor),
                "threshold_resolvable": "" if floor is None else int(floor <= K_Q_THRESHOLD),

                "q_E_Ox_vs_SCE_exp / V": sig(number(q, "E_Ox_vs_SCE_exp")),
                "q_E_Ox_bound": (q or {}).get("E_Ox_bound", ""),
                "q_E_Red_vs_SCE_exp / V": sig(number(q, "E_Red_vs_SCE_exp")),
                "q_E_Red_bound": (q or {}).get("E_Red_bound", ""),
                "q_redox_source_type": (q or {}).get("redox_source_type", ""),
                "q_redox_doi": (q or {}).get("redox_doi", ""),
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

                "pc_E_T_dft / kcal mol-1": sig(pc_e_t),
                "pc_lambda_S0_to_T1_dft / kcal mol-1":
                    sig(number(pcc, "lambda_S0_to_T1 / kcal mol-1")),
                "pc_lambda_T1_to_S0_dft / kcal mol-1":
                    sig(number(pcc, "lambda_T1_to_S0 / kcal mol-1")),
                "pc_E_PCplus_PC / V": sig(number(pcc, "E_PCplus_PC / V")),
                "pc_E_PC_PCminus / V": sig(number(pcc, "E_PC_PCminus / V")),
                "pc_E_PCstar_PCminus / V": sig(pc_star_ox),
                "pc_E_PCplus_PCstar / V": sig(pc_star_red),

                "dG_EnT / kcal mol-1": sig(dG_ent),
                "dG_SET_ox_exp / kcal mol-1": sig(dG_ox_exp),
                "dG_SET_ox_exp_bound": ox_bound if dG_ox_exp is not None else "",
                "dG_SET_red_exp / kcal mol-1": sig(dG_red_exp),
                "dG_SET_red_exp_bound": red_bound if dG_red_exp is not None else "",
                "dG_SET_exp / kcal mol-1": sig(dG_set_exp),
                "dG_SET_exp_bound": set_bound,
                "dG_SET_exp_channel": exp_channel,
                "dG_SET_ox_calc / kcal mol-1": sig(dG_ox_calc),
                "dG_SET_red_calc / kcal mol-1": sig(dG_red_calc),
                "dG_SET_calc / kcal mol-1": sig(dG_set_calc),
                "dG_SET_calc_channel": calc_channel,
            })

    rows.sort(key=lambda row: (row["pc_id"], row["quencher_id"]))
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    pcs = {row["pc_id"] for row in rows}
    qs = {row["quencher_id"] for row in rows}
    print(f"{len(rows)} pairs -> {OUT.name}   ({len(pcs)} photocatalysts x {len(qs)} quenchers)")
    for pc_id, reason in EXCLUDED_PHOTOCATALYSTS.items():
        print(f"  excluded PC{pc_id}: {reason}")
    if excluded_rows:
        print(f"  {excluded_rows} fitted pairs dropped by that exclusion, not by any threshold")
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

    classes = collections.Counter(row["label_class"] for row in rows)
    print(f"\n  label_class: {classes['positive']} positive, {classes['negative']} negative, "
          f"{classes['indeterminate']} indeterminate")
    unresolvable = {row["pc_id"] for row in rows if row["threshold_resolvable"] == 0}
    if unresolvable:
        print(f"  WARNING: {len(unresolvable)} of {len({r['pc_id'] for r in rows})} photocatalysts "
              f"have a detection floor ABOVE the {K_Q_THRESHOLD:.0e} threshold,")
        print(f"           so a 'does not quench' from them is not evidence of k_q < "
              f"{K_Q_THRESHOLD:.0e}. Only {classes['negative']} of the "
              f"{sum(1 for r in rows if r['quenches'] == 0)} apparent non-quenchers are established.")

    sources = collections.Counter(row["q_redox_source_type"] for row in rows)
    print(f"  redox provenance: " + ", ".join(f"{v} {k}" for k, v in sources.most_common() if k))

    for label, field in (("quencher SMILES", "quencher_smiles"),
                         ("quencher DFT E_T", "q_E_T_dft / kcal mol-1"),
                         ("quencher DvTE", "q_dvte_mu_md / kcal mol-1"),
                         ("experimental E_Ox", "q_E_Ox_vs_SCE_exp / V"),
                         ("photocatalyst DFT E_T", "pc_E_T_dft / kcal mol-1"),
                         ("dG_EnT", "dG_EnT / kcal mol-1"),
                         ("dG_SET from measured potentials", "dG_SET_exp / kcal mol-1"),
                         ("dG_SET from computed potentials", "dG_SET_calc / kcal mol-1")):
        have = sum(1 for row in rows if row[field] != "")
        print(f"  {label}: {have}/{len(rows)}")

    exergonic = sum(1 for row in rows
                    if row["dG_EnT / kcal mol-1"] != "" and row["dG_EnT / kcal mol-1"] < 0)
    bounded = sum(1 for row in rows if row["dG_SET_exp_bound"])
    channels = collections.Counter(row["dG_SET_exp_channel"] for row in rows)
    print(f"\n  dG_EnT exergonic for {exergonic} pairs")
    print(f"  dG_SET (measured) preferred channel: "
          + ", ".join(f"{v} {k}" for k, v in channels.most_common() if k))
    print(f"  ...of which {bounded} rest on a bounded potential and are lower bounds only")


if __name__ == "__main__":
    main()
