# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Fitted TCSPC traces -> one excited-state lifetime per photocatalyst.

Reads   ../experimental_data_processed/lifetimes_tcspc.csv
        ../experimental_data_processed/stern_volmer_fits.csv   (screen membership only)
Writes  ../experimental_data_processed/lifetimes_tau0.csv

tau_0 is the quantity that turns a Stern-Volmer constant into a rate constant,
k_q = K_SV / tau_0, so it sets the scale of every rate in the paper. It is measured
here rather than inherited from a curated table: every one of the 27 screened
photocatalysts has at least one usable TCSPC trace, so nothing needs to be filled in
from elsewhere.

Several photocatalysts were measured more than once, sometimes on different time
ranges, and those repeats do not always agree. This script states the selection rule
explicitly and reports the disagreement rather than hiding it behind a single number.

SELECTION
    1. Duplicate files are dropped -- `lifetimes_tcspc.py` marks byte-identical and
       re-labelled copies of one measurement with `duplicate_of`. They are the same
       measurement and must not be treated as repeats.

    2. Traces that could not be fitted are dropped.

    3. A trace is dropped if its acquisition window is shorter than
       MIN_SPAN_IN_LIFETIMES * tau. A window that does not contain several lifetimes
       cannot separate the decay from the baseline, and the fit collapses onto
       whatever fits inside the window. This removes exactly one trace across the
       whole set: PC4's 200 ns range, which returns 51 ns against roughly 1100 ns
       from the same compound's microsecond ranges.

    4. Where a photocatalyst was measured more than once, tau_0 is the LONGEST of the
       surviving measurements, not their mean.

WHY THE LONGEST AND NOT THE MEAN
    These are triplet lifetimes measured under argon, and the dominant error is
    incomplete degassing. Residual oxygen quenches the triplet, so it can only ever
    shorten the measured lifetime -- the error is one-sided. Averaging a well-degassed
    measurement together with a poorly degassed one therefore does not cancel the
    error, it splits the difference with a known-low value and biases tau_0 downward.

    PC22 is the case that shows it: three measurements at 3095, 3135 and 3305 ns, the
    longest of which is the one the experimenter labelled `betterDegas`. The ordering
    is exactly what oxygen quenching predicts.

    The spread between repeats is also one to two orders of magnitude larger than the
    standard error within any one fit -- PC22 scatters by 111 ns across measurements
    each fitted to about 9 ns -- so the disagreement is systematic, not statistical,
    and no weighting over within-fit errors would address it.

UNCERTAINTY
    With repeats, `tau0_std_ns` is the standard deviation across them. Given the
    argument above this is a one-sided systematic spread rather than a symmetric error
    bar: the true value sits at or above `tau0_ns`, not centred on it. With a single
    measurement it is that fit's own standard error, which is a lower bound on the
    real uncertainty, flagged by `n_measurements = 1`.

AGREEMENT WITH THE CURATED TABLE
    This rule reproduces 26 of the 27 curated tau_0 values in
    spectroscopy_data_processed/dataset/properties_photocatalysts.xlsx exactly, which
    establishes that the curated table was itself derived from these traces.

    PC4 is the exception and is deliberately not reconciled here. Its curated value is
    1494.54 ns, which matches none of its measurements: the two usable ones give
    1228.49 and 1039.14 ns, both at unusually low amplitude (133 and 123 counts,
    against 900+ for a typical trace here). Taking the longest gives 1228.49 ns, 18 %
    below the curated figure, which scales k_q for all 48 of PC4's pairs by 1.22. The
    curated value appears to come from a re-fit that is not in the data. PC4 needs a
    decision, not a rule, and `consistent = 0` marks it as such.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent / "experimental_data_processed"
FITS = PROCESSED / "lifetimes_tcspc.csv"
SCREEN = PROCESSED / "stern_volmer_fits.csv"
OUT = PROCESSED / "lifetimes_tau0.csv"

# An acquisition window must contain at least this many lifetimes for the fit to
# determine tau rather than the window determining it.
MIN_SPAN_IN_LIFETIMES = 5.0

# Flag a photocatalyst when its repeats scatter by more than this.
RSD_WARN = 0.10

SIGFIGS = 6

COLUMNS = [
    "pc_id", "tau0_ns", "tau0_std_ns", "tau0_rsd",
    "n_measurements", "n_traces_rejected", "in_screen", "consistent",
    "trace_used", "traces", "note",
]


def sig(x, n=SIGFIGS):
    if x is None:
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def read_traces():
    """pc_id -> [accepted trace dicts], and pc_id -> [(trace, why it was rejected)]."""
    accepted = defaultdict(list)
    rejected = defaultdict(list)

    with FITS.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pc_id = int(row["pc_id"])
            name = row["trace"]

            if row["duplicate_of"]:
                rejected[pc_id].append((name, f"duplicate of {row['duplicate_of']}"))
                continue
            if not row["tau_ns"]:
                rejected[pc_id].append((name, "not fitted"))
                continue

            tau = float(row["tau_ns"])
            span = float(row["span_ns"]) if row["span_ns"] else 0.0
            if tau <= 0:
                rejected[pc_id].append((name, "non-positive lifetime"))
                continue
            if span < MIN_SPAN_IN_LIFETIMES * tau:
                rejected[pc_id].append(
                    (name, f"window spans {span / tau:.1f} lifetimes, "
                           f"below the {MIN_SPAN_IN_LIFETIMES:.0f} required"))
                continue

            accepted[pc_id].append({
                "trace": name,
                "tau": tau,
                "stderr": float(row["tau_stderr_ns"]) if row["tau_stderr_ns"] else None,
            })

    return accepted, rejected


def screened_ids():
    """Photocatalyst ids that appear in the Stern-Volmer screen."""
    if not SCREEN.exists():
        return None
    with SCREEN.open(encoding="utf-8", newline="") as fh:
        return {int(row["pc_id"]) for row in csv.DictReader(fh)}


def main():
    if not FITS.exists():
        sys.exit(f"missing input: {FITS}\nrun lifetimes_tcspc.py first")

    accepted, rejected = read_traces()
    in_screen = screened_ids()

    rows = []
    for pc_id in sorted(set(accepted) | set(rejected)):
        traces = accepted.get(pc_id, [])
        rejects = rejected.get(pc_id, [])
        screened = "" if in_screen is None else int(pc_id in in_screen)

        if not traces:
            rows.append({
                "pc_id": pc_id, "tau0_ns": "", "tau0_std_ns": "", "tau0_rsd": "",
                "n_measurements": 0, "n_traces_rejected": len(rejects),
                "in_screen": screened, "consistent": "", "trace_used": "", "traces": "",
                "note": "; ".join(f"rejected {n}: {why}" for n, why in rejects),
            })
            continue

        taus = [t["tau"] for t in traces]
        # Longest, not mean: incomplete degassing only ever shortens a triplet
        # lifetime, so the error across repeats is one-sided. See the module docstring.
        chosen = max(traces, key=lambda t: t["tau"])
        tau0 = chosen["tau"]

        if len(taus) > 1:
            spread = statistics.stdev(taus)
            basis = f"longest of {len(taus)} measurements"
        else:
            spread = traces[0]["stderr"]
            basis = "single measurement, within-fit standard error"

        rsd = (spread / tau0) if (spread is not None and tau0) else None
        consistent = "" if len(taus) < 2 else int(rsd is not None and rsd <= RSD_WARN)

        notes = [basis]
        if consistent == 0:
            notes.append(f"measurements scatter by {rsd:.0%}, above the "
                         f"{RSD_WARN:.0%} threshold — needs a decision, not a rule")
        notes += [f"rejected {n}: {why}" for n, why in rejects]

        rows.append({
            "pc_id": pc_id,
            "tau0_ns": sig(tau0),
            "tau0_std_ns": sig(spread),
            "tau0_rsd": sig(rsd),
            "n_measurements": len(taus),
            "n_traces_rejected": len(rejects),
            "in_screen": screened,
            "consistent": consistent,
            "trace_used": chosen["trace"],
            "traces": " | ".join(t["trace"] for t in traces),
            "note": "; ".join(notes),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    usable = [r for r in rows if r["tau0_ns"] != ""]
    print(f"{len(usable)} photocatalysts with a lifetime -> {OUT.name}")

    if in_screen is not None:
        missing = sorted(in_screen - {r["pc_id"] for r in usable})
        print(f"  screened: {len(in_screen)}; "
              f"{'all have a lifetime' if not missing else f'MISSING {missing}'}")

    repeats = [r for r in usable if r["n_measurements"] > 1]
    print(f"  measured more than once: {len(repeats)}")
    for r in repeats:
        mark = "   <-- inconsistent" if r["consistent"] == 0 else ""
        print(f"    PC{r['pc_id']}: {r['n_measurements']} measurements, took "
              f"{r['tau0_ns']} ns, spread {r['tau0_rsd']:.1%}{mark}")

    print(f"  traces rejected: {sum(r['n_traces_rejected'] for r in rows)}")


if __name__ == "__main__":
    main()
