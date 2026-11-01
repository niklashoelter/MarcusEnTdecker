# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Distribution of vertical singlet-triplet gaps (DvTE) per quencher from Milo trajectories.

Reads   ../computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz
Writes  ../computational_data_processed/quencher_dvte.csv
        ../computational_data_processed/quencher_dvte_samples.csv.gz

    uv run data/computational_data_scripts/quencher_dvte.py

Per step (`milo_q<n>/md_<n>/step_<n>/`, inside the tar.xz, read as a stream): singlet
(`output_1.out`) and triplet (`output_3.out`) single points at the same geometry.

    gap                 (E_triplet - E_singlet) * 627.509, kcal/mol
    dvte_mu, dvte_std   mean and standard deviation of all gaps of a quencher
    dvte_<p>            p-quantile of a normal distribution, mu + Phi^-1(p) * std
    dvte_<p>_kde        p-quantile of a Gaussian KDE of the gaps, bandwidth by Scott's rule

Statistics from unrounded gaps; kcal/mol rounded to 2 decimals when written, Hartree as
printed by ORCA.
"""

import csv
import gzip
import io
import math
import re
import statistics
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "computational_data_raw" / "quencher_adiabatic_DvTE"
OUT_DIR = Path(__file__).resolve().parent.parent / "computational_data_processed"
ARCHIVE = RAW / "quenchers_vertical_gaps.tar.xz"
OUT = OUT_DIR / "quencher_dvte.csv"
OUT_SAMPLES = OUT_DIR / "quencher_dvte_samples.csv.gz"

H_TO_KCAL = 627.509
THRESHOLDS = (0.005, 0.008, 0.01, 0.015)
DECIMALS = 2                 # for kcal/mol

MEMBER = re.compile(r"milo_q(\d+)/md_(\d+)/step_(\d+)/output_([13])\.out$")
ENERGY = re.compile(rb"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

COLUMNS = (["quencher_id", "n_samples", "n_trajectories", "dvte_mu", "dvte_std"]
           + [f"dvte_{p}" for p in THRESHOLDS]
           + [f"dvte_{p}_kde" for p in THRESHOLDS])


def rounded(x):
    """A value in kcal/mol, to DECIMALS; blank if absent."""
    return "" if x is None else round(x, DECIMALS)


def normal_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > 1 - 0.02425:
        return -normal_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def kde_quantile(samples, p, bandwidth):
    """p-quantile of a Gaussian KDE, by bisection on its CDF."""
    if not samples or bandwidth <= 0:
        return None
    ordered = sorted(samples)
    lo = ordered[0] - 5 * bandwidth
    hi = ordered[-1] + 5 * bandwidth

    def cdf(x):
        total = 0.0
        for s in ordered:
            total += 0.5 * (1.0 + math.erf((x - s) / (bandwidth * math.sqrt(2.0))))
        return total / len(ordered)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return 0.5 * (lo + hi)


def collect_gaps():
    """quencher_id -> {(md, step): {multiplicity: energy}}, read from the archive."""
    energies = defaultdict(dict)
    seen = 0
    started = time.time()
    with tarfile.open(ARCHIVE, "r:xz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            m = MEMBER.search(member.name)
            if not m:
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            found = ENERGY.findall(handle.read())
            if not found:
                continue
            qid, md, step, mult = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            energies[qid].setdefault((md, step), {})[mult] = float(found[-1])
            seen += 1
            if seen % 20000 == 0:
                print(f"    {seen} outputs read, {time.time() - started:.0f} s elapsed",
                      flush=True)
    return energies


def main():
    if not ARCHIVE.exists():
        sys.exit(f"missing input: {ARCHIVE}")

    print(f"streaming {ARCHIVE.name} ({ARCHIVE.stat().st_size / 2**30:.1f} GB) ...", flush=True)
    energies = collect_gaps()

    rows, sample_rows = [], []
    for qid in sorted(energies):
        gaps, trajectories = [], set()
        for (md, step), pair in sorted(energies[qid].items()):
            if "1" in pair and "3" in pair:
                gap = (pair["3"] - pair["1"]) * H_TO_KCAL
                gaps.append(gap)
                trajectories.add(md)
                sample_rows.append({"quencher_id": qid, "md_run": md, "step": step,
                                    "E_singlet_Eh": pair["1"], "E_triplet_Eh": pair["3"],
                                    "gap_kcal": rounded(gap)})
        if len(gaps) < 2:
            print(f"  Q{qid}: only {len(gaps)} paired steps, skipped")
            continue

        mu = statistics.fmean(gaps)
        sd = statistics.stdev(gaps)
        bandwidth = sd * len(gaps) ** (-1 / 5)      # Scott's rule
        row = {"quencher_id": qid, "n_samples": len(gaps),
               "n_trajectories": len(trajectories),
               "dvte_mu": rounded(mu), "dvte_std": rounded(sd)}
        for p in THRESHOLDS:
            row[f"dvte_{p}"] = rounded(mu + normal_ppf(p) * sd)
            row[f"dvte_{p}_kde"] = rounded(kde_quantile(gaps, p, bandwidth))
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # mtime=0, empty filename: reproducible gzip.
    with open(OUT_SAMPLES, "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["quencher_id", "md_run", "step",
                                                "E_singlet_Eh", "E_triplet_Eh", "gap_kcal"])
        writer.writeheader()
        writer.writerows(sample_rows)

    print(f"\n{len(rows)} quenchers, {len(sample_rows)} sampled geometries -> {OUT.name}")
    if rows:
        counts = [r["n_samples"] for r in rows]
        print(f"  samples per quencher: min {min(counts)}, median "
              f"{statistics.median(counts):.0f}, max {max(counts)}")


if __name__ == "__main__":
    main()
