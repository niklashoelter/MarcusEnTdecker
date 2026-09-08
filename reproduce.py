# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Regenerate everything, in dependency order.

    uv run reproduce.py              every stage
    uv run reproduce.py data         one stage
    uv run reproduce.py data dataset several stages
    uv run reproduce.py --list       what would run, without running it

Each script is standalone and declares its own dependencies, so this is only a
runner -- it adds no behaviour of its own beyond ordering and reporting. Running any
script by hand gives the same result.

A stage whose inputs are not present is reported as skipped rather than failed, so a
partial checkout (no computational_data_raw, say) still gets as far as it can.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (stage, script, the input that must exist for it to be runnable)
PIPELINE = [
    ("data", "data/experimental_data_scripts/stern_volmer_intensities.py",
     "data/experimental_data_raw/stern_volmer"),
    ("data", "data/experimental_data_scripts/stern_volmer_fits.py",
     "data/experimental_data_processed/stern_volmer_intensities.csv"),
    ("data", "data/experimental_data_scripts/lifetimes_tcspc.py",
     "data/experimental_data_raw/pc_tcspc"),
    ("data", "data/experimental_data_scripts/lifetimes_tau0.py",
     "data/experimental_data_processed/lifetimes_tcspc.csv"),
    ("data", "data/experimental_data_scripts/absorption_extinction.py",
     "data/experimental_data_raw/quencher_absorption"),
    ("data", "data/experimental_data_scripts/emission_spectra.py",
     "data/experimental_data_raw/pc_emission"),
    ("data", "data/experimental_data_scripts/compound_records.py",
     "data/experimental_data_raw/compound_records"),
    ("data", "data/computational_data_scripts/quencher_dft.py",
     "data/computational_data_raw/quencher_adiabatic_DFT"),
    ("data", "data/computational_data_scripts/quencher_dvte.py",
     "data/computational_data_raw/quencher_adiabatic_DvTE"),
    ("data", "data/computational_data_scripts/photocatalyst_dft.py",
     "data/computational_data_raw/photocatalysts"),
    ("dataset", "dataset/build_dataset.py",
     "data/experimental_data_processed/stern_volmer_fits.csv"),
]

STAGES = ["data", "dataset", "benchmark", "ml", "figures", "paper"]


def run(script):
    """Run one script under uv. Returns (ok, seconds)."""
    started = time.time()
    result = subprocess.run(["uv", "run", "--quiet", str(ROOT / script)], cwd=ROOT)
    return result.returncode == 0, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", default=None,
                        help=f"stages to run (default: all). One of: {', '.join(STAGES)}")
    parser.add_argument("--list", action="store_true", help="show the plan, run nothing")
    args = parser.parse_args()

    wanted = set(args.stages) if args.stages else set(STAGES)
    unknown = wanted - set(STAGES)
    if unknown:
        sys.exit(f"unknown stage(s): {', '.join(sorted(unknown))}\n"
                 f"choose from: {', '.join(STAGES)}")

    plan = [(stage, script, needs) for stage, script, needs in PIPELINE if stage in wanted]
    if not plan:
        print("nothing to do — those stages have no scripts yet")
        return

    if args.list:
        for stage, script, needs in plan:
            state = "ready" if (ROOT / needs).exists() else f"SKIP (missing {needs})"
            print(f"  [{stage}] {script}  -- {state}")
        return

    failed, skipped, ran = [], [], []
    for stage, script, needs in plan:
        if not (ROOT / needs).exists():
            print(f"\n--- SKIP [{stage}] {script}\n    missing input: {needs}")
            skipped.append(script)
            continue
        print(f"\n--- [{stage}] {script}", flush=True)
        ok, seconds = run(script)
        (ran if ok else failed).append(script)
        print(f"    {'ok' if ok else 'FAILED'} in {seconds:.1f} s")

    print(f"\n{len(ran)} ran, {len(skipped)} skipped, {len(failed)} failed")
    for script in failed:
        print(f"  FAILED  {script}")
    for script in skipped:
        print(f"  skipped {script}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
