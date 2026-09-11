# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Regenerate everything, in dependency order.

    uv run reproduce.py              every stage
    uv run reproduce.py data         one stage
    uv run reproduce.py data benchmark   several stages
    uv run reproduce.py --list       what would run, without running it

Each script is standalone and declares its own dependencies, so this is only a
runner -- it adds no behaviour of its own beyond ordering and reporting. Running any
script by hand gives the same result.

A stage whose inputs are not present is reported as skipped rather than failed, so a
partial checkout (no computational_data_raw, say) still gets as far as it can.

NOT EVERY SCRIPT IN THE TREE IS A STAGE
    `photocatalyst_smiles.py` is deliberately absent. It adds a `smiles` column to the
    photocatalyst file that `compound_records.py` writes from scratch, so it would have
    to run after it -- and nothing downstream reads photocatalyst SMILES, the model
    taking the photocatalyst from DFT instead. Running it here would only add an
    ordering constraint for a column nothing wants. Run it by hand if the column is
    ever needed.
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
    ("data", "data/experimental_data_scripts/absorption_extinction.py",
     "data/experimental_data_raw/quencher_absorption"),
    ("data", "data/experimental_data_scripts/emission_spectra.py",
     "data/experimental_data_raw/pc_emission"),
    ("data", "data/experimental_data_scripts/compound_records.py",
     "data/experimental_data_raw/compound_records"),
    ("data", "data/experimental_data_scripts/cyclic_voltammetry_redox.py",
     "data/experimental_data_raw/quencher_cyclic_voltammetry"),
    ("data", "data/experimental_data_scripts/properties_exp.py",
     "data/experimental_data_raw/compound_records/compound_records_quenchers.csv"),
    # The computational extraction needs the deposit, which is not in git. Each stage
    # checks for a job directory rather than for its folder, because the folders are
    # present in a clone -- they hold the tables these stages write -- and would
    # otherwise send a clone without the deposit into three failures instead of skips.
    ("data", "data/computational_data_scripts/quencher_dft.py",
     "data/computational_data_raw/quencher_adiabatic_DFT/Q1"),
    ("data", "data/computational_data_scripts/quencher_dvte.py",
     "data/computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz"),
    ("data", "data/computational_data_scripts/photocatalyst_dft.py",
     "data/computational_data_raw/photocatalysts/PC_1"),
    ("data", "data/computational_data_scripts/properties_calc.py",
     "data/computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv"),
    ("data", "data/build_exp_dataset.py",
     "data/experimental_data_processed/stern_volmer_fits.csv"),
    ("benchmark", "benchmark/run_benchmark.py", "data/exp_dataset.csv"),
]

STAGES = ["data", "benchmark", "ml", "figures", "paper"]


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
