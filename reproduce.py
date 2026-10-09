# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Regenerate the data and the benchmark, in dependency order.

    uv run reproduce.py                  every stage: data, benchmark, further_analysis
    uv run reproduce.py data             one stage
    uv run reproduce.py data benchmark   several stages
    uv run reproduce.py --list           what would run, without running it

Each script is standalone and declares its own dependencies, so this is only a
runner -- it adds no behaviour of its own beyond ordering and reporting. Running any
script by hand gives the same result.

A stage whose inputs are not present is reported as skipped rather than failed, so a
partial checkout (no computational_data_raw, say) still gets as far as it can.

Not every script is a stage:
    `photocatalyst_curate.py` builds the photocatalyst deposit from a job tree given as an
    argument, so it is run by hand.

    `quencher_ml.py` needs the ml extra (torch, torch-geometric, rdkit). Run it by hand:
    `uv run --extra ml python data/computational_data_scripts/quencher_ml.py`. The workbook
    it writes is in git.

    `quencher_ml_timing.py` measures the model's inference time on the machine it runs on,
    also with the ml extra: `uv run --extra ml python
    data/computational_data_scripts/quencher_ml_timing.py`. The table it writes is in git.

    `compound_identification.py` and `properties_workbook.py` in data/, and `pairs.py`,
    `theory.py` and `scoring.py` in benchmark/, are imported modules.

The model training is not a stage either; ml/README.md describes it.
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
    ("data", "data/experimental_data_scripts/quencher_redox_potentials.py",
     "data/experimental_data_raw/quencher_cyclic_voltammetry/quencher_redox_literature.csv"),
    ("data", "data/experimental_data_scripts/properties_exp.py",
     "data/experimental_data_raw/quencher_cyclic_voltammetry/quencher_redox_potentials.csv"),
    # The computational extraction needs the deposit, which is not in git. Each stage
    # checks for a job directory rather than for its folder, because the folders are
    # present in a clone -- they hold the tables these stages write -- and would
    # otherwise send a clone without the deposit into three failures instead of skips.
    ("data", "data/computational_data_scripts/quencher_dft.py",
     "data/computational_data_raw/quencher_adiabatic_DFT/Q1"),
    ("data", "data/computational_data_scripts/quencher_dvte.py",
     "data/computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz"),
    ("data", "data/computational_data_scripts/quencher_timings.py",
     "data/computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz"),
    ("data", "data/computational_data_scripts/photocatalyst_dft.py",
     "data/computational_data_raw/photocatalysts/PC_1"),
    ("data", "data/computational_data_scripts/organophotocatalyst_dft.py",
     "data/computational_data_raw/organophotocatalysts/OPC1"),
    ("data", "data/computational_data_scripts/properties_calc.py",
     "data/computational_data_raw/quencher_adiabatic_DFT/quencher_dft.csv"),
    ("data", "data/build_exp_dataset.py",
     "data/experimental_data_processed/stern_volmer_fits.csv"),
    ("data", "data/build_exp_dataset_ent.py", "data/exp_dataset.csv"),
    ("benchmark", "benchmark/classification.py", "data/exp_dataset_ent.csv"),
    ("benchmark", "benchmark/cost.py", "data/computational_data_processed/quencher_timings_ml.csv"),
    *[("further_analysis", f"benchmark/further_analysis/{name}.py", "data/exp_dataset_ent.csv")
      for name in ("influence", "held_out", "within_groups", "adiabatic_variants",
                   "barrier_variants", "rate_variants", "rate_predictions")],
]

STAGES = ["data", "benchmark", "further_analysis"]


def run(script):
    """Run one script under uv. Returns (ok, seconds)."""
    started = time.time()
    result = subprocess.run(["uv", "run", "--quiet", str(ROOT / script)], cwd=ROOT)
    return result.returncode == 0, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
