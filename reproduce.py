# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Regenerate the data and the benchmark, in dependency order.

    uv run reproduce.py                  every stage: data, benchmark, further_analysis, webapp
    uv run reproduce.py data             one stage
    uv run reproduce.py data benchmark   several stages
    uv run reproduce.py --list           what would run, without running it

A stage whose input is missing is skipped.

Run by hand (not stages):
    uv run --extra ml python data/computational_data_scripts/quencher_ml.py
    uv run --extra ml python data/computational_data_scripts/quencher_ml_timing.py
    photocatalyst_curate.py <job tree>
    benchmark/literature/: predict.py, then the comparison scripts (ml extra)
    model training: ml/README.md
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
    ("data", "data/experimental_data_scripts/quencher_redox_potentials.py",
     "data/experimental_data_processed/quencher_redox_literature.csv"),
    ("data", "data/experimental_data_scripts/properties_exp.py",
     "data/experimental_data_processed/quencher_redox_potentials.csv"),
    # Computational stages need the deposit (not in git); each checks for a job directory.
    ("data", "data/computational_data_scripts/quencher_dft.py",
     "data/computational_data_raw/quencher_adiabatic_DFT/Q1"),
    ("data", "data/computational_data_scripts/quencher_geometry.py",
     "data/computational_data_raw/quencher_adiabatic_DFT/Q1"),
    ("data", "data/computational_data_scripts/quencher_dvte.py",
     "data/computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz"),
    ("data", "data/computational_data_scripts/quencher_timings.py",
     "data/computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz"),
    ("data", "data/computational_data_scripts/photocatalyst_dft.py",
     "data/computational_data_raw/photocatalysts_new_ids/PC1"),
    ("data", "data/computational_data_scripts/organophotocatalyst_dft.py",
     "data/computational_data_raw/organophotocatalysts/OPC1"),
    ("data", "data/computational_data_scripts/properties_calc.py",
     "data/computational_data_processed/quencher_dft.csv"),
    ("data", "data/build_exp_dataset.py",
     "data/experimental_data_processed/stern_volmer_fits.csv"),
    ("data", "data/build_exp_dataset_ent.py", "data/exp_dataset.csv"),
    ("benchmark", "benchmark/classification.py", "data/exp_dataset_ent.csv"),
    ("benchmark", "benchmark/entdecker_1.py", "benchmark/results/entdecker_1/quencher_E_T.csv"),
    ("benchmark", "benchmark/cost.py", "data/computational_data_processed/quencher_timings_ml.csv"),
    ("further_analysis", "benchmark/further_analysis/rate_predictions.py",
     "data/exp_dataset_ent.csv"),
    # quencher_classes first: read by the modules below.
    ("further_analysis", "benchmark/further_analysis/quencher_classes.py",
     "data/computational_data_processed/quencher_geometry_S0_T1.csv"),
    *[("further_analysis", f"benchmark/further_analysis/{name}.py",
       "benchmark/further_analysis/results/quencher_classes/quencher_classes.csv")
      for name in ("metrics_extended", "lifetimes", "figure_tables")],
    ("further_analysis", "benchmark/further_analysis/contact_distance.py",
     "data/exp_dataset_ent.csv"),
    ("further_analysis", "benchmark/further_analysis/gate_offset.py",
     "data/exp_dataset_ent.csv"),
    ("further_analysis", "benchmark/further_analysis/error_propagation.py",
     "data/exp_dataset_ent.csv"),
    ("further_analysis", "benchmark/further_analysis/exclusion.py", "data/exp_dataset_ent.csv"),
    ("further_analysis", "benchmark/further_analysis/anharmonicity.py",
     "data/computational_data_processed/quencher_geometry_S0_T1.csv"),
    ("further_analysis", "benchmark/further_analysis/dvte_reorganization.py",
     "data/exp_dataset_ent.csv"),
    ("further_analysis", "benchmark/further_analysis/ml_vs_dft.py",
     "benchmark/further_analysis/results/quencher_classes/quencher_classes.csv"),
    ("webapp", "webapp/scripts/build_photocatalysts.py",
     "data/computational_data_processed/photocatalyst_properties_calc.xlsx"),
]

STAGES = ["data", "benchmark", "further_analysis", "webapp"]


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
