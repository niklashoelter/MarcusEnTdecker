# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Barrier criteria with every barrier model, with and without dG_EnT <= 0, and with a fixed
threshold of 5 kcal/mol.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/barrier_variants/settings.csv   every DFT setting
        results/barrier_variants/best.csv       the best DFT setting of each variant and the
                                                same setting with ML parameters

    uv run benchmark/further_analysis/barrier_variants.py

Barrier models: Marcus symmetric, Marcus asymmetric, Rehm-Weller (theory.BARRIERS).
Conditions: barrier < threshold; dG_EnT <= 0 and barrier < threshold.
Variants:
    lambda raw                alpha = 1, threshold from the benchmark grid
    lambda scaled             alpha and threshold from the benchmark grids
    threshold 5, lambda raw   alpha = 1, threshold 5 kcal/mol (nothing chosen)
    threshold 5, scaled       alpha from the benchmark grid, threshold 5 kcal/mol
"""

import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "barrier_variants"
FIXED_THRESHOLD = 5.0       # kcal/mol
CONDITIONS = {"barrier only": False, "dG <= 0 and barrier": True}


def variants(model, exergonic):
    base = classification.Setting("barrier", barrier_model=model, requires_exergonic=exergonic)
    grid = classification.BARRIER_THRESHOLDS
    alphas = classification.ALPHAS
    return {
        "lambda raw": [replace(base, barrier_threshold=t) for t in grid],
        "lambda scaled": [replace(base, alpha=a, barrier_threshold=t) for a in alphas for t in grid],
        "threshold 5, lambda raw": [replace(base, barrier_threshold=FIXED_THRESHOLD)],
        "threshold 5, lambda scaled": [replace(base, alpha=a, barrier_threshold=FIXED_THRESHOLD)
                                       for a in alphas],
    }


def run(df):
    every, best = [], []
    for model in theory.BARRIERS:
        for condition, exergonic in CONDITIONS.items():
            for variant, options in variants(model, exergonic).items():
                label = f"{model} barrier, {condition}"
                rows, best_rows, _ = classification.tune(df, label, variant, options)
                every += rows
                best += best_rows
    return pd.DataFrame(every), pd.DataFrame(best)


def main():
    df = pairs.scored(pairs.load())
    every, best = run(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    every.to_csv(RESULTS / "settings.csv", index=False, lineterminator="\n")
    best.to_csv(RESULTS / "best.csv", index=False, lineterminator="\n")

    shown = ["criterion", "parameters", "variant", "alpha", "barrier_threshold / kcal mol-1",
             "recall", "specificity", "MCC", "AUC"]
    print(best[shown].rename(columns={"barrier_threshold / kcal mol-1": "threshold"})
          .round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
