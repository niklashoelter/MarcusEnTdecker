# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Held-out scores of the tuned benchmark criteria.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/held_out/summary.csv   pooled held-out metrics per criterion, source and
                                       grouping, next to the MCC on all pairs
        results/held_out/chosen.csv    the setting chosen for each held-out group

    uv run benchmark/further_analysis/held_out.py

For each quencher (or photocatalyst) in turn, the setting is chosen on the DFT
parameters of all other pairs and applied to that molecule's pairs, with DFT and with ML
parameters. The held-out predictions of all groups are pooled into one confusion matrix.
Only criteria with more than one candidate setting are included.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402

RESULTS = HERE / "results" / "held_out"
GROUPINGS = {"quencher": pairs.Q_NAME, "photocatalyst": pairs.PC_NAME}


def held_out(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    summary, chosen = [], []
    for criterion, (_, _, variants, _, given) in classification.CRITERIA.items():
        for variant in variants:
            options = classification.candidates(criterion, variant)
            if len(options) < 2:
                continue
            sources = ("dft", *given)
            matrices = {s: np.vstack([classification.predict(df, o, s)[0] for o in options])
                        for s in sources}
            for grouping, column in GROUPINGS.items():
                groups = df[column].to_numpy()
                predictions = {s: np.zeros(len(y), bool) for s in sources}
                picks = set()
                for group in pd.unique(groups):
                    inside = groups == group
                    pick = scoring.best_setting(y[~inside], matrices["dft"][:, ~inside])
                    picks.add(pick)
                    for s in sources:
                        predictions[s][inside] = matrices[s][pick, inside]
                    o = options[pick]
                    chosen.append({"criterion": criterion, "variant": variant, "held_out_by": grouping,
                                   "group": group,
                                   "barrier_threshold / kcal mol-1": o.barrier_threshold})
                best = scoring.best_setting(y, matrices["dft"])
                for s in sources:
                    summary.append({"criterion": criterion, "parameters": s, "variant": variant,
                                    "held_out_by": grouping,
                                    "MCC_all_pairs": float(scoring.mcc_rows(y, matrices[s][[best]])[0]),
                                    **scoring.metrics(y, predictions[s]),
                                    "settings_chosen": len(picks)})
    return pd.DataFrame(summary), pd.DataFrame(chosen)


def main():
    df = pairs.scored(pairs.load())
    summary, chosen = held_out(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(summary).to_csv(RESULTS / "summary.csv", index=False, lineterminator="\n")
    scoring.rounded(chosen).to_csv(RESULTS / "chosen.csv", index=False, lineterminator="\n")

    shown = ["criterion", "parameters", "variant", "held_out_by", "MCC_all_pairs", "MCC",
             "settings_chosen"]
    print(summary[shown].round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
