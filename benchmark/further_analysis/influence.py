# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
How the benchmark metrics move when one quencher or one photocatalyst is removed.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/influence/metrics.csv   one row per criterion row of the benchmark and
                                        removed molecule: metrics without that molecule's
                                        pairs, and the change in MCC

    uv run benchmark/further_analysis/influence.py

The settings are those of the benchmark on all pairs; they are not re-chosen after a
removal. A removed molecule of "none" is the full set.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402

RESULTS = HERE / "results" / "influence"
GROUPS = {"quencher": pairs.Q_NAME, "photocatalyst": pairs.PC_NAME}


def influence(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for criterion, variant, source, setting in classification.chosen_settings(df):
        predicted, _ = classification.predict(df, setting, source)
        label = {"criterion": criterion, "parameters": source, "variant": variant}
        full = scoring.metrics(y, predicted)
        rows.append({**label, "removed": "none", "name": "", **full, "delta_MCC": 0.0})
        for kind, column in GROUPS.items():
            for name in df[column].unique():
                keep = (df[column] != name).to_numpy()
                m = scoring.metrics(y[keep], predicted[keep])
                rows.append({**label, "removed": kind, "name": name, **m,
                             "delta_MCC": m["MCC"] - full["MCC"]})
    return pd.DataFrame(rows)


def main():
    df = pairs.scored(pairs.load())
    table = influence(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(table).to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n")

    removed = table[table["removed"] != "none"]
    spread = removed.groupby(["criterion", "parameters", "variant", "removed"])["delta_MCC"]
    print(spread.agg(["min", "max"]).round(3).to_string())
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
