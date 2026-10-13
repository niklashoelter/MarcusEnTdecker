# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Benchmark metrics within each photocatalyst and within each quencher.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/within_groups/metrics.csv   one row per criterion row of the benchmark and
                                            group: metrics and AUC over that group's pairs
        results/within_groups/summary.csv   median MCC and AUC over the groups that
                                            contain both quenching and non-quenching pairs

    uv run benchmark/further_analysis/within_groups.py

The settings are those of the benchmark on all pairs. AUC is blank for a group with only
one class.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402

RESULTS = HERE / "results" / "within_groups"
GROUPS = {"photocatalyst": pairs.PC_NAME, "quencher": pairs.Q_NAME}


def within_groups(df):
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for criterion, variant, source, setting in classification.chosen_settings(df):
        predicted, score = classification.predict(df, setting, source)
        for kind, column in GROUPS.items():
            for name in df[column].unique():
                inside = (df[column] == name).to_numpy()
                rows.append({"criterion": criterion, "parameters": source, "variant": variant,
                             "group": kind, "name": name,
                             "n_quench": int(y[inside].sum()),
                             **scoring.metrics(y[inside], predicted[inside]),
                             "AUC": scoring.auc(y[inside], score[inside])})
    return pd.DataFrame(rows)


def summarize(table):
    mixed = table[(table["n_quench"] > 0) & (table["n_quench"] < table["n"])]
    keys = ["criterion", "parameters", "variant", "group"]
    return (mixed.groupby(keys, sort=False)
            .agg(groups=("name", "size"), median_MCC=("MCC", "median"), median_AUC=("AUC", "median"))
            .reset_index())


def main():
    df = pairs.scored(pairs.load())
    table = within_groups(df)
    summary = summarize(table)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(table).to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n")
    scoring.rounded(summary).to_csv(RESULTS / "summary.csv", index=False, lineterminator="\n")
    print(summary.round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
