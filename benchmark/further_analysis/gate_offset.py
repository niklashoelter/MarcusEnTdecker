# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
The position of the energy gate: the criteria with their first gate moved from 0 to c.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv
Writes  results/gate_offset/sweep.csv          metrics of every criterion at every c
        results/gate_offset/optimum.csv        the c with the highest MCC, and the MCC at
                                               c = 0, at that c and at C_WINDOW
        results/gate_offset/heldout.csv        nested held-out MCC with c re-tuned without
                                               the held-out photocatalyst
        results/gate_offset/window_stats.csv   quenching fractions in C_WINDOW < dG <= 0,
                                               dG <= C_WINDOW and dG > 0
        results/gate_offset/flips_by_group.csv the pairs whose prediction changes between
                                               c = 0 and c = C_WINDOW, by quencher class

    uv run benchmark/further_analysis/gate_offset.py

Criterion   first gate            second gate
adiabatic   dG_EnT <= c           -
dvte        DvTE - E_T(PC) <= c   -                  (DvTE p = 0.008, normal quantile)
two_gate    dG_EnT <= c           asymmetric Marcus barrier <= t_PC, r = r_PC + r_Q (DFT),
                                  r = 7 A (ML)

c = 0 is the benchmark. Only the first gate moves; the barrier uses the unshifted dG_EnT.
adiabatic and two_gate with DFT and ML quencher parameters, dvte with DFT. Continuous
scores as in classification.py.

Optimum c: highest MCC on C_GRID, middle one on a tie (scoring.best_setting_from_scores).
Held out (LOPCO): c chosen without each photocatalyst, applied to its pairs; pooled
held-out predictions scored once.
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
import theory  # noqa: E402

RESULTS = HERE / "results" / "gate_offset"
QUENCHER_CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"

C_GRID = np.round(np.arange(-5.0, 2.0001, 0.1), 1)     # kcal/mol
C_WINDOW = -1.5                                         # kcal/mol
DVTE = ("0.008", "normal")
# (criterion, parameters)
CRITERIA = (("adiabatic", "dft"), ("adiabatic", "ml"), ("dvte", "dft"),
            ("two_gate", "dft"), ("two_gate", "ml"))
FLIP_CRITERIA = (("two_gate", "dft"), ("two_gate", "ml"), ("adiabatic", "dft"))


class Offsets:
    """The criteria of the scored pairs with a movable first gate."""

    def __init__(self, df):
        self.df = df
        self.y = df[pairs.QUENCHES].to_numpy(bool)
        self.dG = {s: pairs.driving_force(df, s) for s in pairs.SOURCES}
        self.dvte_gap = (df[pairs.q_dvte(*DVTE)].to_numpy(float)
                         - df[pairs.PC_E_T].to_numpy(float))
        self.t_pc = {s: theory.lifetime_barrier(
            pairs.tau0_seconds(df), classification.K_SV_CRIT,
            theory.escape_over_frequency(classification.contact_distance(df, s)))
            for s in pairs.SOURCES}
        self.barrier = {s: theory.marcus_barrier_asymmetric(
            self.dG[s], *theory.pair_lambdas(*pairs.lambdas(df, s))) for s in pairs.SOURCES}
        # (criterion, parameters) -> (len(C_GRID) x pairs) predictions
        self.grid = {key: np.vstack([self.predict(*key, c)[0] for c in C_GRID]) for key in CRITERIA}

    def predict(self, criterion, source, c):
        """(predicted quenching, continuous score) with the first gate at c."""
        if criterion == "adiabatic":
            return self.dG[source] <= c, -self.dG[source]
        if criterion == "dvte":
            return self.dvte_gap <= c, -self.dvte_gap
        if criterion == "two_gate":
            margin = self.t_pc[source] - self.barrier[source]
            gate = self.dG[source] <= c
            return (gate & (margin >= 0),
                    np.where(gate, margin, margin - classification.ENDERGONIC_OFFSET))
        raise ValueError(f"unknown criterion {criterion!r}")

    def metrics(self, predicted, score):
        within = scoring.auc_within(self.y, score, self.df[pairs.PC_NAME].to_numpy())
        return {**scoring.metrics(self.y, predicted), "AUC_within_PC_pooled": within["pooled"]}


def index_of(c):
    return int(np.flatnonzero(np.isclose(C_GRID, c))[0])


def mcc_weighted(y, matrix, weights=None):
    """MCC of every row of a (settings x pairs) prediction matrix, with pair weights."""
    matrix = np.atleast_2d(np.asarray(matrix, float))
    weights = np.ones(matrix.shape[1]) if weights is None else np.asarray(weights, float)
    positive, negative = y * weights, (~y) * weights
    tp, fp = matrix @ positive, matrix @ negative
    return scoring._mcc(tp, fp, negative.sum() - fp, positive.sum() - tp)


def sweep(o: Offsets):
    rows = []
    for criterion, source in CRITERIA:
        for c in C_GRID:
            rows.append({"criterion": criterion, "parameters": source, "c / kcal mol-1": c,
                         **o.metrics(*o.predict(criterion, source, c))})
    return pd.DataFrame(rows)


def optimum(o: Offsets):
    zero, window = index_of(0.0), index_of(C_WINDOW)
    rows = []
    for key in CRITERIA:
        mcc = mcc_weighted(o.y, o.grid[key])
        best = scoring.best_setting_from_scores(mcc)
        near = C_GRID[mcc >= mcc[best] - 0.01]
        rows.append({"criterion": key[0], "parameters": key[1], "MCC_c0": mcc[zero],
                     "c_best / kcal mol-1": C_GRID[best], "MCC_best": mcc[best],
                     "MCC_c_window": mcc[window], "gain_best": mcc[best] - mcc[zero],
                     "gain_c_window": mcc[window] - mcc[zero],
                     "c within 0.01 of the best MCC, min / kcal mol-1": near.min(),
                     "c within 0.01 of the best MCC, max / kcal mol-1": near.max()})
    return pd.DataFrame(rows)


def held_out_predictions(o: Offsets, key, groups):
    """Pooled held-out (predicted, score) and the c chosen for each group."""
    predicted, score = np.zeros(len(o.y), bool), np.zeros(len(o.y))
    chosen = []
    for group in pd.unique(groups):
        inside = groups == group
        c = C_GRID[scoring.best_setting_from_scores(
            mcc_weighted(o.y, o.grid[key], (~inside).astype(float)))]
        chosen.append(c)
        p, s = o.predict(*key, c)
        predicted[inside], score[inside] = p[inside], s[inside]
    return predicted, score, np.array(chosen)


def held_out(o: Offsets):
    groups = o.df[pairs.PC_NAME].to_numpy()
    rows = []
    for key in CRITERIA:
        at_zero = o.metrics(*o.predict(*key, 0.0))
        predicted, score, chosen = held_out_predictions(o, key, groups)
        m = o.metrics(predicted, score)
        rows.append({"criterion": key[0], "parameters": key[1], "scheme": "LOPCO",
                     "MCC_c0": at_zero["MCC"], "MCC_heldout": m["MCC"],
                     "gain_MCC": m["MCC"] - at_zero["MCC"],
                     "accuracy_c0": at_zero["accuracy"], "accuracy_heldout": m["accuracy"],
                     "gain_accuracy": m["accuracy"] - at_zero["accuracy"],
                     "AUC_within_PC_pooled_c0": at_zero["AUC_within_PC_pooled"],
                     "AUC_within_PC_pooled_heldout": m["AUC_within_PC_pooled"],
                     "c_chosen_min / kcal mol-1": chosen.min(),
                     "c_chosen_median / kcal mol-1": np.median(chosen),
                     "c_chosen_max / kcal mol-1": chosen.max(), "n_folds": len(chosen)})
    return pd.DataFrame(rows)


def window_stats(o: Offsets):
    k_sv = o.df[pairs.K_SV].to_numpy(float)
    rows = []
    for source in pairs.SOURCES:
        dG = o.dG[source]
        second_gate = o.t_pc[source] >= o.barrier[source]
        for region, inside in ((f"window {C_WINDOW:g} < dG <= 0", (dG > C_WINDOW) & (dG <= 0)),
                               (f"dG <= {C_WINDOW:g}", dG <= C_WINDOW), ("dG > 0", dG > 0)):
            non_quenching = inside & ~o.y
            passed = inside & second_gate
            rows.append({"parameters": source, "region": region, "n": int(inside.sum()),
                         "n_quench": int((inside & o.y).sum()),
                         "frac_quench": (inside & o.y).sum() / max(inside.sum(), 1),
                         "frac_quench_second_gate_passed": (passed & o.y).sum() / max(passed.sum(), 1),
                         "n_second_gate_passed": int(passed.sum()),
                         "median K_SV non-quenching / M-1": (np.nanmedian(k_sv[non_quenching])
                                                             if non_quenching.any() else np.nan)})
    return pd.DataFrame(rows)


def flips(o: Offsets, quencher_class):
    """The changes between c = 0 and C_WINDOW by quencher class."""
    zero, window = index_of(0.0), index_of(C_WINDOW)
    classes = o.df[pairs.Q_NAME].map(quencher_class).to_numpy()
    by_class = []
    for key in FLIP_CRITERIA:
        before, after = o.grid[key][zero], o.grid[key][window]
        flip = before != after
        to_correct, to_wrong = flip & (after == o.y), flip & (after != o.y)
        table = pd.DataFrame({"quencher_class": classes, "flip": flip, "FP_to_TN": to_correct,
                              "TP_to_FN": to_wrong})
        grouped = table.groupby("quencher_class")
        counts = pd.DataFrame({"n_pairs": grouped.size(), "n_flipped": grouped["flip"].sum(),
                               "FP_to_TN": grouped["FP_to_TN"].sum(),
                               "TP_to_FN": grouped["TP_to_FN"].sum()})
        counts.loc["all"] = counts.sum()
        counts["net_correct"] = counts["FP_to_TN"] - counts["TP_to_FN"]
        counts = counts.reset_index()
        counts.insert(0, "parameters", key[1])
        counts.insert(0, "criterion", key[0])
        by_class.append(counts)
    return pd.concat(by_class, ignore_index=True)


def read_quencher_classes():
    if not QUENCHER_CLASSES.exists():
        raise FileNotFoundError(f"{QUENCHER_CLASSES} not found; run "
                                "benchmark/further_analysis/quencher_classes.py first")
    return pd.read_csv(QUENCHER_CLASSES).set_index("id")["proposed_class"]


def main():
    o = Offsets(pairs.scored(pairs.load()))
    quencher_class = read_quencher_classes()
    unclassified = sorted(set(o.df[pairs.Q_NAME]) - set(quencher_class.dropna().index))
    if unclassified:
        raise KeyError(f"no quencher class for {', '.join(unclassified)} in {QUENCHER_CLASSES.name}")
    tables = {"sweep.csv": sweep(o), "optimum.csv": optimum(o), "heldout.csv": held_out(o),
              "window_stats.csv": window_stats(o), "flips_by_group.csv": flips(o, quencher_class)}

    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        scoring.rounded(table).to_csv(RESULTS / name, index=False, lineterminator="\n")

    shown = ["criterion", "parameters", "MCC_c0", "c_best / kcal mol-1", "MCC_best",
             "MCC_c_window"]
    print(tables["optimum.csv"][shown].round(3).to_string(index=False))
    print()
    print(tables["heldout.csv"][["criterion", "parameters", "MCC_heldout",
                                 "c_chosen_median / kcal mol-1"]].round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
