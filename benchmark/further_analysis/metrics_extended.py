# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Extended metrics of the benchmark criteria: the per-pair table, PR AUC, within-
photocatalyst and held-out metrics, the photocatalyst-only baseline, confusion counts and
the split by quencher class.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv   (quencher_classes.py)
Writes  results/metrics_extended/pairs.csv          one row per scored pair: parameters,
                                                    barriers, t_PC, prediction and outcome
                                                    of every shown criterion, class
        results/metrics_extended/metrics.csv        one row per criterion and source
        results/metrics_extended/confusion.csv      TP, FP, TN, FN per criterion (long)
        results/metrics_extended/class_metrics.csv  the same and specificity, recall, MCC
                                                    and accuracy per quencher class

    uv run benchmark/further_analysis/metrics_extended.py

Criterion (key)        source     predicts quenching when                  tuned
adiabatic              dft, ml    dG_EnT <= 0                              -
dvte_2024              dvte       E_T(PC) >= DvTE (p = 0.008, normal)      -
twogate_lifetime       dft, ml    dG_EnT <= 0 and asym barrier <= t_PC,    -
                                  r = r_PC + r_Q of the pair
twogate_lifetime_r7A   dft, ml    the same with r = 7 A                    -
pc_only                -          E_T(PC) >= c                             c
benchmark_raw_lambda   dft, ml    symmetric barrier (geometric-mean        t
                                  lambda) < t

The first five are the classification.py criteria and settings. pc_only chooses c among
the midpoints between the sorted photocatalyst triplet energies (and one value below and
above them); benchmark_raw_lambda chooses t on classification.BARRIER_THRESHOLDS. Both
take the highest MCC on the DFT parameters, the middle setting on a tie, and the ML row
applies the DFT setting. Photocatalyst parameters are DFT throughout.

Scores (ROC AUC, PR AUC): -dG_EnT; E_T(PC) - DvTE; t_PC - barrier with every endergonic
pair below every exergonic one; E_T(PC); -barrier. PR AUC is the average precision
(precision at each distinct score, weighted by the recall gained there). Within-PC: over
the photocatalysts with both outcomes, the median of the per-PC ROC AUCs, their mean
weighted by n_pos x n_neg (pooled), and the median per-PC MCC. LOQO and LOPCO: tuned
settings chosen without the held-out quencher or photocatalyst, pooled over the groups;
for untuned criteria they equal the all-pairs values. Class metrics: the classes of
quencher_classes.py, and flexible and dissociative together; MCC is 0 where undefined.

Energies and distances in the pair table are rounded to 2 decimals, metrics to 3.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "metrics_extended"
CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"
CLASS_GROUPS = {"rigid": ("rigid",), "flexible": ("flexible",), "dissociative": ("dissociative",),
                "flexible + dissociative": ("flexible", "dissociative")}
KC = "kcal mol-1"

ROLE_SHOWN, ROLE_SECONDARY = "shown", "secondary"
ROLE_BASELINE, ROLE_REFERENCE = "PC-only baseline", "benchmark reference"
TWO_GATE_LABEL = "two-gate: dG_EnT <= 0 and asym barrier <= t_PC(tau0, {r})"
LABELS = {
    "adiabatic": "adiabatic dG_EnT <= 0",
    "dvte_2024": "DvTE 2024 (p = 0.008, normal quantile): E_T(PC) >= DvTE",
    "twogate_lifetime": TWO_GATE_LABEL.format(r="r_PC + r_Q"),
    "twogate_lifetime_r7A": TWO_GATE_LABEL.format(r="r = 7 A"),
    "pc_only": "PC-only rule E_T(PC) >= c (c tuned)",
    "benchmark_raw_lambda": "benchmark sym Marcus barrier, raw lambda (t tuned)",
}


@dataclass
class Criterion:
    """A criterion on fixed pairs. predictions: (settings x pairs) with its own
    parameters; tuning: the same with DFT parameters, on which the setting is chosen;
    score: one continuous score per pair, the same for every setting."""

    key: str
    source: str
    role: str
    predictions: np.ndarray
    tuning: np.ndarray
    score: np.ndarray
    settings: tuple = ("",)

    @property
    def name(self):
        return (self.key, self.source)

    @property
    def n_free(self):
        return int(len(self.predictions) > 1)

    def choose(self, y, w):
        """Index of the setting with the highest weighted MCC on the DFT parameters."""
        if len(self.tuning) == 1:
            return 0
        return scoring.best_setting_from_scores(weighted_mcc(self.tuning, y, w))

    def predict(self, y, w=None):
        w = np.ones(len(y)) if w is None else w
        return self.predictions[self.choose(y, w)]


def weighted_mcc(matrix, y, w):
    """MCC of every row of a (settings x pairs) prediction matrix, pairs weighted by w."""
    p = np.atleast_2d(matrix).astype(float)
    yf = np.asarray(y, float)
    tp, fp = p @ (w * yf), p @ (w * (1.0 - yf))
    fn, tn = (w * yf).sum() - tp, (w * (1.0 - yf)).sum() - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (tp * tn - fp * fn) / d, 0.0)


def pc_cuts(pc_et):
    """Candidate cuts c of the photocatalyst-only rule: midpoints between the sorted
    distinct E_T(PC), and one value 1 kcal/mol below and above them."""
    u = np.unique(pc_et)
    return np.r_[u[0] - 1, (u[:-1] + u[1:]) / 2, u[-1] + 1]


def two_gate_setting(variant):
    return classification.candidates("two_gate", variant)[0]


def criteria(df):
    """Every criterion of this module, in table order."""
    out = []
    pc_et = df[pairs.PC_E_T].to_numpy(float)

    def single(key, source, role, setting, parameters):
        p, s = classification.predict(df, setting, parameters)
        return Criterion(key, source, role, p[None, :], p[None, :], s)

    for source in pairs.SOURCES:
        out.append(single("adiabatic", source, ROLE_SHOWN, classification.Setting("adiabatic"), source))
    out.append(single("dvte_2024", "dvte", ROLE_SHOWN,
                      classification.candidates("dvte", "p = 0.008, normal")[0], "dft"))
    for key, variant, role in (("twogate_lifetime", "r from DFT sizes", ROLE_SHOWN),
                               ("twogate_lifetime_r7A", "r = 7 A", ROLE_SECONDARY)):
        for source in pairs.SOURCES:
            out.append(single(key, source, role, two_gate_setting(variant), source))

    cuts = pc_cuts(pc_et)
    p = pc_et[None, :] >= cuts[:, None]
    out.append(Criterion("pc_only", "-", ROLE_BASELINE, p, p, pc_et,
                         tuple(f"c = {c:.2f} kcal/mol" for c in cuts)))

    options = classification.candidates("marcus_barrier", "")
    matrix = {s: np.vstack([classification.predict(df, o, s)[0] for o in options]) for s in pairs.SOURCES}
    for source in pairs.SOURCES:
        score = classification.predict(df, options[0], source)[1]
        out.append(Criterion("benchmark_raw_lambda", source, ROLE_REFERENCE, matrix[source], matrix["dft"],
                             score, tuple(f"t = {o.barrier_threshold:.2f} kcal/mol" for o in options)))
    return out


def average_precision(y, score) -> float:
    """Area under the precision-recall curve as the average precision: the precision at
    each distinct score threshold, weighted by the increase in recall."""
    y = np.asarray(y, bool)
    order = np.argsort(-np.asarray(score, float), kind="mergesort")
    s, t = np.asarray(score, float)[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]     # end of each distinct score
    tp = np.cumsum(t)[last]
    precision = tp / (last + 1)
    recall = tp / t.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def within_pc_mcc(y, predicted, groups):
    """Median per-group MCC over the groups with both outcomes."""
    values = [scoring.metrics(y[groups == g], predicted[groups == g])["MCC"]
              for g in pd.unique(groups) if 0 < y[groups == g].sum() < (groups == g).sum()]
    return float(np.median(values))


def held_out(c, y, groups):
    """Predictions with the setting chosen without each group in turn."""
    predicted = np.zeros(len(y), bool)
    for g in pd.unique(groups):
        inside = groups == g
        predicted[inside] = c.predict(y, (~inside).astype(float))[inside]
    return predicted


def metrics_table(df, crits):
    y = df[pairs.QUENCHES].to_numpy(bool)
    pcs, quenchers = df[pairs.PC_NAME].to_numpy(), df[pairs.Q_NAME].to_numpy()
    rows = []
    for c in crits:
        k = c.choose(y, np.ones(len(y)))
        predicted = c.predictions[k]
        within = scoring.auc_within(y, c.score, pcs)
        loqo = scoring.metrics(y, held_out(c, y, quenchers))
        lopco = scoring.metrics(y, held_out(c, y, pcs))
        rows.append({"criterion": c.key, "source": c.source, "role": c.role, "label": LABELS[c.key],
                     "n_free": c.n_free, "setting": c.settings[k] if c.n_free else "",
                     **scoring.metrics(y, predicted),
                     "ROC_AUC": scoring.auc(y, c.score), "PR_AUC": average_precision(y, c.score),
                     "AUC_within_PC_median": within["median"], "AUC_within_PC_pooled": within["pooled"],
                     "MCC_within_PC_median": within_pc_mcc(y, predicted, pcs),
                     "n_PC_both_classes": int(sum(0 < y[pcs == g].sum() < (pcs == g).sum()
                                                  for g in pd.unique(pcs))),
                     "MCC_LOQO": loqo["MCC"], "MCC_LOPCO": lopco["MCC"],
                     "accuracy_LOQO": loqo["accuracy"], "accuracy_LOPCO": lopco["accuracy"]})
    return pd.DataFrame(rows)


def confusion_table(y, crits):
    rows = []
    for c in crits:
        m = scoring.metrics(y, c.predict(y))
        rows += [{"criterion": c.key, "source": c.source, "role": c.role, "outcome": o, "n_pairs": m[o]}
                 for o in ("TP", "FP", "TN", "FN")]
    return pd.DataFrame(rows)


def class_table(y, classes, crits):
    rows = []
    for c in crits:
        predicted = c.predict(y)
        for group, members in CLASS_GROUPS.items():
            inside = np.isin(classes, members)
            m = scoring.metrics(y[inside], predicted[inside])
            rows.append({"criterion": c.key, "source": c.source, "role": c.role, "quencher_class": group,
                         "n": m["n"], "n_quench": int(y[inside].sum()),
                         **{x: m[x] for x in ("TP", "FP", "TN", "FN", "specificity", "recall", "MCC",
                                              "accuracy")}})
    return pd.DataFrame(rows)


def outcome(y, predicted):
    return np.select([predicted & y, predicted & ~y, ~predicted & ~y], ["TP", "FP", "TN"], "FN")


def column(key, source):
    return f"{key}_{source}" if source in pairs.SOURCES else key


def pair_table(df, classes, crits):
    """The scored pairs with their inputs, barriers, t_PC and the shown predictions."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    r2 = lambda x: np.round(np.asarray(x, float), 2)  # noqa: E731
    t = pd.DataFrame({
        "pc": df[pairs.PC_NAME].to_numpy(), "quencher": df[pairs.Q_NAME].to_numpy(),
        "quencher_class": classes, "quenches": y.astype(int),
        pairs.K_SV: df[pairs.K_SV].to_numpy(float), pairs.TAU0: df[pairs.TAU0].to_numpy(float),
        pairs.K_Q: df[pairs.K_Q].to_numpy(float),
        f"E_T_pc_dft / {KC}": r2(df[pairs.PC_E_T]),
        f"E_T_q_dft / {KC}": r2(df[f"q_E_T_dft / {KC}"]), f"E_T_q_ml / {KC}": r2(df[f"q_E_T_ml / {KC}"]),
        f"lambda_S0_T1_pc_dft / {KC}": r2(df[pairs.PC_LAMBDA_S0_T1]),
        f"lambda_T1_S0_pc_dft / {KC}": r2(df[pairs.PC_LAMBDA_T1_S0]),
    })
    lam = {s: pairs.lambdas(df, s) for s in pairs.SOURCES}
    for s in pairs.SOURCES:
        t[f"lambda_S0_T1_q_{s} / {KC}"] = r2(lam[s][2])
        t[f"lambda_T1_S0_q_{s} / {KC}"] = r2(lam[s][3])
    for s in pairs.SOURCES:
        t[f"dG_EnT_{s} / {KC}"] = r2(pairs.driving_force(df, s))
    for s in pairs.SOURCES:
        t[f"barrier_asym_{s} / {KC}"] = r2(theory.marcus_barrier_asymmetric(
            pairs.driving_force(df, s), *theory.pair_lambdas(*lam[s])))
    tau = pairs.tau0_seconds(df)
    r = pairs.contact_distances(df)
    k_sv = classification.K_SV_CRIT
    t["contact_distance / A"] = r2(r)
    t[f"t_PC / {KC}"] = r2(theory.lifetime_barrier(tau, k_sv, theory.escape_over_frequency(r)))
    t[f"t_PC_7A / {KC}"] = r2(theory.lifetime_barrier(tau, k_sv, theory.escape_over_frequency(
        theory.CONTACT_DISTANCE)))
    t[f"dvte_0.008_q / {KC}"] = r2(df[pairs.q_dvte("0.008", "normal")])
    shown = [c for c in crits if c.role in (ROLE_SHOWN, ROLE_SECONDARY)]
    for c in shown:
        t[f"pred_{column(c.key, c.source)}"] = c.predict(y).astype(int)
    for c in shown:
        t[f"outcome_{column(c.key, c.source)}"] = outcome(y, c.predict(y))
    return t


def load():
    """(scored pairs, quencher class per pair, criteria)."""
    if not CLASSES.exists():
        raise FileNotFoundError(f"{CLASSES} not found; run benchmark/further_analysis/quencher_classes.py first")
    df = pairs.scored(pairs.load())
    cls = pd.read_csv(CLASSES).set_index("id")["proposed_class"]
    classes = df[pairs.Q_NAME].map(cls).to_numpy()
    if pd.isna(classes).any():
        raise KeyError("a scored quencher has no class in quencher_classes.csv")
    return df, classes, criteria(df)


def main():
    df, classes, crits = load()
    y = df[pairs.QUENCHES].to_numpy(bool)
    tables = {"pairs.csv": pair_table(df, classes, crits),
              "metrics.csv": scoring.rounded(metrics_table(df, crits)),
              "confusion.csv": confusion_table(y, [c for c in crits if c.role != ROLE_REFERENCE]),
              "class_metrics.csv": scoring.rounded(class_table(y, classes,
                                                               [c for c in crits if c.role != ROLE_REFERENCE]))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(RESULTS / name, index=False, lineterminator="\n")

    m = tables["metrics.csv"]
    print(m[["criterion", "source", "setting", "accuracy", "MCC", "ROC_AUC", "PR_AUC",
             "AUC_within_PC_pooled", "MCC_LOQO", "MCC_LOPCO"]].to_string(index=False))
    fd = tables["class_metrics.csv"]
    print()
    print(fd[fd.quencher_class == "flexible + dissociative"][
        ["criterion", "source", "n", "FP", "accuracy", "MCC", "recall", "specificity"]].to_string(index=False))
    print(f"\n{len(df)} pairs -> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
