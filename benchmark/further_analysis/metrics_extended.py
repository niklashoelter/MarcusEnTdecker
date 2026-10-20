# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Extended metrics of the benchmark criteria: the per-pair table, PR AUC, within-
photocatalyst metrics and the split by quencher class.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv   (quencher_classes.py)
Writes  results/metrics_extended/pairs.csv          one row per scored pair: parameters,
                                                    barriers, t_PC, prediction and outcome
                                                    of every shown criterion, class
        results/metrics_extended/metrics.csv        one row per criterion and source
        results/metrics_extended/class_metrics.csv  the same and specificity, recall, MCC
                                                    and accuracy per quencher class

    uv run benchmark/further_analysis/metrics_extended.py

Criterion (key)        source     predicts quenching when
adiabatic              dft, ml    dG_EnT <= 0
dvte_2024              dvte       E_T(PC) >= DvTE (p = 0.008, normal)
twogate_lifetime       dft, ml    dG_EnT <= 0 and asym barrier <= t_PC,
                                  r = r_PC + r_Q of the pair
twogate_lifetime_r7A   dft, ml    the same with r = 7 A

Settings of classification.py, untuned (n_free = 0). Photocatalyst parameters DFT. Role
"shown": twogate_lifetime with DFT, twogate_lifetime_r7A with ML
(classification.TWO_GATE_DEFAULT); the other two "secondary".

Scores: -dG_EnT; E_T(PC) - DvTE; t_PC - barrier, endergonic pairs below all exergonic ones.
PR AUC = average precision. Within-PC (photocatalysts with both outcomes): median ROC AUC,
mean weighted by n_pos x n_neg (pooled), median MCC. Class metrics: classes of
quencher_classes.py plus flexible + dissociative; MCC = 0 where undefined.

Pair table: energies and distances 2 decimals; metrics 3.
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
TWO_GATE_LABEL = "two-gate: dG_EnT <= 0 and asym barrier <= t_PC(tau0, {r})"
LABELS = {
    "adiabatic": "adiabatic dG_EnT <= 0",
    "dvte_2024": "DvTE 2024 (p = 0.008, normal quantile): E_T(PC) >= DvTE",
    "twogate_lifetime": TWO_GATE_LABEL.format(r="r_PC + r_Q"),
    "twogate_lifetime_r7A": TWO_GATE_LABEL.format(r="r = 7 A"),
}


@dataclass
class Criterion:
    """A criterion on fixed pairs: its prediction and one continuous score per pair."""

    key: str
    source: str
    role: str
    prediction: np.ndarray
    score: np.ndarray

    @property
    def name(self):
        return (self.key, self.source)


def two_gate_setting(variant):
    return classification.candidates("two_gate", variant)[0]


def criteria(df):
    """Every criterion of this module, in table order."""
    out = []

    def single(key, source, role, setting, parameters):
        p, s = classification.predict(df, setting, parameters)
        return Criterion(key, source, role, p, s)

    for source in pairs.SOURCES:
        out.append(single("adiabatic", source, ROLE_SHOWN, classification.Setting("adiabatic"), source))
    out.append(single("dvte_2024", "dvte", ROLE_SHOWN,
                      classification.candidates("dvte", "p = 0.008, normal")[0], "dft"))
    for key, variant in (("twogate_lifetime", "r from DFT sizes"), ("twogate_lifetime_r7A", "r = 7 A")):
        for source in pairs.SOURCES:
            role = ROLE_SHOWN if classification.TWO_GATE_DEFAULT[source] == variant else ROLE_SECONDARY
            out.append(single(key, source, role, two_gate_setting(variant), source))
    return out


def average_precision(y, score) -> float:
    """Average precision: precision at each distinct score threshold, weighted by the recall gain."""
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


def metrics_table(df, crits):
    y = df[pairs.QUENCHES].to_numpy(bool)
    pcs = df[pairs.PC_NAME].to_numpy()
    rows = []
    for c in crits:
        predicted = c.prediction
        within = scoring.auc_within(y, c.score, pcs)
        rows.append({"criterion": c.key, "source": c.source, "role": c.role, "label": LABELS[c.key],
                     "n_free": 0, "setting": "",
                     **scoring.metrics(y, predicted),
                     "ROC_AUC": scoring.auc(y, c.score), "PR_AUC": average_precision(y, c.score),
                     "AUC_within_PC_median": within["median"], "AUC_within_PC_pooled": within["pooled"],
                     "MCC_within_PC_median": within_pc_mcc(y, predicted, pcs),
                     "n_PC_both_classes": int(sum(0 < y[pcs == g].sum() < (pcs == g).sum()
                                                  for g in pd.unique(pcs)))})
    return pd.DataFrame(rows)


def class_table(y, classes, crits):
    rows = []
    for c in crits:
        predicted = c.prediction
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
    for c in crits:
        t[f"pred_{column(c.key, c.source)}"] = c.prediction.astype(int)
    for c in crits:
        t[f"outcome_{column(c.key, c.source)}"] = outcome(y, c.prediction)
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
              "class_metrics.csv": scoring.rounded(class_table(y, classes, crits))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(RESULTS / name, index=False, lineterminator="\n")

    m = tables["metrics.csv"]
    print(m[["criterion", "source", "setting", "accuracy", "MCC", "ROC_AUC", "PR_AUC",
             "AUC_within_PC_pooled"]].to_string(index=False))
    fd = tables["class_metrics.csv"]
    print()
    print(fd[fd.quencher_class == "flexible + dissociative"][
        ["criterion", "source", "n", "FP", "accuracy", "MCC", "recall", "specificity"]].to_string(index=False))
    print(f"\n{len(df)} pairs -> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
