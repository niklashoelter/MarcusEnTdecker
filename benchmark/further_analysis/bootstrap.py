# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Two-way cluster bootstrap of the benchmark metrics and of the differences between
criteria.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv   (through metrics_extended.py)
Writes  results/bootstrap/bootstrap.csv             per criterion: MCC, ROC AUC and pooled
                                                    within-PC AUC with their intervals
                                                    ('value' rows), and differences to other
                                                    criteria ('difference' rows)
        results/bootstrap/deltas_vs_adiabatic.csv   accuracy, MCC and pooled within-PC AUC
        results/bootstrap/deltas_vs_dvte.csv        of a criterion minus adiabatic (DFT) or
                                                    minus DvTE, with intervals
        results/bootstrap/ml_vs_dft.csv             ML minus DFT quencher parameters for the
                                                    adiabatic and two-gate criteria

    uv run benchmark/further_analysis/bootstrap.py

The criteria are those of metrics_extended.py without the plain-lambda barrier. In each of
REPLICATES replicates, the photocatalysts and the quenchers are drawn with replacement
(numpy default_rng(SEED)); a pair's weight is the number of copies of its photocatalyst
times that of its quencher. Metrics are the weighted ones; the photocatalyst-only rule is
re-tuned on the weighted pairs in every replicate. ROC AUC: ties count one half; within-
PC AUC: pairs compared only within their photocatalyst, each comparison weighted.

Columns: point = value on all pairs; median, lo95, hi95 = median and 2.5 / 97.5 %
percentiles over the replicates (of the difference, for differences, computed on the
same replicate); P_gt_0 = fraction of replicates with a positive difference. Values are
rounded to 3 decimals.

Difference rows of bootstrap.csv: each criterion minus DvTE and minus the adiabatic
criterion of the same source (the photocatalyst-only rule and DvTE minus both adiabatic
criteria), and two-gate DFT minus two-gate ML.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import metrics_extended as mx  # noqa: E402
import pairs  # noqa: E402

RESULTS = HERE / "results" / "bootstrap"
REPLICATES = 2000
SEED = 20261009

DVTE = ("dvte_2024", "dvte")
METRICS = {"MCC": "MCC", "AUC": "ROC_AUC", "wAUC": "AUC_within_PC_pooled"}
EXTENDED = (("adiabatic", "dft"), ("adiabatic", "ml"), DVTE,
            ("twogate_lifetime", "dft"), ("twogate_lifetime", "ml"))
DELTA_METRICS = ("accuracy", "MCC", "AUC_within_PC_pooled")
ML_METRICS = ("accuracy", "MCC", "ROC_AUC", "AUC_within_PC_pooled", "precision", "recall", "specificity")


class WeightedAUC:
    """ROC AUC with pair weights, over all pairs or pooled within groups. Ties count 1/2."""

    def __init__(self, y, score, groups=None):
        self.y = np.asarray(y, float)
        g = np.zeros(len(self.y), int) if groups is None else np.asarray(groups)
        r = rankdata(score, method="dense").astype(np.int64)
        key = g.astype(np.int64) * (r.max() + 1) + r
        u, self.inv = np.unique(key, return_inverse=True)
        self.k = len(u)
        group = (u // (r.max() + 1)).astype(int)                # group of each (group, rank)
        self.first = np.r_[True, group[1:] != group[:-1]]
        self.gid = np.cumsum(self.first) - 1

    def __call__(self, w):
        wp = np.bincount(self.inv, w * self.y, self.k)
        wn = np.bincount(self.inv, w * (1.0 - self.y), self.k)
        cn = np.cumsum(wn)
        start = (cn - wn)[self.first]                           # negatives before each group
        below = cn - wn - start[self.gid]
        num = (wp * (below + 0.5 * wn)).sum()
        den = (np.bincount(self.gid, wp) * np.bincount(self.gid, wn)).sum()
        return float(num / den) if den > 0 else np.nan


def counts(p, y, w):
    yf = y.astype(float)
    tp, fp = p @ (w * yf), p @ (w * (1.0 - yf))
    return tp, fp, (w * (1.0 - yf)).sum() - fp, (w * yf).sum() - tp


class Bootstrap:
    """The criteria on one set of pairs and their metrics for a weight vector."""

    def __init__(self, df, crits):
        self.y = df[pairs.QUENCHES].to_numpy(bool)
        pcs, quenchers = df[pairs.PC_NAME].to_numpy(), df[pairs.Q_NAME].to_numpy()
        self.pc_index = np.unique(pcs, return_inverse=True)[1]
        self.q_index = np.unique(quenchers, return_inverse=True)[1]
        self.crits = {c.name: c for c in crits}
        self.auc = {c.name: WeightedAUC(self.y, c.score) for c in crits}
        self.wauc = {c.name: WeightedAUC(self.y, c.score, self.pc_index) for c in crits}
        self.fixed = {name: self.crits[name].predict(self.y).astype(float) for name in EXTENDED}

    def weights(self, rng):
        n_pc, n_q = self.pc_index.max() + 1, self.q_index.max() + 1
        cp = np.bincount(rng.integers(0, n_pc, n_pc), minlength=n_pc)
        cq = np.bincount(rng.integers(0, n_q, n_q), minlength=n_q)
        return (cp[self.pc_index] * cq[self.q_index]).astype(float)

    def stats(self, w):
        """MCC, AUC and within-PC AUC of every criterion, the setting re-tuned on w."""
        out = {}
        for name, c in self.crits.items():
            p = c.predictions[c.choose(self.y, w)]
            out[("MCC", name)] = float(mx.weighted_mcc(p, self.y, w)[0])
            out[("AUC", name)] = self.auc[name](w)
            out[("wAUC", name)] = self.wauc[name](w)
        return out

    def extended(self, w):
        """Point metrics of the untuned criteria of EXTENDED."""
        out = {}
        for name in EXTENDED:
            tp, fp, tn, fn = counts(self.fixed[name], self.y, w)
            d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            out[("accuracy", name)] = (tp + tn) / (tp + fp + tn + fn)
            out[("precision", name)] = tp / (tp + fp) if tp + fp > 0 else np.nan
            out[("recall", name)] = tp / (tp + fn) if tp + fn > 0 else np.nan
            out[("specificity", name)] = tn / (tn + fp) if tn + fp > 0 else np.nan
            out[("MCC", name)] = (tp * tn - fp * fn) / d if d > 0 else 0.0
            out[("ROC_AUC", name)] = self.auc[name](w)
            out[("AUC_within_PC_pooled", name)] = self.wauc[name](w)
        return out

    def run(self, replicates=REPLICATES, seed=SEED):
        """(point stats, replicate stats, point extended, replicate extended)."""
        rng = np.random.default_rng(seed)
        w1 = np.ones(len(self.y))
        reps, xreps = [], []
        for _ in range(replicates):
            w = self.weights(rng)
            reps.append(self.stats(w))
            xreps.append(self.extended(w))
        return self.stats(w1), pd.DataFrame(reps), self.extended(w1), pd.DataFrame(xreps)


def interval(v):
    return {"median": v.median(), "lo95": v.quantile(0.025), "hi95": v.quantile(0.975)}


def references(name):
    """The criteria `name` is compared with in bootstrap.csv."""
    key, source = name
    out = [DVTE]
    if source == "-":
        out += [("adiabatic", "dft"), ("adiabatic", "ml")]
    elif source != "dvte":
        out.append(("adiabatic", source))
    if name == DVTE:
        out += [("adiabatic", "dft"), ("adiabatic", "ml")]
    if name == ("twogate_lifetime", "dft"):
        out.append(("twogate_lifetime", "ml"))
    return [r for r in out if r != name]


def bootstrap_table(bs, point, R):
    rows = []
    for name, c in bs.crits.items():
        head = {"criterion": c.key, "source": c.source, "role": c.role}
        for m, label in METRICS.items():
            rows.append({**head, "comparison": "value", "vs_criterion": "", "vs_source": "", "metric": label,
                         "point": point[(m, name)], **interval(R[(m, name)]), "P_gt_0": np.nan})
        for ref in references(name):
            for m, label in METRICS.items():
                v = R[(m, name)] - R[(m, ref)]
                rows.append({**head, "comparison": "difference", "vs_criterion": ref[0], "vs_source": ref[1],
                             "metric": f"d{label}", "point": point[(m, name)] - point[(m, ref)],
                             **interval(v), "P_gt_0": (v > 0).mean()})
    return pd.DataFrame(rows)


def delta_table(point, R, comparisons):
    rows = []
    for a, b in comparisons:
        for m in DELTA_METRICS:
            v = R[(m, a)] - R[(m, b)]
            rows.append({"criterion": a[0], "source": a[1], "vs_criterion": b[0], "vs_source": b[1], "metric": m,
                         "value": point[(m, a)], "vs_value": point[(m, b)], "difference": point[(m, a)] - point[(m, b)],
                         **interval(v), "P_gt_0": (v > 0).mean()})
    return pd.DataFrame(rows)


def ml_table(point, R):
    rows = []
    for key in ("adiabatic", "twogate_lifetime"):
        dft, ml = (key, "dft"), (key, "ml")
        for m in ML_METRICS:
            rows.append({"criterion": key, "metric": m, "dft": point[(m, dft)], "ml": point[(m, ml)],
                         "ml_minus_dft": point[(m, ml)] - point[(m, dft)],
                         **interval(R[(m, ml)] - R[(m, dft)])})
    return pd.DataFrame(rows)


def main():
    df, _, crits = mx.load()
    bs = Bootstrap(df, [c for c in crits if c.role != mx.ROLE_REFERENCE])
    point, R, xpoint, XR = bs.run()
    adiabatic, dvte = ("adiabatic", "dft"), DVTE
    two_gate = (("twogate_lifetime", "dft"), ("twogate_lifetime", "ml"))
    tables = {
        "bootstrap.csv": bootstrap_table(bs, point, R),
        "deltas_vs_adiabatic.csv": delta_table(xpoint, XR, [(dvte, adiabatic), *((t, adiabatic) for t in two_gate),
                                                            (("adiabatic", "ml"), adiabatic)]),
        "deltas_vs_dvte.csv": delta_table(xpoint, XR, [(adiabatic, dvte), (("adiabatic", "ml"), dvte),
                                                       *((t, dvte) for t in two_gate)]),
        "ml_vs_dft.csv": ml_table(xpoint, XR),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(3).to_csv(RESULTS / name, index=False, lineterminator="\n")

    b = tables["bootstrap.csv"].round(3)
    shown = b[(b.metric == "dMCC") & b.vs_criterion.isin(["dvte_2024", "adiabatic"])]
    print(shown[["criterion", "source", "vs_criterion", "vs_source", "point", "lo95", "hi95", "P_gt_0"]]
          .to_string(index=False))
    print(f"\n{REPLICATES} replicates, seed {SEED} -> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
