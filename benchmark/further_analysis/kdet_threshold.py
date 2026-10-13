# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
The two-gate criterion with the measured detection limit of each photocatalyst in place
of the global K_SV,crit = 10 M-1.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
        results/labels/noise/detection_limits_pc.csv        K_det per photocatalyst
        results/labels/noise/pair_labels_probabilistic.csv  P(K > K_det) and label class
                                                            per scored pair
        (both from labels/s2_posteriors.py and labels/s4_downstream.py)
Writes  results/kdet_threshold/metrics.csv      one row per label set, criterion and
                                                source: metrics and AUCs
        results/kdet_threshold/t_pc_per_pc.csv  per K_SV,crit variant and photocatalyst:
                                                median, min and max t_PC over its scored
                                                pairs
        results/kdet_threshold/t_pc_range.csv   per variant: min, median and max of the
                                                per-photocatalyst medians

    uv run benchmark/further_analysis/kdet_threshold.py

K_SV,crit variants in t_PC (theory.lifetime_barrier, Eyring k_-d/nu at the contact
distance of the pair): kdet, K_det of the photocatalyst; floor200, 200 M-1 for PC25-27
(0.2 mM) and 10 M-1 otherwise; global10, 10 M-1 (the benchmark). Where k_d tau0 / K_SV,crit
<= 1, t_PC is -inf and no pair of that photocatalyst passes. Label sets: the labels of the
data set; P(K > K_det) >= 0.5; the confident pairs only (P <= 0.05 or >= 0.95), labelled by
the posterior. Reference criteria: adiabatic (DFT, ML) and DvTE p = 0.008 normal quantile.
Metrics written with three decimals, t_PC in kcal/mol with two.
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

NOISE = HERE / "results" / "labels" / "noise"
RESULTS = HERE / "results" / "kdet_threshold"
ZERO_SV_PCS = ("PC25", "PC26", "PC27")      # screened up to 0.2 mM
FLOOR_02MM = 200.0                          # M-1


def two_gate(df, k_sv_crit, source):
    """(prediction, score, t_PC) of the two-gate criterion with a per-pair K_SV,crit."""
    dG = pairs.driving_force(df, source)
    barrier = theory.marcus_barrier_asymmetric(dG, *theory.pair_lambdas(*pairs.lambdas(df, source)))
    with np.errstate(invalid="ignore"):
        t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), k_sv_crit,
                                       theory.escape_over_frequency(pairs.contact_distances(df)))
    t_pc = np.where(np.isfinite(t_pc), t_pc, -np.inf)
    margin = t_pc - barrier
    exergonic = dG <= 0
    score = np.where(exergonic, margin, margin - classification.ENDERGONIC_OFFSET)
    score = np.where(np.isfinite(score), score, -1e6)
    return exergonic & (margin >= 0), score, t_pc


def load():
    df = pairs.scored(pairs.load())
    lab = pd.read_csv(NOISE / "pair_labels_probabilistic.csv")
    det = pd.read_csv(NOISE / "detection_limits_pc.csv")
    df = df.merge(lab[[pairs.PC_NAME, pairs.Q_NAME, pairs.QUENCHES, "P_K>Kdet [free]", "label_class_Kdet"]]
                  .rename(columns={pairs.QUENCHES: "q_check"}),
                  on=[pairs.PC_NAME, pairs.Q_NAME], how="left", validate="one_to_one")
    assert df["q_check"].notna().all() and (df["q_check"] == df[pairs.QUENCHES]).all()
    kdet = det.set_index("pc")["K_det_3sd / M-1"]
    assert set(df[pairs.PC_NAME]) <= set(kdet.index)
    return df, kdet


def variants(df, kdet):
    return {"kdet": df[pairs.PC_NAME].map(kdet).to_numpy(float),
            "floor200": np.where(df[pairs.PC_NAME].isin(ZERO_SV_PCS), FLOOR_02MM, classification.K_SV_CRIT),
            "global10": np.full(len(df), classification.K_SV_CRIT)}


def main():
    df, kdet = load()
    K = variants(df, kdet)
    reference = classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                       k_sv_crit=classification.K_SV_CRIT, contact_distance=None)
    for source in pairs.SOURCES:
        assert (two_gate(df, K["global10"], source)[0] == classification.predict(df, reference, source)[0]).all()

    labels = {
        "current": (np.ones(len(df), bool), df[pairs.QUENCHES].to_numpy(bool)),
        "prob P>=0.5": (np.ones(len(df), bool), (df["P_K>Kdet [free]"] >= 0.5).to_numpy()),
        "confident": (df["label_class_Kdet"].isin(["confident quench", "confident no quench"]).to_numpy(),
                      (df["label_class_Kdet"] == "confident quench").to_numpy()),
    }
    preds, t_pcs = [], {}
    for v, ksv in K.items():
        for source in pairs.SOURCES:
            p, s, t = two_gate(df, ksv, source)
            preds.append((f"two-gate {v}", source, p, s))
            t_pcs[v] = t
    for source in pairs.SOURCES:
        p, s = classification.predict(df, classification.Setting("adiabatic"), source)
        preds.append(("adiabatic", source, p, s))
    p, s = classification.predict(df, classification.Setting("dvte", p="0.008", dvte_kind="normal"), "dft")
    preds.append(("DvTE q_dvte_0.008", "-", p, s))

    groups = df[pairs.PC_NAME].to_numpy()
    rows = []
    for name, (mask, y) in labels.items():
        for criterion, source, p, s in preds:
            m = scoring.metrics(y[mask], p[mask])
            rows.append({"label_set": name, "criterion": criterion, "parameters": source,
                         "n": m["n"], "n_pos": int(y[mask].sum()), "accuracy": m["accuracy"], "MCC": m["MCC"],
                         "AUC": scoring.auc(y[mask], s[mask]),
                         "AUC_within_PC_pooled": scoring.auc_within(y[mask], s[mask], groups[mask])["pooled"],
                         "specificity": m["specificity"], "recall": m["recall"],
                         "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"]})
    metrics = pd.DataFrame(rows)

    per_pc = []
    for v, ksv in K.items():
        t = pd.DataFrame({"pc": groups, "tau0": df[pairs.TAU0].to_numpy(float), "K": ksv, "t": t_pcs[v]})
        for pc, g in t.groupby("pc", sort=False):
            per_pc.append({"variant": v, "pc": pc, "tau0 / ns": g.tau0.iloc[0], "K_SV_crit / M-1": g.K.iloc[0],
                           "t_PC / kcal mol-1": g.t.median(), "t_PC min over pairs / kcal mol-1": g.t.min(),
                           "t_PC max over pairs / kcal mol-1": g.t.max()})
    per_pc = pd.DataFrame(per_pc)
    rng = per_pc.groupby("variant", sort=False)["t_PC / kcal mol-1"].agg(["min", "median", "max"])
    rng.columns = [f"t_PC {c} / kcal mol-1" for c in rng.columns]

    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(metrics).to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n")
    per_pc.round({"tau0 / ns": 2, "K_SV_crit / M-1": 1, "t_PC / kcal mol-1": 2,
                  "t_PC min over pairs / kcal mol-1": 2, "t_PC max over pairs / kcal mol-1": 2}).to_csv(
        RESULTS / "t_pc_per_pc.csv", index=False, lineterminator="\n")
    rng.round(2).to_csv(RESULTS / "t_pc_range.csv", lineterminator="\n")

    shown = ["criterion", "parameters", "MCC", "recall", "AUC"]
    print(metrics[metrics.label_set == "current"][shown].round(3).to_string(index=False))
    print(rng.round(2).to_string())
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
