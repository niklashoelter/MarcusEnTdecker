# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Scores the criteria that predict whether a photocatalyst-quencher pair quenches.

Reads   ../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/classification/settings.csv   each fixed DFT variant of every criterion
        results/classification/best.csv       fixed variants per criterion and parameter source

    uv run benchmark/classification.py

Criterion            predicts quenching when                              variants
adiabatic            dG_EnT <= 0                                          dft, ml
dvte                 E_T(PC) >= DvTE(p)                                   p x (normal, KDE quantile)
two_gate             dG_EnT <= 0 and asymmetric Marcus barrier <= t_PC    dft, ml; r = r_PC + r_Q
                                                                          or r = 7 A

t_PC: barrier at which k tau0 = K_SV,crit for exp(dG/RT) -> 0 (theory.lifetime_barrier),
k_-d/nu = (k_d/K_A)/(k_B T/h), K_A at contact distance r. two_gate
outside the variant comparisons (TWO_GATE_DEFAULT): r = r_PC + r_Q with DFT, r = 7 A with ML.
Photocatalyst properties are DFT throughout.

Ground truth: `quenches`. AUC on the scores -dG_EnT, E_T(PC) - DvTE, t_PC - barrier
(endergonic pairs last); also within each photocatalyst with both classes (median and
pooled).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import pairs
import scoring
import theory

RESULTS = pairs.RESULTS / "classification"

K_SV_CRIT = 10.0            # M-1
# two_gate variant: contact distance in Angstrom, None for r_PC + r_Q of each pair
TWO_GATE_VARIANTS = {"r from DFT sizes": None, "r = 7 A": theory.CONTACT_DISTANCE}
# two_gate variant of each source where the variants are not compared
TWO_GATE_DEFAULT = {"dft": "r from DFT sizes", "ml": "r = 7 A"}
ENDERGONIC_OFFSET = 1e3     # kcal/mol, ranks endergonic pairs last

# criterion: (kind, barrier model, variants, parameter sources)
CRITERIA = {
    "adiabatic": ("adiabatic", None, ("",), pairs.SOURCES),
    "dvte": ("dvte", None, tuple(f"p = {p}, {kind}" for p in pairs.DVTE_P
                                 for kind in pairs.DVTE_KINDS), ("dft",)),
    "two_gate": ("two_gate", "marcus_asymmetric", tuple(TWO_GATE_VARIANTS), pairs.SOURCES),
}


@dataclass(frozen=True)
class Setting:
    kind: str                           # adiabatic, dvte, two_gate
    p: str | None = None                # DvTE quantile
    dvte_kind: str | None = None        # normal or kde
    barrier_model: str = "marcus_asymmetric"
    k_sv_crit: float = np.nan           # M-1
    contact_distance: float | None = None   # Angstrom; None: r_PC + r_Q of each pair


def predict(df, setting: Setting, source):
    """(predicted quenching, continuous score) for every row of df."""
    dG = pairs.driving_force(df, source)
    if setting.kind == "adiabatic":
        return dG <= 0, -dG

    if setting.kind == "dvte":
        gap = df[pairs.PC_E_T].to_numpy(float) - df[pairs.q_dvte(setting.p, setting.dvte_kind)].to_numpy(float)
        return gap >= 0, gap

    if setting.kind == "two_gate":
        forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
        barrier = theory.barrier(setting.barrier_model, dG, forward, reverse)
        r = (pairs.contact_distances(df) if setting.contact_distance is None
             else setting.contact_distance)
        t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), setting.k_sv_crit,
                                       theory.escape_over_frequency(r))
        margin = t_pc - barrier
        exergonic = dG <= 0
        return exergonic & (margin >= 0), np.where(exergonic, margin, margin - ENDERGONIC_OFFSET)

    raise ValueError(f"unknown criterion kind {setting.kind!r}")


def candidates(criterion, variant):
    """The fixed setting of a criterion variant."""
    kind, model = CRITERIA[criterion][:2]
    if kind == "adiabatic":
        return [Setting("adiabatic")]
    if kind == "dvte":
        p, dvte_kind = (part.strip() for part in variant.removeprefix("p = ").split(","))
        return [Setting("dvte", p=p, dvte_kind=dvte_kind)]

    return [Setting(kind, barrier_model=model, k_sv_crit=K_SV_CRIT,
                    contact_distance=TWO_GATE_VARIANTS[variant])]


def two_gate_setting(source):
    """The two_gate setting of TWO_GATE_DEFAULT[source]."""
    return candidates("two_gate", TWO_GATE_DEFAULT[source])[0]


def contact_distance(df, source):
    """Contact distance in Angstrom of the TWO_GATE_DEFAULT variant, one per row of df."""
    r = TWO_GATE_VARIANTS[TWO_GATE_DEFAULT[source]]
    return pairs.contact_distances(df) if r is None else np.full(len(df), float(r))


def evaluate(df, label, variant, source, settings):
    """One row per setting: the setting, its metrics and its AUCs."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    photocatalysts = df[pairs.PC_NAME].to_numpy()
    rows = []
    for s in settings:
        predicted, score_ = predict(df, s, source)
        within = scoring.auc_within(y, score_, photocatalysts)
        rows.append({"criterion": label, "parameters": source, "variant": variant,
                     "n_free": 0,
                     "K_SV_crit / M-1": s.k_sv_crit,
                     **scoring.metrics(y, predicted), "AUC": scoring.auc(y, score_),
                     "AUC_within_PC_median": within["median"],
                     "AUC_within_PC_pooled": within["pooled"]})
    return rows


def score(df):
    """(all DFT variants, fixed variants per criterion and source) as DataFrames."""
    every, best = [], []
    for criterion, (_, _, variants, sources) in CRITERIA.items():
        for variant in variants:
            settings = candidates(criterion, variant)
            for source in sources:
                rows = evaluate(df, criterion, variant, source, settings)
                if source == "dft":
                    every.extend(rows)
                best.extend({**row, "n_settings": len(settings)} for row in rows)
    return pd.DataFrame(every), pd.DataFrame(best)


def main():
    table = pairs.load()
    print(pairs.summary(table))
    every, best = score(pairs.scored(table))

    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(every).to_csv(RESULTS / "settings.csv", index=False, lineterminator="\n")
    scoring.rounded(best).to_csv(RESULTS / "best.csv", index=False, lineterminator="\n")

    shown = ["criterion", "parameters", "variant", "accuracy", "MCC", "AUC",
             "AUC_within_PC_median", "AUC_within_PC_pooled"]
    print()
    print(best[shown].round(3).to_string(index=False))
    print(f"\n-> {len(every)} settings, {len(best)} rows in "
          f"{RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
