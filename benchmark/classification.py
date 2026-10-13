# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Scores the criteria that predict whether a photocatalyst-quencher pair quenches.

Reads   ../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/classification/settings.csv   every DFT setting of every criterion
        results/classification/best.csv       the best DFT setting of each criterion and
                                              the same setting with ML parameters

    uv run benchmark/classification.py

Criterion            predicts quenching when                              variants
adiabatic            dG_EnT <= 0                                          dft, ml
dvte                 E_T(PC) >= DvTE(p)                                   p x (normal, KDE quantile)
two_gate             dG_EnT <= 0 and asymmetric Marcus barrier <= t_PC    dft, ml; r from DFT
                                                                          sizes or r = 7 A
marcus_barrier       Marcus barrier < threshold                           dft, ml

The reported DvTE variant is p = 0.008 with the normal quantile; the reported two_gate
variant is r from DFT sizes.

two_gate: t_PC is the barrier at which k * tau0 = K_SV,crit in the exergonic limit of
theory.rate (exp(dG/RT) -> 0), with k_-d/nu = (k_d/K_A)/(k_B T/h) and K_A the encounter
constant at the contact distance r (theory.lifetime_barrier). r is r_PC + r_Q of the pair
from the DFT molecular radii in the dataset (pairs.CONTACT), or theory.CONTACT_DISTANCE
for every pair. t_PC follows from tau0 of the photocatalyst; nothing is chosen on the data.

marcus_barrier: the barrier is symmetric with the geometric-mean lambda. The barrier
threshold is chosen by the highest MCC on the DFT parameters (the middle setting on a
tie); the ML row applies that same setting to the ML quencher parameters.

Photocatalyst properties are DFT throughout.

The ground truth is `quenches`. AUC is computed on each criterion's continuous score:
-dG_EnT, E_T(PC) - DvTE, t_PC - barrier (every endergonic pair below every exergonic
one), -barrier. It is also computed within each photocatalyst that has both classes: the
median over these photocatalysts, and pooled, comparing each quenching pair only with the
non-quenching pairs of its own photocatalyst.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import pairs
import scoring
import theory

RESULTS = pairs.RESULTS / "classification"

BARRIER_THRESHOLDS = tuple(round(t, 1) for t in np.arange(0.1, 20.0001, 0.1))  # kcal/mol
K_SV_CRIT = 10.0                                                              # M-1
# two_gate variant: contact distance in Angstrom, None for r_PC + r_Q of each pair
TWO_GATE_VARIANTS = {"r from DFT sizes": None, "r = 7 A": theory.CONTACT_DISTANCE}
ENDERGONIC_OFFSET = 1e3     # kcal/mol, puts every endergonic pair below every exergonic one

# criterion: (kind, barrier model, variants, sources tuned on, sources given the DFT setting)
CRITERIA = {
    "adiabatic": ("adiabatic", None, ("",), pairs.SOURCES, ()),
    "dvte": ("dvte", None, tuple(f"p = {p}, {kind}" for p in pairs.DVTE_P
                                 for kind in pairs.DVTE_KINDS), ("dft",), ()),
    "two_gate": ("two_gate", "marcus_asymmetric", tuple(TWO_GATE_VARIANTS), pairs.SOURCES, ()),
    "marcus_barrier": ("barrier", "marcus_symmetric", ("",), ("dft",), ("ml",)),
}


@dataclass(frozen=True)
class Setting:
    kind: str                           # adiabatic, dvte, two_gate, barrier
    p: str | None = None                # DvTE quantile
    dvte_kind: str | None = None        # normal or kde
    barrier_model: str = "marcus_symmetric"
    barrier_threshold: float = np.nan   # kcal/mol
    k_sv_crit: float = np.nan           # M-1
    contact_distance: float | None = None   # Angstrom; None: r_PC + r_Q of each pair
    requires_exergonic: bool = False    # also require dG_EnT <= 0


def predict(df, setting: Setting, source):
    """(predicted quenching, continuous score) for every row of df."""
    dG = pairs.driving_force(df, source)
    if setting.kind == "adiabatic":
        return dG <= 0, -dG

    if setting.kind == "dvte":
        gap = df[pairs.PC_E_T].to_numpy(float) - df[pairs.q_dvte(setting.p, setting.dvte_kind)].to_numpy(float)
        return gap >= 0, gap

    forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
    barrier = theory.barrier(setting.barrier_model, dG, forward, reverse)
    if setting.kind == "two_gate":
        r = (pairs.contact_distances(df) if setting.contact_distance is None
             else setting.contact_distance)
        t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), setting.k_sv_crit,
                                       theory.escape_over_frequency(r))
        margin = t_pc - barrier
        exergonic = dG <= 0
        return exergonic & (margin >= 0), np.where(exergonic, margin, margin - ENDERGONIC_OFFSET)

    if setting.kind == "barrier":
        predicted = barrier < setting.barrier_threshold
        if setting.requires_exergonic:
            predicted &= dG <= 0
        return predicted, -barrier

    raise ValueError(f"unknown criterion kind {setting.kind!r}")


def candidates(criterion, variant):
    """Every setting a criterion variant chooses from, in grid order."""
    kind, model = CRITERIA[criterion][:2]
    if kind == "adiabatic":
        return [Setting("adiabatic")]
    if kind == "dvte":
        p, dvte_kind = (part.strip() for part in variant.removeprefix("p = ").split(","))
        return [Setting("dvte", p=p, dvte_kind=dvte_kind)]

    base = Setting(kind, barrier_model=model)
    if kind == "barrier":
        return [replace(base, barrier_threshold=t) for t in BARRIER_THRESHOLDS]
    return [replace(base, k_sv_crit=K_SV_CRIT, contact_distance=TWO_GATE_VARIANTS[variant])]


def n_free(options):
    """Number of setting fields that vary across the candidates, i.e. chosen on the data."""
    fields = [tuple(getattr(s, f) for s in options) for f in Setting.__dataclass_fields__]
    return sum(len(set(map(str, values))) > 1 for values in fields)


def choose(df, options, source="dft"):
    """The candidate with the highest MCC; the middle one on a tie."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    matrix = np.vstack([predict(df, s, source)[0] for s in options])
    return options[scoring.best_setting(y, matrix)]


def evaluate(df, label, variant, source, settings, n_free_parameters):
    """One row per setting: the setting, its metrics and its AUCs."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    photocatalysts = df[pairs.PC_NAME].to_numpy()
    rows = []
    for s in settings:
        predicted, score_ = predict(df, s, source)
        within = scoring.auc_within(y, score_, photocatalysts)
        rows.append({"criterion": label, "parameters": source, "variant": variant,
                     "n_free": n_free_parameters,
                     "barrier_threshold / kcal mol-1": s.barrier_threshold,
                     "K_SV_crit / M-1": s.k_sv_crit,
                     **scoring.metrics(y, predicted), "AUC": scoring.auc(y, score_),
                     "AUC_within_PC_median": within["median"],
                     "AUC_within_PC_pooled": within["pooled"]})
    return rows


def tune(df, label, variant, options, tuned_on=("dft",), given=("ml",), every_setting=True):
    """Chooses a setting per source in tuned_on and applies it to the sources in given.

    Returns (rows for every candidate on DFT, best rows, [(source, chosen setting)]).
    """
    k = n_free(options)
    every, best, chosen = [], [], []
    for source in tuned_on:
        s = choose(df, options, source)
        if source == "dft" and every_setting:
            every += evaluate(df, label, variant, source, options, k)
        best += [{**row, "n_settings": len(options)}
                 for row in evaluate(df, label, variant, source, [s], k)]
        chosen.append((source, s))
        for other in given:
            best += [{**row, "n_settings": 0} for row in evaluate(df, label, variant, other, [s], k)]
            chosen.append((other, s))
    return every, best, chosen


def score(df):
    """(every DFT setting, best setting per criterion and source) as DataFrames."""
    every, best = [], []
    for criterion, (_, _, variants, tuned_on, given) in CRITERIA.items():
        for variant in variants:
            rows, best_rows, _ = tune(df, criterion, variant, candidates(criterion, variant),
                                      tuned_on, given)
            every += rows
            best += best_rows
    return pd.DataFrame(every), pd.DataFrame(best)


def chosen_settings(df):
    """[(criterion, variant, source, setting)] of the benchmark, without scoring every setting."""
    out = []
    for criterion, (_, _, variants, tuned_on, given) in CRITERIA.items():
        for variant in variants:
            for source in tuned_on:
                s = choose(df, candidates(criterion, variant), source)
                out += [(criterion, variant, source, s)] + [(criterion, variant, g, s) for g in given]
    return out


def main():
    table = pairs.load()
    print(pairs.summary(table))
    every, best = score(pairs.scored(table))

    RESULTS.mkdir(parents=True, exist_ok=True)
    scoring.rounded(every).to_csv(RESULTS / "settings.csv", index=False, lineterminator="\n")
    scoring.rounded(best).to_csv(RESULTS / "best.csv", index=False, lineterminator="\n")

    shown = ["criterion", "parameters", "variant", "barrier_threshold / kcal mol-1",
             "accuracy", "MCC", "AUC", "AUC_within_PC_median", "AUC_within_PC_pooled"]
    print()
    print(best[shown].rename(columns={"barrier_threshold / kcal mol-1": "threshold"})
          .round(3).to_string(index=False))
    print(f"\n-> {len(every)} settings, {len(best)} rows in "
          f"{RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
