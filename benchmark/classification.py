# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""
Scores the criteria that predict whether a photocatalyst-quencher pair quenches.

Reads   ../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/classification/settings.csv   every DFT setting of every criterion
        results/classification/best.csv       the best DFT setting of each criterion and
                                              the same setting with ML parameters

    uv run benchmark/classification.py

Criterion            predicts quenching when                     variants
adiabatic            dG_EnT <= 0                                 dft, ml
dvte                 E_T(PC) >= DvTE(p)                          p x (normal, KDE quantile)
marcus_barrier       Marcus barrier < threshold                  lambda raw, lambda scaled
marcus_rate          k * tau0 >= K_SV,crit, Marcus barrier       lambda raw, lambda scaled
rehm_weller_rate     k * tau0 >= K_SV,crit, Rehm-Weller barrier  lambda raw, lambda scaled

The barrier is symmetric with the geometric-mean lambda; the rate is the full
encounter-complex scheme of theory.rate. "lambda raw" uses the DFT lambda as is
(alpha = 1), "lambda scaled" multiplies the pair lambdas by alpha. The barrier threshold
and alpha are chosen by the highest MCC on the DFT parameters (the middle setting on a
tie); the ML row applies that same setting to the ML quencher parameters. Photocatalyst
properties are DFT throughout.

The ground truth is `quenches`. AUC is computed on each criterion's continuous score:
-dG_EnT, E_T(PC) - DvTE, -barrier, log10(k * tau0).
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import pairs
import scoring
import theory

RESULTS = pairs.RESULTS / "classification"

ALPHAS = tuple(round(a, 2) for a in np.arange(0.05, 1.0001, 0.05))
BARRIER_THRESHOLDS = tuple(round(t, 1) for t in np.arange(0.1, 20.0001, 0.1))  # kcal/mol
K_SV_CRIT = 10.0                                                              # M-1

LAMBDA_VARIANTS = ("lambda raw", "lambda scaled")

# criterion: (kind, barrier model, variants, sources tuned on, sources given the DFT setting)
CRITERIA = {
    "adiabatic": ("adiabatic", None, ("",), pairs.SOURCES, ()),
    "dvte": ("dvte", None, tuple(f"p = {p}, {kind}" for p in pairs.DVTE_P
                                 for kind in pairs.DVTE_KINDS), ("dft",), ()),
    "marcus_barrier": ("barrier", "marcus_symmetric", LAMBDA_VARIANTS, ("dft",), ("ml",)),
    "marcus_rate": ("rate", "marcus_symmetric", LAMBDA_VARIANTS, ("dft",), ("ml",)),
    "rehm_weller_rate": ("rate", "rehm_weller", LAMBDA_VARIANTS, ("dft",), ("ml",)),
}


@dataclass(frozen=True)
class Setting:
    kind: str                           # adiabatic, dvte, barrier, rate
    p: str | None = None                # DvTE quantile
    dvte_kind: str | None = None        # normal or kde
    barrier_model: str = "marcus_symmetric"
    alpha: float = 1.0
    barrier_threshold: float = np.nan   # kcal/mol
    k_sv_crit: float = np.nan           # M-1
    requires_exergonic: bool = False    # also require dG_EnT <= 0
    rate_law: str = "full"              # full or irreversible


def predict(df, setting: Setting, source):
    """(predicted quenching, continuous score) for every row of df."""
    dG = pairs.driving_force(df, source)
    if setting.kind == "adiabatic":
        return dG <= 0, -dG

    if setting.kind == "dvte":
        gap = df[pairs.PC_E_T].to_numpy(float) - df[pairs.q_dvte(setting.p, setting.dvte_kind)].to_numpy(float)
        return gap >= 0, gap

    forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, source))
    barrier = theory.barrier(setting.barrier_model, dG,
                             setting.alpha * forward, setting.alpha * reverse)
    if setting.kind == "barrier":
        predicted = barrier < setting.barrier_threshold
        if setting.requires_exergonic:
            predicted &= dG <= 0
        return predicted, -barrier

    if setting.kind == "rate":
        k = theory.rate(dG, barrier) if setting.rate_law == "full" else theory.rate_irreversible(barrier)
        log_k_tau0 = np.log10(k * pairs.tau0_seconds(df))
        return log_k_tau0 >= np.log10(setting.k_sv_crit), log_k_tau0

    raise ValueError(f"unknown criterion kind {setting.kind!r}")


def candidates(criterion, variant):
    """Every setting a criterion variant chooses from, in grid order."""
    kind, model = CRITERIA[criterion][:2]
    if kind == "adiabatic":
        return [Setting("adiabatic")]
    if kind == "dvte":
        p, dvte_kind = (part.strip() for part in variant.removeprefix("p = ").split(","))
        return [Setting("dvte", p=p, dvte_kind=dvte_kind)]

    alphas = (1.0,) if variant == "lambda raw" else ALPHAS
    base = Setting(kind, barrier_model=model)
    if kind == "barrier":
        return [replace(base, alpha=a, barrier_threshold=t) for a in alphas for t in BARRIER_THRESHOLDS]
    return [replace(base, alpha=a, k_sv_crit=K_SV_CRIT) for a in alphas]


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
    """One row per setting: the setting, its metrics and its AUC."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for s in settings:
        predicted, score_ = predict(df, s, source)
        rows.append({"criterion": label, "parameters": source, "variant": variant,
                     "n_free": n_free_parameters,
                     "alpha": s.alpha if s.kind in ("barrier", "rate") else np.nan,
                     "barrier_threshold / kcal mol-1": s.barrier_threshold,
                     "K_SV_crit / M-1": s.k_sv_crit,
                     **scoring.metrics(y, predicted), "AUC": scoring.auc(y, score_)})
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

    shown = ["criterion", "parameters", "variant", "alpha", "barrier_threshold / kcal mol-1",
             "recall", "specificity", "MCC", "AUC"]
    print()
    print(best[shown].rename(columns={"barrier_threshold / kcal mol-1": "threshold"})
          .round(3).to_string(index=False))
    print(f"\n-> {len(every)} settings, {len(best)} rows in "
          f"{RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
