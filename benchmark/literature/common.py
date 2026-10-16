"""
Shared pieces for the literature comparisons.

Not a script; imported by predict.py and the comparison scripts.

lambda_S0_T1: S0 surface at the T1 geometry; lambda_T1_S0: T1 surface at the S0 geometry
(per molecule; the tables in data/literature_benchmark/ use this convention). Per pair:

    lambda_forward = pc.lambda_T1_S0 + substrate.lambda_S0_T1
    lambda_reverse = pc.lambda_S0_T1 + substrate.lambda_T1_S0

Reorganisation energies enter unscaled. Ma: `barrier_symmetric(dG, lambda_forward)`; every
other case: `barrier_asymmetric`.

`estimate`: mixture mean ("mean") or mean of the highest-weight component (`max_mode`).
`model`: "deployed" (ml/data/model_final.pt) or one of the five random-split members of
ml.ensemble.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "literature_benchmark"
OUT = REPO / "benchmark" / "results" / "literature"

RT = 1.98720425e-3 * 298.15
ESTIMATES = ("mean", "mode")
PROPERTIES = ("E_T", "lambda_S0_T1", "lambda_T1_S0")


def column(prop, estimate="mean", uncertainty=False):
    """The predictions-file column of one property."""
    if uncertainty:
        return f"{prop}_ml_uncertainty / kcal mol-1"
    return f"{prop}_ml{'_mode' if estimate == 'mode' else ''} / kcal mol-1"


def literature(case):
    """The published table for one case, in our lambda convention."""
    return pd.read_csv(DATA / f"{case}.csv")


def predictions(case):
    """Deployed-model predictions for one case, indexed by compound_id."""
    return pd.read_csv(OUT / f"predictions_{case}.csv").set_index("compound_id")


def member_predictions(case):
    """Predictions of the deployed model and every ensemble member, one row per model and compound."""
    return pd.read_csv(OUT / f"predictions_members_{case}.csv")


def models(case):
    """Each model's predictions for one case, as {model: frame indexed by compound_id}."""
    m = member_predictions(case)
    return {name: g.set_index("compound_id") for name, g in m.groupby("model", sort=False)}


def ours(row, estimate="mean"):
    """One compound's predicted properties, as a plain mapping."""
    out = {prop: row[column(prop, estimate)] for prop in PROPERTIES}
    out.update({f"{prop}_sd": row[column(prop, uncertainty=True)] for prop in PROPERTIES})
    return out


def barrier_symmetric(dG, lambda_total):
    """Marcus barrier from a single reorganisation energy."""
    lambda_total = np.asarray(lambda_total, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):   # masked branch is evaluated too
        return np.where(lambda_total > 0, (lambda_total + dG) ** 2 / (4 * lambda_total), np.nan)


def barrier_asymmetric(dG, lambda_forward, lambda_reverse):
    """Marcus barrier from two reorganisation energies (Sole-Daura's equation 4).

    Symmetric form on the geometric mean where the parabolas do not intersect.
    """
    root = lambda_forward * lambda_reverse + (lambda_forward - lambda_reverse) * dG
    if root < 0 or abs(lambda_forward - lambda_reverse) < 1e-10:
        return float(barrier_symmetric(dG, np.sqrt(max(lambda_forward * lambda_reverse, 0.0))))
    return float(lambda_forward * ((np.sqrt(root) - lambda_reverse)
                                   / (lambda_forward - lambda_reverse)) ** 2)


def pair_lambdas(pc, substrate):
    """(forward, reverse) reorganisation energy for a pair, from two property mappings."""
    return (pc["lambda_T1_S0"] + substrate["lambda_S0_T1"],
            pc["lambda_S0_T1"] + substrate["lambda_T1_S0"])


def pair_barrier(pc, substrate):
    """Driving force and asymmetric barrier of one photocatalyst-substrate pair."""
    forward, reverse = pair_lambdas(pc, substrate)
    dG = substrate["E_T"] - pc["E_T"]
    return dG, barrier_asymmetric(dG, forward, reverse)


def split_difference(barrier, a, b):
    """barrier(a) - barrier(b), split into a driving-force and a reorganisation part.

    `a` and `b` are (dG, lambda_forward, lambda_reverse); two-factor Shapley split.
    """
    full = barrier(*a) - barrier(*b)
    dg = 0.5 * ((barrier(a[0], *a[1:]) - barrier(b[0], *a[1:]))
                + (barrier(a[0], *b[1:]) - barrier(b[0], *b[1:])))
    return full, dg, full - dg


def spearman(x, y):
    """Spearman rho, p and n over the rows where both are present."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 4:
        return np.nan, np.nan, len(d)
    r = stats.spearmanr(d.x, d.y)
    return float(r.statistic), float(r.pvalue), len(d)


def bootstrap_spearman(x, y, draws=10000, seed=20261006):
    """Percentile 95 % interval of Spearman rho, resampling compounds."""
    d = pd.DataFrame({"x": x, "y": y}).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(draws):
        s = d[rng.integers(0, len(d), len(d))]
        if np.ptp(s[:, 0]) > 0 and np.ptp(s[:, 1]) > 0:
            rhos.append(stats.spearmanr(s[:, 0], s[:, 1]).statistic)
    return tuple(np.percentile(rhos, [2.5, 97.5]))


def auc(score, positive):
    """Probability that a positive scores higher than a negative (ties count half)."""
    d = pd.DataFrame({"s": score, "p": positive}).dropna()
    pos, neg = d.s[d.p.astype(bool)].to_numpy(), d.s[~d.p.astype(bool)].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


# Energies to two decimals; correlations, AUCs, probabilities, rates, fractions and counts
# at full precision.
_ENERGY = re.compile(r"(^|_)(E_T|lambda|barrier|dG|ddG|margin|gap|dE|offset|mae|error|part)", re.I)
_NOT_ENERGY = re.compile(r"(auc|rho|spearman|pearson|p_value|p_sign|correct|fraction|percent|k_q|ratio|^n_|^n$)", re.I)


def write(frame, name):
    """One result table, energies rounded at write."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    for c in frame.columns:
        if (_ENERGY.search(str(c)) and not _NOT_ENERGY.search(str(c))
                and pd.api.types.is_float_dtype(frame[c])):
            frame[c] = frame[c].round(2)
    frame.to_csv(OUT / name, index=False, lineterminator="\n")
    print(f"{name}: {len(frame)} rows")
