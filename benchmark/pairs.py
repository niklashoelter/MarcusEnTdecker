# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
The pair table, and the column contract every theory in this folder reads it through.

Reads   ../dataset/pairs.csv
Writes  nothing

WHY THIS MODULE EXISTS
    Every column name in the benchmark is written down once, here. The theory modules
    take arrays, never a file path and never a header string, so a rename in
    `dataset/build_dataset.py` breaks in one place with a message instead of in six
    places with a KeyError. This is the centralization rule the original implementation
    kept in `property_retrieval.py`, minus the part that made it unreproducible: nothing
    is copied in from a sibling directory, and importing this module has no side effect
    beyond reading one CSV.

    It also carries what the column names mean, which for the reorganization energies is
    not optional. See "WHICH SURFACE EACH LAMBDA LIVES ON" below.

THE TWO LABEL REGIMES
    `pairs.csv` carries `quenches` (the naive `k_q > 1e7` cutoff) and `label_class`
    (what the measurement actually established). They disagree for 465 of the 1248
    pairs, because the 1e7 cutoff sits below the Stern-Volmer detection floor
    `25 / tau_0` for 19 of the 26 photocatalysts, covering 912 pairs. For those, "did not
    quench" records that the measurement could not resolve a rate constant, not that the
    rate constant is small: PC10's floor is 3.35e8 M-1 s-1, so it could have a pair going
    at 3e8 and still be counted a non-quencher.

    Only 156 of the 553 apparent non-quenchers are established as non-quenchers. A method
    scored against all 553 has its specificity measured partly against pairs that carry
    no information, and its MCC with it.

    `label_regimes()` returns both readings -- the full 1248 with the naive labels, and
    the 783 pairs the experiment can adjudicate. Which one the paper should report is an
    open scientific question; `../dataset/README.md` sets out four options and takes
    none of them. This module takes none of them either. Both are always computed and
    both are always reported.

WHICH SURFACE EACH LAMBDA LIVES ON
    The quencher and photocatalyst reorganization energies reach pairs.csv under two
    different naming schemes, and they do not line up the way the letters suggest. Both
    are differences on ONE potential energy surface between TWO geometries:

        q_lambda_S_T_dft   = E_singlet(T1 geometry) - E_singlet(S0 geometry)
                             the SINGLET surface, distorted to the triplet geometry
        q_lambda_T_S_dft   = E_triplet(S0 geometry) - E_triplet(T1 geometry)
                             the TRIPLET surface, distorted to the singlet geometry

        pc_lambda_S0_to_T1 = E_triplet(S0 geometry) - E_triplet(T1 geometry)
                             the TRIPLET surface -- the same quantity as q_lambda_T_S
        pc_lambda_T1_to_S0 = E_singlet(T1 geometry) - E_singlet(S0 geometry)
                             the SINGLET surface -- the same quantity as q_lambda_S_T

    So `q_lambda_S_T` pairs with `pc_lambda_T1_to_S0`, and `q_lambda_T_S` pairs with
    `pc_lambda_S0_to_T1`. The letters read in opposite orders. The photocatalyst script's
    own docstring says why it chose the explicit form: "a name that does not say which
    surface it lives on is a name that will be misread." It was, in the code this
    benchmark is ported from -- see marcus.py.

    The four aliases below exist so that no module downstream has to hold both schemes in
    its head. They are named for the surface, which is the thing that matters.
"""

from pathlib import Path

import pandas as pd

PAIRS_CSV = Path(__file__).resolve().parent.parent / "dataset" / "pairs.csv"

# --- identity ---------------------------------------------------------------------
PC_ID = "pc_id"
PC_NAME = "pc_name"
PC_COMPOUND = "pc_compound"
Q_ID = "quencher_id"
Q_NAME = "quencher_name"

# --- measurement and ground truth ---------------------------------------------------
K_Q = "k_q / M-1 s-1"
LOG10_K_Q = "log10_k_q"
QUENCHES = "quenches"
CENSORED = "censored"
K_Q_FLOOR = "k_q_floor / M-1 s-1"
LABEL_CLASS = "label_class"
THRESHOLD_RESOLVABLE = "threshold_resolvable"
TAU0_NS = "tau0 / ns"

# --- quencher, computed --------------------------------------------------------------
Q_E_T = "q_E_T_dft / kcal mol-1"
Q_LAMBDA_S_T = "q_lambda_S_T_dft / kcal mol-1"
Q_LAMBDA_T_S = "q_lambda_T_S_dft / kcal mol-1"
Q_DVTE_MU = "q_dvte_mu_md / kcal mol-1"
Q_DVTE_STD = "q_dvte_std_md / kcal mol-1"

# --- photocatalyst, computed ---------------------------------------------------------
PC_E_T = "pc_E_T_dft / kcal mol-1"
PC_LAMBDA_S0_TO_T1 = "pc_lambda_S0_to_T1_dft / kcal mol-1"
PC_LAMBDA_T1_TO_S0 = "pc_lambda_T1_to_S0_dft / kcal mol-1"

# The same four columns, named for the surface they are a difference on, so that a call
# site cannot pair them up wrongly. This is the only naming the theory modules use.
LAMBDA_SINGLET_SURFACE = {"pc": PC_LAMBDA_T1_TO_S0, "q": Q_LAMBDA_S_T}
LAMBDA_TRIPLET_SURFACE = {"pc": PC_LAMBDA_S0_TO_T1, "q": Q_LAMBDA_T_S}

# --- quencher, experimental redox and its bounds -------------------------------------
Q_E_OX_EXP = "q_E_Ox_vs_SCE_exp / V"
Q_E_OX_BOUND = "q_E_Ox_bound"
Q_E_RED_EXP = "q_E_Red_vs_SCE_exp / V"
Q_E_RED_BOUND = "q_E_Red_bound"

# --- driving forces ------------------------------------------------------------------
# Quencher-centric, negative = exergonic, F = 23.0605 kcal mol-1 V-1:
#     dG_EnT     = E_T(Q) - E_T(PC)
#     dG_SET_ox  = F * (E_Ox(Q)     - E(PC*/PC-))     Q oxidized, PC* is the oxidant
#     dG_SET_red = F * (E(PC+/PC*)  - E_Red(Q))       Q reduced,  PC* is the reductant
#     dG_SET     = whichever channel is more accessible
DG_ENT = "dG_EnT / kcal mol-1"

# The SET driving force exists twice over, from the measured quencher potentials and from
# the computed ones. The dataset keeps them apart deliberately -- a measured potential and
# a DFT potential are not interchangeable, and which one an analysis used is part of what
# it reports -- so this module keeps them apart too and every output row says which.
SET_SOURCES = ("exp", "calc")


def set_columns(source: str) -> dict:
    """The SET driving-force column names for one source."""
    if source not in SET_SOURCES:
        raise ValueError(f"unknown SET source {source!r}; expected one of {SET_SOURCES}")
    cols = {
        "ox": f"dG_SET_ox_{source} / kcal mol-1",
        "red": f"dG_SET_red_{source} / kcal mol-1",
        "combined": f"dG_SET_{source} / kcal mol-1",
        "channel": f"dG_SET_{source}_channel",
    }
    if source == "exp":
        cols["bound"] = "dG_SET_exp_bound"
    return cols


# A pair counts as quenching above this rate constant. Set in `dataset/build_dataset.py`;
# repeated here for reporting only, never to relabel anything.
K_Q_THRESHOLD = 1.0e7

# The Faraday constant in the units the driving forces use, for anyone converting a
# potential difference by hand. The value `dataset/README.md` states.
F_KCAL_PER_MOL_V = 23.0605

# Photocatalysts whose DFT input is known to be suspect, and which are still in the
# table. Nothing is dropped on account of this list: it exists so a run says out loud
# what is in the numbers. `run_benchmark.py --exclude-pc` drops one for a sensitivity
# check, and reports that it did.
#
# PC24 is not here. It was excluded upstream, by name, in `dataset/build_dataset.py`, so
# pairs.csv is 26 photocatalysts and this benchmark never sees it. That is a scientific
# decision recorded in dataset/README.md, not something this folder can revisit.
FLAGGED_PHOTOCATALYSTS = {
    "PC19": "imaginary frequencies in both the S0 and the reduced optimization. It has "
            "a triplet energy but no singlet-surface reorganization energy, so its 48 "
            "pairs drop out of every barrier and rate method. They stay in the "
            "adiabatic and DvTE numbers, which need only E_T.",
    "PC20": "lambda on the triplet surface is 33.67 kcal/mol against 12.70 on the "
            "singlet surface, a 2.7x asymmetry no other non-pmb complex shows. Never "
            "verified. It is the pair of numbers the asymmetric Marcus form is most "
            "sensitive to.",
}


class MissingColumns(KeyError):
    """A required column is not in pairs.csv.

    Raised instead of scoring on whatever subset happens to be available. The message
    names the columns, who wanted them, and where their definition lives.
    """

    def __str__(self):  # KeyError would otherwise repr() the message and eat newlines
        return self.args[0]


def load_pairs(path: Path = PAIRS_CSV) -> pd.DataFrame:
    """Read the pair table. No filtering, no imputation, no derived columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:  uv run dataset/build_dataset.py"
        )
    return pd.read_csv(path)


def require(df: pd.DataFrame, columns, wanted_by: str) -> None:
    """Fail loudly if any of `columns` is absent.

    The alternatives -- dropping the theory, or scoring it on the rows that happen to be
    complete -- both produce a table that looks finished and is not. A benchmark is a
    claim about a fixed set of pairs; silently changing that set per method makes the
    columns of its output table incomparable, which is the one thing it exists to do.
    """
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return
    raise MissingColumns("\n".join([
        f"{wanted_by} needs {len(missing)} column(s) that pairs.csv does not have:",
        *(f"    {c!r}" for c in missing),
        "",
        "Every column name this benchmark expects is defined at the top of",
        "benchmark/pairs.py, with the sign and surface conventions each one follows.",
        "If the dataset renamed something, change it there and nowhere else.",
        "See dataset/README.md for what the table does and does not yet state.",
    ]))


def has(df: pd.DataFrame, columns) -> bool:
    """True if every column is present. For deciding what to offer, never what to score."""
    return all(c in df.columns for c in columns)


def dG_set(df: pd.DataFrame, source: str = "exp"):
    """The more accessible single-electron-transfer channel, and its bound flag.

    Returns (values, is_lower_bound). `is_lower_bound` is all-False for the computed
    source, which has no bounds -- a DFT potential is always a value.
    """
    cols = set_columns(source)
    require(df, [cols["combined"]], f"the {source} single-electron-transfer driving force")
    values = df[cols["combined"]].astype(float)
    if "bound" in cols and cols["bound"] in df.columns:
        is_bound = df[cols["bound"]].astype(str).str.lower() == "lower"
    else:
        is_bound = pd.Series(False, index=df.index)
    return values, is_bound


def label_regimes(df: pd.DataFrame) -> dict:
    """The two ways of reading the ground truth, both of them, always.

    Returns {name: (subset, y)}:

      'full'         all pairs, y = `quenches` -- the naive reading, in which every pair
                     that did not quench is a negative. 553 are negatives on this
                     reading; only 156 were established as such.

      'resolvable'   the 783 pairs with label_class != indeterminate: above the cutoff
                     and actually detected, or below the cutoff and detectable by that
                     photocatalyst. Drops the 465 pairs the measurement cannot
                     adjudicate, and with them most of the short-lived photocatalysts.

    Nothing here chooses between them. The difference between the two is a headline the
    benchmark has to show, not a detail to settle before reporting.
    """
    require(df, [QUENCHES, LABEL_CLASS], "the ground truth")
    full_y = df[QUENCHES].astype(int).astype(bool)

    sub = df[df[LABEL_CLASS] != "indeterminate"]
    sub_y = sub[LABEL_CLASS] == "positive"
    # The two definitions have to agree wherever both apply. If they ever stop agreeing,
    # one has been redefined upstream and the benchmark cannot tell which is now truth.
    if not (sub_y.to_numpy() == sub[QUENCHES].astype(int).astype(bool).to_numpy()).all():
        raise ValueError(
            "label_class and quenches disagree on a resolvable pair. One of the two has "
            "been redefined in dataset/build_dataset.py; the benchmark cannot tell which "
            "is now the ground truth."
        )
    return {"full": (df, full_y), "resolvable": (sub, sub_y)}


def describe(df: pd.DataFrame) -> str:
    """One block of text saying what was loaded and what is known to be odd about it."""
    counts = df[LABEL_CLASS].value_counts()
    flagged = sorted(set(df[PC_NAME]) & set(FLAGGED_PHOTOCATALYSTS))
    lines = [
        f"{len(df)} pairs  ({df[PC_NAME].nunique()} photocatalysts x "
        f"{df[Q_NAME].nunique()} quenchers)",
        f"  quenches = 1        {int(df[QUENCHES].sum()):5d}   at the "
        f"{K_Q_THRESHOLD:g} M-1 s-1 cutoff",
        f"  censored            {int(df[CENSORED].sum()):5d}   below the Stern-Volmer "
        f"detection floor; k_q is an upper bound for these",
        f"  label_class               positive {counts.get('positive', 0)}, "
        f"negative {counts.get('negative', 0)}, "
        f"indeterminate {counts.get('indeterminate', 0)}",
    ]
    if "dG_SET_exp_bound" in df.columns:
        n_bound = int((df["dG_SET_exp_bound"].astype(str).str.lower() == "lower").sum())
        lines.append(f"  dG_SET_exp is a lower bound for {n_bound} pairs -- the true "
                     f"value can only be more endergonic")
    if flagged:
        lines.append("  photocatalysts flagged as suspect, present and NOT dropped: "
                     + ", ".join(flagged))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(load_pairs()))
