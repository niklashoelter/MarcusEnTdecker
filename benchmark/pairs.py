# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
The pair table, and the column contract every theory in this folder reads it through.

Reads   ../data/exp_dataset.csv
Writes  nothing

WHY THIS MODULE EXISTS
    Every column name in the benchmark is written down once, here. The theory modules
    take arrays, never a file path and never a header string, so a rename in
    `data/build_exp_dataset.py` breaks in one place with a message instead of in six
    places with a KeyError. This is the centralization rule the original implementation
    kept in `property_retrieval.py`, minus the part that made it unreproducible: nothing
    is copied in from a sibling directory, and importing this module has no side effect
    beyond reading one CSV.

    It also carries what the column names mean, which for the reorganization energies is
    not optional. See "WHICH SURFACE EACH LAMBDA LIVES ON" below.

THE TWO LABEL REGIMES
    The ground truth is set on K_SV, the observable, and not on k_q. A cutoff on k_q was
    incoherent: k_q = K_SV / tau_0, so the Stern-Volmer detection floor moves with the
    photocatalyst, and at a 1e7 cutoff that floor sat ABOVE the cutoff for 19 of the 26
    photocatalysts. 465 pairs could not be labelled at all. On K_SV the floor is one
    number for every photocatalyst and every pair is labelled by the same criterion, so
    the indeterminate class is gone and with it the 'resolvable' regime.

    What remains is a choice of how permissive the criterion is, and `exp_dataset.csv` carries
    both readings:

      `quenches`         K_SV > 10, or K_SV <= 10 with a slope significant across the
                         points the fit used. 790 of 1248.
      `quenches_strict`  the published criterion, K_SV >= 25 and r2 >= 0.90. 634 of 1248.

    The 156 pairs between them are weak quenchers: real monotone 3-4 % signal, well above
    read noise, at rate constants the strict criterion never counted. Whether the paper
    claims them is a scientific question about what "quenching" means, not a data
    question, so `label_regimes()` returns both and this module takes no position.

WHICH SURFACE EACH LAMBDA LIVES ON
    The quencher and photocatalyst reorganization energies reach exp_dataset.csv under two
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

PAIRS_CSV = Path(__file__).resolve().parent.parent / "data" / "exp_dataset.csv"

# --- identity ---------------------------------------------------------------------
# Photocatalysts are named by their new id, PC1-PC27; `pc_old_id` is the number the raw
# files use. See data/photocatalyst_ids.py.
PC_ID = "pc_new_id"
PC_OLD_ID = "pc_old_id"
PC_NAME = "pc_name"
Q_ID = "quencher_id"
Q_NAME = "quencher_name"

# --- measurement and ground truth ---------------------------------------------------
K_SV = "K_SV / M-1"
TAU0_NS = "tau0 / ns"
K_Q = "k_q / M-1 s-1"
K_Q_FLOOR = "k_q_floor / M-1 s-1"
QUENCHES = "quenches"
QUENCHES_STRICT = "quenches_strict"
CENSORED = "censored"

# --- quencher, computed --------------------------------------------------------------
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
    """The SET driving-force column names for one source: the more accessible channel,
    and for the measured potentials the flag saying it is a lower bound."""
    if source not in SET_SOURCES:
        raise ValueError(f"unknown SET source {source!r}; expected one of {SET_SOURCES}")
    cols = {"combined": f"dG_SET_{source} / kcal mol-1"}
    if source == "exp":
        cols["bound"] = "dG_SET_exp_bound"
    return cols


# The rate cutoff the ground truth used to be set on. Nothing is labelled or classified
# on it any more -- the label is K_SV > 10 and the matching rate cutoff is per-pair,
# k_q_floor. Kept so the old reading can still be quoted for comparison.
K_Q_THRESHOLD = 1.0e7

# The Faraday constant in the units the driving forces use, for anyone converting a
# potential difference by hand. The dataset uses 23.060548.
F_KCAL_PER_MOL_V = 23.0605

# Photocatalysts in the dataset that no theory is scored on. The dataset keeps every
# screened photocatalyst; this is where one is set aside, with the reason, and load_pairs()
# says so every time it does it.
EXCLUDED_PHOTOCATALYSTS = {
    "PC22": "old PC24, [Ir(d(OMe)ppy)2(dtbbpy)]PF6. Its emission was too weak for the "
            "Stern-Volmer intensities to be trusted, so it is not used for the theory "
            "analyses, as stated in the SI.",
}

# Photocatalysts whose DFT input is known to be suspect, and which are still scored.
# Nothing is dropped on account of this list: it exists so a run says out loud what is in
# the numbers. `run_benchmark.py --exclude-pc` drops one for a sensitivity check, and
# reports that it did.
FLAGGED_PHOTOCATALYSTS = {
    "PC17": "old PC19. Imaginary frequencies in both the S0 and the reduced optimization. "
            "It has "
            "a triplet energy but no triplet-surface reorganization energy, so its 48 "
            "pairs drop out of every barrier and rate method. They stay in the "
            "adiabatic and DvTE numbers, which need only E_T.",
    "PC18": "old PC20. Lambda on the triplet surface is 33.67 kcal/mol against 12.70 on the "
            "singlet surface, a 2.7x asymmetry no other non-pmb complex shows. Never "
            "verified. It is the pair of numbers the asymmetric Marcus form is most "
            "sensitive to.",
}


class MissingColumns(KeyError):
    """A required column is not in exp_dataset.csv.

    Raised instead of scoring on whatever subset happens to be available. The message
    names the columns, who wanted them, and where their definition lives.
    """

    def __str__(self):  # KeyError would otherwise repr() the message and eat newlines
        return self.args[0]


def load_pairs(path: Path = PAIRS_CSV, verbose: bool = True) -> pd.DataFrame:
    """Read the pair table, minus EXCLUDED_PHOTOCATALYSTS. No imputation, no derived
    columns, and no other filtering."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:  uv run data/build_exp_dataset.py"
        )
    df = pd.read_csv(path)
    dropped = df[PC_NAME].isin(EXCLUDED_PHOTOCATALYSTS)
    if verbose:
        for name, reason in EXCLUDED_PHOTOCATALYSTS.items():
            print(f"not scored: {name} ({int((df[PC_NAME] == name).sum())} pairs), {reason}")
    return df[~dropped].reset_index(drop=True)


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
        f"{wanted_by} needs {len(missing)} column(s) that exp_dataset.csv does not have:",
        *(f"    {c!r}" for c in missing),
        "",
        "Every column name this benchmark expects is defined at the top of",
        "benchmark/pairs.py, with the sign and surface conventions each one follows.",
        "If the dataset renamed something, change it there and nowhere else.",
        "See data/build_exp_dataset.py for what the table carries and why.",
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

    Returns {name: (subset, y)}. Both cover all 1248 scored pairs -- nothing is dropped now that
    the label is set on the observable -- and they differ only in how permissive the
    criterion is:

      'inclusive'  y = `quenches`. K_SV > 10, or K_SV <= 10 with a slope significant
                   across the points the fit used. 790 positives.

      'strict'     y = `quenches_strict`. The published criterion, K_SV >= 25 and
                   r2 >= 0.90. 634 positives.

    The 156 pairs that separate them are weak but real quenchers. Nothing here chooses
    between the two; the gap between them is a headline the benchmark has to show, not a
    detail to settle before reporting.
    """
    require(df, [QUENCHES, QUENCHES_STRICT], "the ground truth")
    inclusive = df[QUENCHES].astype(int).astype(bool)
    strict = df[QUENCHES_STRICT].astype(int).astype(bool)
    # Strict is a subset of inclusive by construction: K_SV >= 25 implies K_SV > 10. If
    # that ever stops holding, one of the two has been redefined upstream and the
    # benchmark cannot tell which is now the ground truth.
    if (strict & ~inclusive).any():
        raise ValueError(
            f"{int((strict & ~inclusive).sum())} pairs quench under the strict criterion "
            "but not the inclusive one, which cannot happen while the strict K_SV cutoff "
            "is the higher of the two. One of the two has been redefined in "
            "data/experimental_data_scripts/stern_volmer_fits.py."
        )
    return {"inclusive": (df, inclusive), "strict": (df, strict)}


def describe(df: pd.DataFrame) -> str:
    """One block of text saying what was loaded and what is known to be odd about it."""
    flagged = sorted(set(df[PC_NAME]) & set(FLAGGED_PHOTOCATALYSTS))
    lines = [
        f"{len(df)} pairs  ({df[PC_NAME].nunique()} photocatalysts x "
        f"{df[Q_NAME].nunique()} quenchers)",
        f"  quenches            {int(df[QUENCHES].sum()):5d}   K_SV > 10, or a "
        f"significant slope below it",
        f"  quenches_strict     {int(df[QUENCHES_STRICT].sum()):5d}   published: "
        f"K_SV >= 25 and r2 >= 0.90",
        f"  censored            {int(df[CENSORED].sum()):5d}   below the K_SV floor; "
        f"K_SV is an upper bound for these",
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
