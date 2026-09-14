# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "openpyxl>=3.1", "pandas"]
# ///
"""
The pair table, and the column contract every theory in this folder reads it through.

Reads   ../data/exp_dataset.csv
        ../data/computational_data_processed/photocatalyst_properties_calc.xlsx
        ../data/computational_data_processed/quencher_properties_calc.xlsx
        ../data/experimental_data_processed/quencher_properties_exp.xlsx
Writes  nothing

WHY THIS MODULE EXISTS
    Every column name in the benchmark is written down once, here. The theory modules
    take arrays, never a file path and never a header string, so a rename upstream breaks
    in one place with a message instead of in six places with a KeyError. This is the
    centralization rule the original implementation kept in `property_retrieval.py`, minus
    the part that made it unreproducible: nothing is copied in from a sibling directory,
    every path is inside ../data/, and importing this module has no side effect beyond
    reading one CSV and three property workbooks.

    It also carries what the column names mean. See "HOW THE LAMBDAS ARE NAMED" below.

WHERE THE CALCULATED COLUMNS COME FROM
    `exp_dataset.csv` is the experimental table -- ids, K_SV, tau_0, the rate constants
    derived from them and the labels -- and carries nothing calculated. `load_pairs()`
    joins the calculated properties on by id and derives the driving forces, which is the
    arithmetic that used to sit in `data/build_exp_dataset.py`, moved here unchanged.

    That move is mechanical. How the pairs ought to be assembled, and whether the driving
    forces belong in this module at all rather than beside the theories that use them, is
    an open question and deliberately not answered here.

THE TWO LABEL REGIMES
    The ground truth is set on K_SV, the observable, and not on k_q. A cutoff on k_q was
    incoherent: k_q = K_SV / tau_0, so the Stern-Volmer detection floor moves with the
    photocatalyst, and at a 1e7 cutoff that floor sat ABOVE the cutoff for 19 of the 26
    photocatalysts. 465 pairs could not be labelled at all. On K_SV the floor is one
    number for every photocatalyst and every pair is labelled by the same criterion, so
    the indeterminate class is gone and with it the 'resolvable' regime.

    What remains is a choice of how permissive the criterion is, and `exp_dataset.csv` carries
    both readings:

      `quenches`         a ladder of (K_SV, r2) rungs, weaker quenching needing a
                         straighter line. 735 of 1248.
      `quenches_strict`  the published criterion, K_SV >= 25 and r2 >= 0.90. 637 of 1248.

    The 98 pairs between them are weak quenchers: real monotone 3-4 % signal, well above
    read noise, at rate constants the strict criterion never counted. Whether the paper
    claims them is a scientific question about what "quenching" means, not a data
    question, so `label_regimes()` returns both and this module takes no position.

HOW THE LAMBDAS ARE NAMED
    lambda_<surface>_<geometry>, for photocatalysts and quenchers alike, as in both
    workbooks: the energy of one state at another state's optimized geometry, minus that
    state at its own.

        lambda_S0_T1 = E_S0(T1 geometry) - E_S0(S0 geometry)     the singlet surface
        lambda_T1_S0 = E_T1(S0 geometry) - E_T1(T1 geometry)     the triplet surface

    The pair table prefixes the partner: `pc_lambda_S0_T1_dft`, `q_lambda_T1_S0_dft`.
    The code this benchmark is ported from used two schemes whose letters read in
    opposite orders, and paired them wrongly -- see marcus.py.
"""

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

DATA = Path(__file__).resolve().parent.parent / "data"
PAIRS_CSV = DATA / "exp_dataset.csv"
PC_CALC_XLSX = DATA / "computational_data_processed" / "photocatalyst_properties_calc.xlsx"
Q_CALC_XLSX = DATA / "computational_data_processed" / "quencher_properties_calc.xlsx"
Q_EXP_XLSX = DATA / "experimental_data_processed" / "quencher_properties_exp.xlsx"

# --- identity ---------------------------------------------------------------------
# Photocatalysts are named by their new id, PC1-PC27; `pc_old_id` is the number the raw
# files use. See data/compound_identification.py.
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
Q_LAMBDA_S0_T1 = "q_lambda_S0_T1_dft / kcal mol-1"
Q_LAMBDA_T1_S0 = "q_lambda_T1_S0_dft / kcal mol-1"
Q_DVTE_MU = "q_dvte_mu_md / kcal mol-1"
Q_DVTE_STD = "q_dvte_std_md / kcal mol-1"

# --- photocatalyst, computed ---------------------------------------------------------
PC_E_T = "pc_E_T_dft / kcal mol-1"
PC_LAMBDA_S0_T1 = "pc_lambda_S0_T1_dft / kcal mol-1"
PC_LAMBDA_T1_S0 = "pc_lambda_T1_S0_dft / kcal mol-1"

# --- driving forces ------------------------------------------------------------------
# Quencher-centric, negative = exergonic, F = 23.0605 kcal mol-1 V-1:
#     dG_EnT     = E_T(Q) - E_T(PC)
#     dG_SET_ox  = F * (E_Ox(Q)     - E(PC*/PC-))     Q oxidized, PC* is the oxidant
#     dG_SET_red = F * (E(PC+/PC*)  - E_Red(Q))       Q reduced,  PC* is the reductant
#     dG_SET     = whichever channel is more accessible
DG_ENT = "dG_EnT / kcal mol-1"

# The SET driving force exists twice over, from the measured quencher potentials and from
# the computed ones. They are kept apart deliberately -- a measured potential and a DFT
# potential are not interchangeable, and which one an analysis used is part of what it
# reports -- so both are derived, neither is merged, and every output row says which.
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
# on it any more -- the label reads K_SV and r2, and the per-pair detection floor is
# k_q_floor. Kept so the old reading can still be quoted for comparison.
K_Q_THRESHOLD = 1.0e7

# The Faraday constant in the units the driving forces use, for anyone converting a
# potential difference by hand. The driving forces themselves use F_KCAL_PER_V below,
# which carries the digits the dataset was built with.
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
            "a triplet energy but no triplet-surface lambda_T1_S0, so its 48 "
            "pairs drop out of every barrier and rate method. They stay in the "
            "adiabatic and DvTE numbers, which need only E_T.",
    "PC18": "old PC20. Lambda on the triplet surface is 33.67 kcal/mol against 12.70 on the "
            "singlet surface, a 2.7x asymmetry no other non-pmb complex shows. Never "
            "verified. It is the pair of numbers the asymmetric Marcus form is most "
            "sensitive to.",
}


# --- the join ------------------------------------------------------------------------
# Everything from here to load_pairs() is `data/build_exp_dataset.py` as it stood before
# the calculated columns left it: the same workbook reader, the same Faraday constant, the
# same "more accessible channel" rule, the same six-significant-figure rounding. It was
# moved rather than rewritten, so the numbers are the ones the dataset used to carry.

# Row 1 of every property workbook names the column blocks; the headers are on row 2.
WORKBOOK_HEADER_ROW = 2

SIGFIGS = 6
F_KCAL_PER_V = 23.060548


def _load(path: Path, key: str) -> dict:
    """A property workbook as {key: row}, blanks as "". A missing file is fatal."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:  uv run reproduce.py data"
        )
    sheet = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = sheet.iter_rows(min_row=WORKBOOK_HEADER_ROW, values_only=True)
    header = next(rows)
    table = {}
    for values in rows:
        row = {h: ("" if v is None else v) for h, v in zip(header, values) if h}
        if row.get(key) not in ("", None):
            table[int(row[key])] = row
    return table


def _sig(x, n=SIGFIGS):
    if x is None:
        return None
    return 0.0 if float(x) == 0 else float(f"{float(x):.{n}g}")


def _number(row, field):
    """A numeric field, or None when absent or blank."""
    if row is None:
        return None
    value = row.get(field, "")
    return float(value) if value not in ("", None) else None


def _better_channel(ox, red):
    """The more accessible SET channel, (value, 'ox'|'red'), or (None, '') if neither."""
    available = [(v, name) for v, name in ((ox, "ox"), (red, "red")) if v is not None]
    return min(available) if available else (None, "")


def _computed(df: pd.DataFrame) -> pd.DataFrame:
    """The calculated columns and the driving forces, one row per row of `df`.

    Joined by id and nothing else: the photocatalyst workbook by `pc_old_id`, both
    quencher workbooks by `quencher_id`. A value the calculation does not have stays
    blank -- it is never filled in and never turned into a zero.
    """
    pc_calc = _load(PC_CALC_XLSX, "old_id")
    q_calc = _load(Q_CALC_XLSX, "quencher_id")
    q_exp = _load(Q_EXP_XLSX, "quencher_id")
    exp_cols, calc_cols = set_columns("exp"), set_columns("calc")

    rows = []
    for pc_id, q_id in zip(df[PC_OLD_ID], df[Q_ID]):
        pcc, qc, qe = pc_calc.get(int(pc_id)), q_calc.get(int(q_id)), q_exp.get(int(q_id))

        pc_e_t = _number(pcc, "E_T / kcal mol-1")
        q_e_t = _number(qc, "E_T / kcal mol-1")
        pc_star_ox = _number(pcc, "E_PCstar_PCminus / V")    # PC* as the oxidant
        pc_star_red = _number(pcc, "E_PCplus_PCstar / V")    # PC* as the reductant

        def set_forces(e_ox, e_red):
            ox = F_KCAL_PER_V * (e_ox - pc_star_ox) if None not in (e_ox, pc_star_ox) else None
            red = (F_KCAL_PER_V * (pc_star_red - e_red)
                   if None not in (e_red, pc_star_red) else None)
            return ox, red

        set_exp, channel = _better_channel(*set_forces(_number(qe, "E_Ox_vs_SCE_exp / V"),
                                                       _number(qe, "E_Red_vs_SCE_exp / V")))
        # Either bound becomes a lower bound on the driving force: the quencher was not
        # oxidized (or reduced) within the window, so SET is at least this endergonic.
        bound_column = {"ox": "E_Ox_bound", "red": "E_Red_bound"}.get(channel)
        set_calc, _ = _better_channel(*set_forces(_number(qc, "E_Ox_vs_SCE_calc / V"),
                                                  _number(qc, "E_Red_vs_SCE_calc / V")))

        rows.append({
            PC_E_T: _sig(pc_e_t),
            PC_LAMBDA_S0_T1: _sig(_number(pcc, "lambda_S0_T1 / kcal mol-1")),
            PC_LAMBDA_T1_S0: _sig(_number(pcc, "lambda_T1_S0 / kcal mol-1")),
            Q_LAMBDA_S0_T1: _sig(_number(qc, "lambda_S0_T1 / kcal mol-1")),
            Q_LAMBDA_T1_S0: _sig(_number(qc, "lambda_T1_S0 / kcal mol-1")),
            Q_DVTE_MU: _sig(_number(qc, "dvte_mu / kcal mol-1")),
            Q_DVTE_STD: _sig(_number(qc, "dvte_std / kcal mol-1")),
            DG_ENT: _sig(q_e_t - pc_e_t) if None not in (q_e_t, pc_e_t) else None,
            exp_cols["combined"]: _sig(set_exp),
            exp_cols["bound"]: "lower" if bound_column and (qe or {}).get(bound_column) else "",
            calc_cols["combined"]: _sig(set_calc),
        })
    return pd.DataFrame(rows, index=df.index)


class MissingColumns(KeyError):
    """A required column is not in the pair table.

    Raised instead of scoring on whatever subset happens to be available. The message
    names the columns, who wanted them, and where their definition lives.
    """

    def __str__(self):  # KeyError would otherwise repr() the message and eat newlines
        return self.args[0]


def load_pairs(path: Path = PAIRS_CSV, verbose: bool = True) -> pd.DataFrame:
    """Read the experimental pair table, join the calculated columns and the driving
    forces onto it, and drop EXCLUDED_PHOTOCATALYSTS. No imputation and no other
    filtering."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:  uv run data/build_exp_dataset.py"
        )
    df = pd.read_csv(path)
    df = pd.concat([df, _computed(df)], axis=1)
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
        f"{wanted_by} needs {len(missing)} column(s) that the pair table does not have:",
        *(f"    {c!r}" for c in missing),
        "",
        "Every column name this benchmark expects is defined at the top of",
        "benchmark/pairs.py, with the sign and surface conventions each one follows.",
        "If something upstream was renamed, change it there and nowhere else.",
        "The measured columns come from data/exp_dataset.csv; the calculated ones are",
        "joined on from the property workbooks by load_pairs().",
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

      'inclusive'  y = `quenches`. A ladder of (K_SV, r2) rungs, weaker quenching
                   needing a straighter line. 735 positives.

      'strict'     y = `quenches_strict`. The published criterion, K_SV >= 25 and
                   r2 >= 0.90. 637 positives.

    The 98 pairs that separate them are weak but real quenchers. Nothing here chooses
    between the two; the gap between them is a headline the benchmark has to show, not a
    detail to settle before reporting.
    """
    require(df, [QUENCHES, QUENCHES_STRICT], "the ground truth")
    inclusive = df[QUENCHES].astype(int).astype(bool)
    strict = df[QUENCHES_STRICT].astype(int).astype(bool)
    # Strict is a subset of inclusive by construction: K_SV >= 25 and r2 >= 0.90 clears
    # the ladder's `K_SV > 20 and r2 > 0.80` rung. If that ever stops holding, one of the
    # two has been redefined upstream and the benchmark cannot tell which is now truth.
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
        f"  quenches            {int(df[QUENCHES].sum()):5d}   the (K_SV, r2) ladder",
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
