# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""
Score every theory in this folder against the measured rate constants, on one footing.

Reads   ../dataset/pairs.csv
Writes  results/benchmark_full.csv     every method at every setting, both label regimes
        results/benchmark_best.csv     the best setting per method family, plus its
                                       leave-one-photocatalyst-out score
        results/label_regimes.csv      each method under both readings of the ground truth
        results/rate_agreement.csv     predicted rate constants against measured ones

    uv run benchmark/run_benchmark.py
    uv run benchmark/run_benchmark.py --theories marcus rehm_weller
    uv run benchmark/run_benchmark.py --mechanism-rules none strict relaxed dominant_ddG_5
    uv run benchmark/run_benchmark.py --exclude-pc PC19 PC20
    uv run benchmark/run_benchmark.py --legacy-lambda-pairing

ON ONE FOOTING MEANS
    Same pairs, same ground truth, same metric, same sweep discipline. Every method is
    evaluated under both readings of the ground truth, both are written out, and neither
    is presented as the answer. Every method carries its free-parameter count, and every
    method with a free parameter also carries a leave-one-photocatalyst-out score beside
    its tuned one.

THE TWO NUMBERS THAT ARE NOT THE SAME NUMBER
    inclusive    all 1248 pairs, y = `quenches`: K_SV > 10, or K_SV <= 10 with a
                 slope significant across the points the fit used. 458 negatives.
    strict       all 1248 pairs, y = `quenches_strict`: the published K_SV >= 25 and
                 r2 >= 0.90. 614 negatives.

    The gap between them is not a rounding difference. The classification cutoff of
    1e7 M-1 s-1 lies below the Stern-Volmer detection floor `25 / tau_0` for 19 of the 26
    photocatalysts, covering 912 of the 1248 pairs, so for most of the screen "did not
    quench" records that the measurement could not resolve a rate constant, not that the
    rate constant is small. PC10's floor is 3.35e8 M-1 s-1: it could have a pair going at
    3e8 and still be counted a non-quencher.

    397 of the 553 apparent non-quenchers are in that position. A method scored against
    all of them has its specificity measured partly against pairs that carry no
    information, and its MCC with it.

    Which regime the paper should report is a scientific question about what counts
    as quenching, not a data-processing one. The 156 pairs between the two are weak
    but real: monotone 3-4 % signal, well above read noise, at rate constants the
    published criterion never counted. Both regimes are always scored and both are
    always reported; this harness takes no position.

WHAT IS NOT SCORED HERE
    Single-electron transfer as a classification target. The original project removed it
    and said why: with the mechanism filters applied there were only 9 to 55
    unambiguously SET-only positives, too few to carry a benchmark number. The SET
    driving forces are still used -- as a filter on which quenching pairs are admitted as
    energy-transfer positives, which is what `--mechanism-rules` selects -- but SET is
    never a label here.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import adiabatic
import dvte
import marcus
import pairs
import rehm_weller
import sandros
import scoring
import theory

RESULTS = Path(__file__).resolve().parent / "results"

# Ordered so the printed table reads from the zero-parameter baseline outward.
THEORIES = [adiabatic, dvte, sandros, marcus, rehm_weller]

# Rules for admitting a quenching pair as an energy-transfer positive, by ruling out the
# competing single-electron-transfer channel. Ported from the original's
# build_channel_datasets.py. Non-quenchers are never dropped by any rule: they carry no
# mechanistic label, so they are legitimate negatives either way.
#
#   none              admit every quenching pair. Nothing is dropped and no potential is
#                     compared, so no decision about mechanism enters the benchmark. The
#                     default, because that decision is not this folder's to make.
#   strict            dG_SET > +5 kcal/mol, single-electron transfer strongly uphill
#   relaxed           dG_SET > 0, uphill at all
#   dominant_ddG_5    dG_EnT < dG_SET - 5, energy transfer more exergonic by 5 kcal/mol
MECHANISM_RULES = ("none", "strict", "relaxed", "dominant_ddG_5")
SET_UPHILL_THRESHOLD = 5.0
ENT_DOMINANCE_DELTA = 5.0

_SCALE_TAG = re.compile(r"_a\d*\.?\d+$")


def method_type(family: str) -> str:
    """Collapse the per-alpha families into one reportable method.

    `marcus_barrier_sym` keeps raw DFT lambda and is its own method;
    `marcus_barrier_sym_a0.45` and its fifteen siblings collapse into
    `marcus_barrier_sym_scaled`, one method whose lambda scale is chosen along with its
    threshold. Keeping the two apart is the point: the scaled method has one more free
    parameter than the unscaled one, and reporting only the winner of the sweep would
    hide that a sweep happened.

    This is also the unit the threshold and the lambda scale are selected over, so the
    leave-one-photocatalyst-out score for a `_scaled` method tests whether the choice of
    alpha generalizes across photocatalysts -- not only the choice of threshold at a
    fixed alpha, which is a much weaker question.
    """
    match = _SCALE_TAG.search(family)
    return family[:match.start()] + "_scaled" if match else family


def mechanism_mask(df, rule, set_source="exp", bounds="face"):
    """Rows to keep under a mechanism rule, and a note about what it did.

    THE BOUND PROBLEM, WHICH THIS DOES NOT SOLVE ON ITS OWN AUTHORITY
        165 pairs have a `dG_SET_exp` that is a lower bound: the quencher was not
        oxidized or reduced anywhere inside the accessible potential window, so the true
        driving force can only be MORE endergonic than the number in the column. All
        three rules ask whether single-electron transfer is sufficiently uphill, so a
        lower bound always points towards "yes, it is ruled out" -- and reading the bound
        at face value therefore understates how inaccessible the channel is, in the
        direction that wrongly drops pairs out of the energy-transfer positives. That is
        the specific mistake dataset/README.md says the bound flags exist to prevent.

        `bounds='face'` reads the number as written, which is what the original did.
        `bounds='respect'` treats a lower-bounded pair as one where single-electron
        transfer cannot be shown to be accessible, so the pair is admitted. Neither is
        obviously right: 'face' can drop genuine energy-transfer pairs, 'respect' can
        admit pairs that really do go by electron transfer. The default is 'face' so the
        port reproduces the original, and the count of pairs the two disagree about is
        printed and carried into the output every time a rule is used.
    """
    if rule == "none":
        return (pd.Series(True, index=df.index),
                "every quenching pair admitted; no mechanism decision made")

    dG_set, is_lower_bound = pairs.dG_set(df, set_source)
    pairs.require(df, [pairs.DG_ENT], f"the {rule!r} mechanism rule")
    dG_ent = df[pairs.DG_ENT].astype(float)
    quencher = df[pairs.QUENCHES].astype(int) == 1

    if rule == "strict":
        ruled_out = dG_set > SET_UPHILL_THRESHOLD
    elif rule == "relaxed":
        ruled_out = dG_set > 0
    elif rule.startswith("dominant_ddG_"):
        ruled_out = dG_ent < dG_set - ENT_DOMINANCE_DELTA
    else:
        raise ValueError(f"unknown mechanism rule {rule!r}; expected one of {MECHANISM_RULES}")

    disputed = quencher & ~ruled_out & is_lower_bound
    if bounds == "respect":
        ruled_out = ruled_out | is_lower_bound
    elif bounds != "face":
        raise ValueError(f"unknown bounds policy {bounds!r}; expected 'face' or 'respect'")

    keep = (~quencher) | ruled_out
    reading = ("lower bounds read at face value" if bounds == "face"
               else "lower bounds treated as 'cannot be shown accessible'")
    disagreement = (f"{int(disputed.sum())} of them turn on a lower-bounded dG_SET and "
                    f"would be admitted under --bounds respect" if bounds == "face"
                    else f"{int(disputed.sum())} pairs were admitted on the strength of a "
                         f"lower bound and would be dropped under --bounds face")
    note = (f"dG_SET from the {set_source} potentials, {reading}; "
            f"{int((~keep).sum())} quenching pairs dropped as mechanistically ambiguous. "
            f"{disagreement}.")
    return keep, note


def preflight(df, modules, rules, set_sources):
    """One error naming every column the selected work needs and does not have."""
    needed = {}
    for mod in modules:
        for col in mod.required_columns():
            needed.setdefault(col, []).append(mod.NAME)
    if any(rule != "none" for rule in rules):
        needed.setdefault(pairs.DG_ENT, []).append("mechanism rules")
        for source in set_sources:
            needed.setdefault(pairs.set_columns(source)["combined"], []).append(
                f"mechanism rules ({source})")

    missing = [c for c in needed if c not in df.columns]
    if not missing:
        return
    who = "; ".join(f"{c} (for {', '.join(sorted(set(needed[c])))})" for c in missing)
    pairs.require(df, missing, f"this run [{who}]")


def undefined_rows(df, mod):
    """(count, photocatalysts) for rows where a column the theory needs is not finite."""
    bad = pd.Series(False, index=df.index)
    for col in mod.required_columns():
        if col not in df.columns:
            continue
        bad |= ~np.isfinite(pd.to_numeric(df[col], errors="coerce"))
    return int(bad.sum()), sorted(set(df.loc[bad, pairs.PC_NAME]))


def run(df, modules, rules, set_sources, bounds, k_diff, mean_type, legacy_pairing):
    """Everything, for every (mechanism rule x SET source x label regime x setting)."""
    full_rows, best_rows, regime_rows, rate_rows = [], [], [], []

    for rule in rules:
        sources = ["-"] if rule == "none" else list(set_sources)
        for source in sources:
            keep, note = mechanism_mask(df, rule, set_source=source, bounds=bounds)
            subset = df[keep].reset_index(drop=True)
            tag = rule if rule == "none" else f"{rule}/{source}"
            print(f"\n--- mechanism rule: {tag}   ({len(subset)} pairs)")
            print(f"    {note}")

            regimes = pairs.label_regimes(subset)
            positions = {name: np.flatnonzero(subset.index.isin(rows.index))
                         for name, (rows, _) in regimes.items()}
            context = {"mechanism_rule": rule, "set_source": source,
                       "bounds_policy": bounds if rule != "none" else "-"}

            for mod in modules:
                # Every prepare() takes the same three knobs; the ones with no use for
                # them swallow them, so the harness never has to know which is which.
                prep = mod.prepare(subset, k_diff=k_diff, mean_type=mean_type,
                                   legacy_pairing=legacy_pairing)
                n_undefined, affected = undefined_rows(subset, mod)
                print(f"    {mod.NAME:12s} {mod.DESCRIPTION}")
                extra = getattr(mod, "note", None)
                if extra:
                    print(f"    {'':12s} {extra(prep)}")
                if n_undefined:
                    print(f"    {'':12s} WARNING: {n_undefined} of {len(subset)} pairs "
                          f"have a non-finite input ({', '.join(affected)}). They are "
                          f"predicted 'does not quench' and scored as such, which is a "
                          f"free pass on every true negative among them.")

                configs = mod.grid()
                if configs:
                    predictions = np.vstack([np.asarray(mod.classify(prep, c), dtype=bool)
                                             for c in configs])
                    _score_configs(mod, configs, predictions, regimes, positions,
                                   context, n_undefined, full_rows, best_rows,
                                   regime_rows)

                for cfg in mod.rate_grid():
                    k_pred = mod.predicted_rate(prep, cfg)
                    if k_pred is None:
                        continue
                    rate_rows.append({
                        **context, "theory": mod.NAME, "family": cfg.family,
                        "method": cfg.name, "n_free": cfg.n_free,
                        "k_diff / M-1 s-1": k_diff,
                        **scoring.rate_agreement(subset[pairs.K_Q],
                                                 subset[pairs.K_Q_FLOOR],
                                                 subset[pairs.CENSORED], k_pred),
                    })

    return (scoring.as_frame(full_rows), scoring.as_frame(best_rows),
            scoring.as_frame(regime_rows), scoring.as_frame(rate_rows))


def _score_configs(mod, configs, predictions, regimes, positions, context, n_undefined,
                   full_rows, best_rows, regime_rows):
    index_of = {c.name: i for i, c in enumerate(configs)}
    by_type = {}
    for cfg in configs:
        by_type.setdefault(method_type(cfg.family), []).append(cfg)
    best_index = {}

    for regime, (rows, y) in regimes.items():
        pos = positions[regime]
        y_arr = np.asarray(y, dtype=bool)
        groups = rows[pairs.PC_NAME].to_numpy()

        for cfg in configs:
            full_rows.append({
                **context, "label_regime": regime, "theory": mod.NAME,
                "method_type": method_type(cfg.family), "family": cfg.family,
                "method": cfg.name, "n_free": cfg.n_free,
                "n_undefined_inputs": n_undefined,
                **scoring.metrics(y_arr, predictions[index_of[cfg.name]][pos]),
                **{f"param_{k}": v for k, v in cfg.params.items() if k != "kind"},
            })

        for mtype, members in by_type.items():
            matrix = np.vstack([predictions[index_of[c.name]][pos] for c in members])
            pick = scoring.best_by_mcc(y_arr, matrix)
            chosen = members[pick]
            best_index[(regime, mtype)] = index_of[chosen.name]
            lopo = scoring.leave_one_photocatalyst_out(y_arr, matrix, groups)
            best_rows.append({
                **context, "label_regime": regime, "theory": mod.NAME,
                "method_type": mtype, "family": chosen.family, "method": chosen.name,
                "n_free": chosen.n_free, "n_settings_swept": len(members),
                "n_undefined_inputs": n_undefined,
                **scoring.metrics(y_arr, matrix[pick]),
                "MCC_leave_one_pc_out": lopo["MCC"],
                "Accuracy_leave_one_pc_out": lopo["Accuracy"],
                "settings_chosen_across_folds": lopo["n_settings_chosen"],
                **{f"param_{k}": v for k, v in chosen.params.items() if k != "kind"},
            })

    # The comparison the reader has to see: the same method under both readings of the
    # ground truth, including what the operating point chosen on the full table does on
    # the subset the experiment can actually adjudicate.
    for mtype, members in by_type.items():
        row = {**context, "theory": mod.NAME, "method_type": mtype,
               "n_free": max(c.n_free for c in members)}
        for regime, (_rows, y) in regimes.items():
            pos = positions[regime]
            m = scoring.metrics(np.asarray(y, dtype=bool),
                                predictions[best_index[(regime, mtype)]][pos])
            row[f"n_{regime}"] = m["n"]
            row[f"n_positive_{regime}"] = m["n_positive"]
            row[f"MCC_{regime}"] = m["MCC"]
            row[f"Accuracy_{regime}"] = m["Accuracy"]
            row[f"Sensitivity_{regime}"] = m["Sensitivity"]
            row[f"Specificity_{regime}"] = m["Specificity"]
        if "inclusive" in regimes and "strict" in regimes:
            carried = predictions[best_index[("inclusive", mtype)]]
            y_strict = np.asarray(regimes["strict"][1], dtype=bool)
            row["MCC_strict_at_inclusive_operating_point"] = scoring.metrics(
                y_strict, carried[positions["strict"]])["MCC"]
            row["dMCC_inclusive_minus_strict"] = row["MCC_inclusive"] - row["MCC_strict"]
        regime_rows.append(row)


def print_regime_comparison(regimes_df, best_df):
    """The headline table: every method under both readings of the ground truth."""
    if regimes_df.empty:
        return
    print("\n" + "=" * 108)
    print("THE SAME METHODS UNDER BOTH READINGS OF THE GROUND TRUTH")
    print("=" * 108)
    print("inclusive   every pair, y = quenches: K_SV > 10, or a significant slope below it.")
    print("strict      every pair, y = quenches_strict: the published K_SV >= 25 and r2 >= 0.90.")
    print("carried     the setting chosen on 'inclusive', scored on 'strict'.")
    print("lopo        leave-one-photocatalyst-out on 'inclusive': the setting is chosen on the")
    print("            other photocatalysts and applied to the held-out one. For a method")
    print("            with no free parameter it equals the tuned score by construction.")
    print("free        free parameters selected by maximizing MCC on the scored data.")
    print()
    print("Neither MCC column is the answer. The scientific choice between them is open --")
    print("see dataset/README.md and the README beside this script.")
    print()

    best_inclusive = best_df[best_df["label_regime"] == "inclusive"].set_index(
        ["mechanism_rule", "set_source", "method_type"])
    view = regimes_df.copy()
    keys = list(zip(view["mechanism_rule"], view["set_source"], view["method_type"]))
    view["lopo"] = [best_inclusive["MCC_leave_one_pc_out"].get(k, float("nan")) for k in keys]
    view["alpha"] = [best_inclusive["param_lam_scale"].get(k, float("nan")) for k in keys] \
        if "param_lam_scale" in best_inclusive.columns else float("nan")

    rank = {mod.NAME: i for i, mod in enumerate(THEORIES)}
    view["_rank"] = view["theory"].map(rank).fillna(99)
    view = view.sort_values(["mechanism_rule", "set_source", "_rank", "MCC_inclusive"],
                            ascending=[True, True, True, False])
    cols = ["mechanism_rule", "set_source", "method_type", "n_free", "alpha",
            "n_inclusive", "MCC_inclusive", "lopo", "n_strict", "MCC_strict",
            "MCC_strict_at_inclusive_operating_point", "dMCC_inclusive_minus_strict"]
    view = view[[c for c in cols if c in view.columns]].rename(columns={
        "mechanism_rule": "rule", "set_source": "SET", "method_type": "method",
        "n_free": "free", "MCC_strict_at_inclusive_operating_point": "carried",
        "dMCC_inclusive_minus_strict": "dMCC"})
    print(view.round(3).to_string(index=False))
    print("\nEvery setting of every method is in results/benchmark_full.csv; the chosen")
    print("setting and its parameters are in results/benchmark_best.csv.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--theories", nargs="*", default=[m.NAME for m in THEORIES],
                        help="subset of: " + ", ".join(m.NAME for m in THEORIES))
    parser.add_argument("--mechanism-rules", nargs="*", default=["none"],
                        help=f"subset of {MECHANISM_RULES}. Default 'none': every "
                             f"quenching pair is admitted and no mechanism decision "
                             f"enters the benchmark.")
    parser.add_argument("--set-source", nargs="*", default=list(pairs.SET_SOURCES),
                        choices=list(pairs.SET_SOURCES),
                        help="which dG_SET to filter on: 'exp' from the measured "
                             "quencher potentials, 'calc' from the computed ones. Both "
                             "by default. Ignored when --mechanism-rules is 'none'.")
    parser.add_argument("--bounds", choices=("face", "respect"), default="face",
                        help="how a lower-bounded dG_SET is read by a mechanism rule. "
                             "See mechanism_mask.")
    parser.add_argument("--k-diff", type=float, default=theory.K_DIFF_ACETONITRILE,
                        help=f"bimolecular diffusion limit, M-1 s-1. Default "
                             f"{theory.K_DIFF_ACETONITRILE:g}, the acetonitrile "
                             f"literature value. A solvent property, so it is not swept.")
    parser.add_argument("--mean-type", choices=("geometric", "arithmetic"),
                        default="geometric",
                        help="how the forward and reverse reorganization energies are "
                             "combined into one effective lambda.")
    parser.add_argument("--legacy-lambda-pairing", action="store_true",
                        help="assign lambda_forward and lambda_reverse the way the "
                             "original implementation did, which pairs the "
                             "photocatalyst's triplet-surface term with the quencher's "
                             "singlet-surface term. See marcus.py. Affects only the "
                             "asymmetric Marcus form.")
    parser.add_argument("--exclude-pc", nargs="*", default=[],
                        help="photocatalysts to drop, by pc_name. Nothing is dropped by "
                             "default; see the notice printed for flagged photocatalysts.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args(argv)

    unknown = set(args.theories) - {m.NAME for m in THEORIES}
    if unknown:
        sys.exit(f"unknown theory: {', '.join(sorted(unknown))}")
    bad_rules = set(args.mechanism_rules) - set(MECHANISM_RULES)
    if bad_rules:
        sys.exit(f"unknown mechanism rule: {', '.join(sorted(bad_rules))}")
    modules = [m for m in THEORIES if m.NAME in args.theories]

    df = pairs.load_pairs()
    print(pairs.describe(df))

    present = sorted(set(df[pairs.PC_NAME]) & set(pairs.FLAGGED_PHOTOCATALYSTS))
    if present:
        print("\nPhotocatalysts flagged as suspect, present and NOT excluded:")
        for name in present:
            print(f"  {name}: {pairs.FLAGGED_PHOTOCATALYSTS[name]}")
        print("  Nothing is dropped on account of this list. Use --exclude-pc for the")
        print("  sensitivity run, and report both.")

    if args.exclude_pc:
        before = len(df)
        df = df[~df[pairs.PC_NAME].isin(args.exclude_pc)].reset_index(drop=True)
        print(f"\nExcluded {', '.join(args.exclude_pc)}: {before} -> {len(df)} pairs")

    if args.legacy_lambda_pairing:
        print("\nUsing the original implementation's lambda_forward / lambda_reverse "
              "assignment.\nSee marcus.py for why it is not the default.")

    preflight(df, modules, args.mechanism_rules, args.set_source)

    full, best, regimes_df, rates = run(
        df, modules, args.mechanism_rules, args.set_source, args.bounds,
        args.k_diff, args.mean_type, args.legacy_lambda_pairing)

    print_regime_comparison(regimes_df, best)

    if not rates.empty:
        print("\n" + "=" * 108)
        print("RATE PREDICTIONS AGAINST THE MEASURED RATE CONSTANTS")
        print("=" * 108)
        print("Detected pairs get a residual. Censored pairs get consistency with their")
        print("own detection floor, because their k_q is an upper bound and a residual")
        print("against a bound means nothing. The two are not combined into one score.\n")
        show = ["family", "n_free", "n_detected", "rmse_dex_detected",
                "median_bias_dex_detected", "n_censored", "fraction_consistent_with_floor"]
        first = rates[rates["mechanism_rule"] == rates["mechanism_rule"].iloc[0]]
        print(first[[c for c in show if c in first.columns]].round(3).to_string(index=False))

    if not args.no_write:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        print()
        for name, frame in (("benchmark_full", full), ("benchmark_best", best),
                            ("label_regimes", regimes_df), ("rate_agreement", rates)):
            path = args.results_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            print(f"{len(frame):7d} rows -> {path.relative_to(args.results_dir.parent)}")

    return full, best, regimes_df, rates


if __name__ == "__main__":
    main()
