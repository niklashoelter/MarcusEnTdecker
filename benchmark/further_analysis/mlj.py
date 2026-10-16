# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Marcus-Levich-Jortner (MLJ) rates from the mode-resolved reorganization energies, as
rate predictions and as feasibility criteria next to the two-gate criterion.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py, and for the four-point lambda of
        every molecule)
        ../../data/computational_data_processed/reorganization_modes.csv
        results/quencher_classes/quencher_classes.csv
Writes  results/mlj/metrics.csv       every criterion and source: confusion counts, MCC,
                                      recall, specificity and dMCC against the two-gate
                                      criterion with the same parameters (bootstrap 95% CI)
        results/mlj/classes.csv       the same confusion counts per quencher class
        results/mlj/rate_parity.csv   predicted against measured log10 k_q on the quenching
                                      pairs, projected quantum mode
        results/mlj/barriers.csv      median classical barrier and MLJ effective barrier per
                                      quencher class (DFT)
        results/mlj/summary.csv       medians of the MLJ parameters (DFT), the prefactor and
                                      tunnelling factors, and the lifetime thresholds at
                                      k_-d/nu = 0.25
        results/mlj/tuned.csv         the classical lambda chosen by the tuned variant

    uv run benchmark/further_analysis/mlj.py

Mode split. The reorganization energy of each molecule and surface is split at 3 k_B T into
a classical part (modes below) and a quantum part at the lambda-weighted frequency of the
modes above (reorganization_modes.csv). The internal-coordinate projection is used where its
sum lies within 30% of the four-point lambda, else the Cartesian one where that does; for
the remaining molecules, and for the ML quencher lambdas, the median split of the reliable
molecules of the same kind and surface is taken. Reverse (product state, PC on S0, Q on
T1): lambda_v = quantum parts of lambda_S0_T1(PC) and lambda_T1_S0(Q), hw their
lambda-weighted frequency, lambda_s,r = lambda_r - lambda_v. Forward: lambda_s,f = classical
parts of lambda_T1_S0(PC) and lambda_S0_T1(Q). Classical parts are floored at 0.05 kcal/mol.

Franck-Condon factor of one quantum mode with Huang-Rhys factor S = lambda_v / hw:

    F = sum_n exp(-S) S^n / n!  exp(-B(dG + n hw) / RT)

with B the symmetric barrier (dG + lambda_s,r)^2 / (4 lambda_s,r) or the asymmetric barrier
of theory.marcus_barrier_asymmetric with lambda_s,f and lambda_s,r. For S = 0 the asymmetric
form is exp(-barrier/RT) of the two-gate criterion. F takes the place of exp(-barrier/RT)
in the rate law,

    k_q = k_d / (1 + exp(dG/RT) + (k_-d/nu) / F)

with k_-d/nu = 0.25 or the Eyring value (k_d/K_A)/(k_B T/h) at the contact distance r_PC + r_Q
of each pair. Decisions: k_q tau0 >= K_SV,crit = 10 M-1 (rate), the same and dG_EnT <= 0, or
the effective barrier -RT ln F against the two-gate threshold t_PC and dG_EnT <= 0. The
quantum mode is the projected one or fixed at 1200, 1500 or 1800 cm-1.

Tuned variant: one classical lambda_s for all pairs (0.5 to 60 kcal/mol in steps of 0.5,
capped at the geometric-mean pair lambda), the rest of that lambda in a quantum mode of
1500 cm-1, symmetric line shape, k_-d/nu = 0.25, rate and dG_EnT <= 0. lambda_s is chosen
by the highest MCC with DFT parameters on all pairs, and leaving out each quencher (LOQO)
or photocatalyst (LOPCO) in turn; the ML row applies the DFT choice.

dMCC: two-way cluster bootstrap over photocatalysts and quenchers, 2000 replicates, fixed
seed; a pair is weighted by the number of copies of its photocatalyst times that of its
quencher.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "mlj"
MODES = pairs.HERE.parent / "data" / "computational_data_processed" / "reorganization_modes.csv"
CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"

RT = theory.RT
KCAL_PER_CM = 1.0 / 349.755         # kcal/mol per cm-1
K_SV_CRIT = 10.0                    # M-1
RELIABLE_TOLERANCE = 0.30           # projected sum within 30% of the four-point lambda
LAMBDA_FLOOR = 0.05                 # kcal/mol
N_CHANNELS = 90
PREFACTORS = ("Eyring", "k_-d/nu = 0.25")
ESCAPE_OVER_NU_LOW = 0.25            # k_-d/nu
QUANTUM_MODES = {"projected mode": None, "1200 cm-1": 1200.0, "1500 cm-1": 1500.0,
                 "1800 cm-1": 1800.0}
LINE_SHAPES = {"mlj_symmetric": "symmetric", "mlj_asymmetric": "asymmetric"}
DECISIONS = ("rate", "rate and dG <= 0", "effective barrier and dG <= 0")
TUNED_GRID = np.round(np.arange(0.5, 60.0001, 0.5), 2)     # kcal/mol
TUNED_MODE = 1500.0                                         # cm-1
HELD_OUT = {"LOQO": pairs.Q_NAME, "LOPCO": pairs.PC_NAME}
BOOTSTRAP_REPLICATES = 2000
SEED = 20261009
CLASS_ORDER = ("rigid", "flexible", "dissociative")
KCAL_COLUMNS = ("barrier", "lambda")


# ------------------------------------------------------------------ mode split
def molecule_split(modes, four_point):
    """Classical fraction and quantum frequency per (kind, id, surface): the internal
    projection if reliable, else the Cartesian one, else NaN. Returns the table and the
    medians of the reliable molecules per (kind, surface)."""
    table = modes.merge(four_point, on=["kind", "id", "surface"], validate="m:1")
    table["reliable"] = ((table["lambda / kcal mol-1"] / table["lambda_four_point"] - 1).abs()
                         <= RELIABLE_TOLERANCE)
    rows = []
    for (kind, number, surface), g in table.groupby(["kind", "id", "surface"]):
        g = g.set_index("displacement")
        chosen = next((d for d in ("internal", "cartesian") if g.loc[d, "reliable"]), None)
        rows.append({"kind": kind, "id": number, "surface": surface, "reliable": chosen is not None,
                     "classical_fraction": g.loc[chosen, "classical_fraction"] if chosen else np.nan,
                     "quantum_frequency": g.loc[chosen, "quantum_frequency / cm-1"] if chosen else np.nan})
    split = pd.DataFrame(rows)
    medians = (split[split.reliable].groupby(["kind", "surface"])
               [["classical_fraction", "quantum_frequency"]].median())
    for column in ("classical_fraction", "quantum_frequency"):
        fallback = [medians.loc[(k, s), column] for k, s in zip(split.kind, split.surface)]
        split[column] = np.where(split.reliable, split[column], fallback)
    return split.set_index(["kind", "id", "surface"]), medians


def four_point_lambdas(dataset):
    """Four-point lambda of every molecule in the dataset, as (kind, id, surface) rows."""
    rows = []
    for kind, key, prefix in (("photocatalyst", "pc_old_id", "pc"), ("quencher", pairs.Q_ID, "q")):
        first = dataset.groupby(key).first()
        for surface, geometry in (("S0", "T1"), ("T1", "S0")):
            column = f"{prefix}_lambda_{surface}_{geometry}_dft / kcal mol-1"
            rows += [{"kind": kind, "id": int(i), "surface": surface, "lambda_four_point": v}
                     for i, v in first[column].items()]
    return pd.DataFrame(rows)


def inputs(df, source, split, medians):
    """dG, pair lambdas and the MLJ split of every pair. Photocatalysts are DFT; ML quencher
    lambdas are split with the medians of the reliable quenchers."""
    dG = pairs.driving_force(df, source)
    pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0 = pairs.lambdas(df, source)
    forward, reverse = theory.pair_lambdas(pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0)

    def get(kind, ids, surface, column):
        return split.loc[[(kind, int(i), surface) for i in ids], column].to_numpy(float)

    pc_ids, q_ids = df["pc_old_id"].to_numpy(int), df[pairs.Q_ID].to_numpy(int)
    f_pc_s0 = get("photocatalyst", pc_ids, "S0", "classical_fraction")
    w_pc_s0 = get("photocatalyst", pc_ids, "S0", "quantum_frequency")
    f_pc_t1 = get("photocatalyst", pc_ids, "T1", "classical_fraction")
    if source == "dft":
        f_q_t1 = get("quencher", q_ids, "T1", "classical_fraction")
        w_q_t1 = get("quencher", q_ids, "T1", "quantum_frequency")
        f_q_s0 = get("quencher", q_ids, "S0", "classical_fraction")
    else:
        f_q_t1 = np.full(len(df), medians.loc[("quencher", "T1"), "classical_fraction"])
        w_q_t1 = np.full(len(df), medians.loc[("quencher", "T1"), "quantum_frequency"])
        f_q_s0 = np.full(len(df), medians.loc[("quencher", "S0"), "classical_fraction"])
    lv_pc, lv_q = (1 - f_pc_s0) * pc_s0_t1, (1 - f_q_t1) * q_t1_s0
    lam_v = lv_pc + lv_q
    hw = (lv_pc * w_pc_s0 + lv_q * w_q_t1) / np.where(lam_v > 0, lam_v, 1.0) * KCAL_PER_CM
    return {"dG": dG, "forward": forward, "reverse": reverse,
            "geometric": np.sqrt(forward * reverse),
            "ls_f": np.maximum(f_pc_t1 * pc_t1_s0 + f_q_s0 * q_s0_t1, LAMBDA_FLOOR),
            "ls_r": np.maximum(reverse - lam_v, LAMBDA_FLOOR),
            "lam_v": lam_v, "hw": hw,
            "barrier": theory.marcus_barrier_asymmetric(dG, forward, reverse)}


# ------------------------------------------------------------------ rates
def log_fc(dG, ls_f, ls_r, lam_v, hw, line_shape, n_channels=N_CHANNELS):
    """ln F, the Franck-Condon factor that replaces exp(-barrier/RT)."""
    dG = np.asarray(dG, float)
    hw = np.broadcast_to(np.asarray(hw, float), dG.shape)
    S = np.where(hw > 0, lam_v / np.maximum(hw, 1e-12), 0.0)
    n = np.arange(n_channels)[:, None]
    log_weight = np.where(S > 0, -S + n * np.log(np.maximum(S, 1e-300)) - gammaln(n + 1),
                          np.where(n == 0, 0.0, -np.inf))
    E = dG[None, :] + n * hw[None, :]
    if line_shape == "symmetric":
        B = (E + ls_r[None, :]) ** 2 / (4 * ls_r[None, :])
    elif line_shape == "asymmetric":
        B = theory.marcus_barrier_asymmetric(E, np.broadcast_to(ls_f, E.shape),
                                             np.broadcast_to(ls_r, E.shape))
    else:
        raise ValueError(f"unknown line shape {line_shape!r}")
    return logsumexp(log_weight - B / RT, axis=0)


def log10_rate(dG, ln_fc, escape_over_nu):
    """log10 of k_d / (1 + exp(dG/RT) + (k_-d/nu) / F)."""
    back = np.exp(np.minimum(np.asarray(dG, float) / RT, 500.0))
    barrier = escape_over_nu * np.exp(np.minimum(-np.asarray(ln_fc, float), 700.0))
    return np.log10(theory.K_DIFF / (1.0 + back + barrier))


def gated(dG, score):
    return np.where(dG <= 0, score, score - 1e3)


# ------------------------------------------------------------------ criteria
def criteria(df, inp, ratios, t_pc):
    """{(criterion, variant, source): predicted} and {(line shape, prefactor, source): log10 k}
    for the parameter-free criteria."""
    log_threshold = np.log10(K_SV_CRIT) - np.log10(pairs.tau0_seconds(df))
    predicted, rates = {}, {}
    for source, I in inp.items():
        dG = I["dG"]
        predicted[("adiabatic", "", source)] = dG <= 0
        for prefactor in PREFACTORS:
            predicted[("two_gate", prefactor, source)] = (dG <= 0) & (I["barrier"] <= t_pc[prefactor])
        for criterion, shape in LINE_SHAPES.items():
            for mode, fixed in QUANTUM_MODES.items():
                hw = I["hw"] if fixed is None else np.full(len(df), fixed * KCAL_PER_CM)
                ln_fc = log_fc(dG, I["ls_f"], I["ls_r"], I["lam_v"], hw, shape)
                for prefactor in PREFACTORS:
                    log_k = log10_rate(dG, ln_fc, ratios[prefactor])
                    if fixed is None:
                        rates[(shape, prefactor, source)] = log_k
                    margin = log_k - log_threshold
                    for decision in DECISIONS:
                        if decision == "rate":
                            p = margin >= 0
                        elif decision == "rate and dG <= 0":
                            p = (margin >= 0) & (dG <= 0)
                        else:
                            p = (dG <= 0) & (-RT * ln_fc <= t_pc[prefactor])
                        predicted[(criterion, f"{mode}, {prefactor}, {decision}", source)] = p
    return predicted, rates


def tuned(df, inp):
    """{(criterion, variant, source): predicted} of the tuned variant and the chosen lambda_s."""
    y = df[pairs.QUENCHES].to_numpy(bool)
    log_threshold = np.log10(K_SV_CRIT) - np.log10(pairs.tau0_seconds(df))
    matrices = {}
    for source, I in inp.items():
        rows = []
        for value in TUNED_GRID:
            ls = np.minimum(value, I["geometric"])
            ln_fc = log_fc(I["dG"], ls, ls, I["geometric"] - ls,
                           np.full(len(df), TUNED_MODE * KCAL_PER_CM), "symmetric")
            rows.append(log10_rate(I["dG"], ln_fc, ESCAPE_OVER_NU_LOW))
        matrices[source] = (np.vstack(rows) >= log_threshold) & (I["dG"] <= 0)[None, :]

    variant = f"{TUNED_MODE:.0f} cm-1, k_-d/nu = 0.25, rate and dG <= 0"
    predicted, chosen = {}, []
    k = scoring.best_setting(y, matrices["dft"])
    chosen.append({"evaluation": "all pairs", "held_out": "", "lambda_s": TUNED_GRID[k]})
    for source in inp:
        predicted[("mlj_tuned", f"{variant}, all pairs", source)] = matrices[source][k]
    for scheme, column in HELD_OUT.items():
        groups = df[column].to_numpy()
        held = {s: np.zeros(len(df), bool) for s in inp}
        for group in pd.unique(groups):
            inside = groups == group
            k = scoring.best_setting(y[~inside], matrices["dft"][:, ~inside])
            chosen.append({"evaluation": scheme, "held_out": group, "lambda_s": TUNED_GRID[k]})
            for s in inp:
                held[s][inside] = matrices[s][k, inside]
        for source in inp:
            predicted[("mlj_tuned", f"{variant}, {scheme}", source)] = held[source]
    return predicted, pd.DataFrame(chosen)


def bootstrap_dmcc(df, predicted):
    """dMCC of every row against two-gate (Eyring, same source), with 95% intervals."""
    y = df[pairs.QUENCHES].to_numpy(float)
    keys = list(predicted)
    M = np.vstack([predicted[k] for k in keys]).astype(float)

    def weighted_mcc(w):
        tp, fp = M @ (w * y), M @ (w * (1 - y))
        fn, tn = (w * y).sum() - tp, (w * (1 - y)).sum() - fp
        d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, (tp * tn - fp * fn) / d, 0.0)

    rng = np.random.default_rng(SEED)
    pcs, pc_index = np.unique(df[pairs.PC_NAME].to_numpy(), return_inverse=True)
    qs, q_index = np.unique(df[pairs.Q_NAME].to_numpy(), return_inverse=True)
    point = weighted_mcc(np.ones(len(df)))
    replicates = np.empty((BOOTSTRAP_REPLICATES, len(keys)))
    for r in range(BOOTSTRAP_REPLICATES):
        cp = np.bincount(rng.integers(0, len(pcs), len(pcs)), minlength=len(pcs))
        cq = np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs))
        replicates[r] = weighted_mcc((cp[pc_index] * cq[q_index]).astype(float))
    out = {}
    for i, key in enumerate(keys):
        j = keys.index(("two_gate", "Eyring", key[2]))
        d = replicates[:, i] - replicates[:, j]
        out[key] = {"dMCC": point[i] - point[j], "dMCC lo95": np.quantile(d, 0.025),
                    "dMCC hi95": np.quantile(d, 0.975)}
    return out


# ------------------------------------------------------------------ tables
def metrics_table(df, predicted, dmcc):
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for (criterion, variant, source), p in predicted.items():
        m = scoring.metrics(y, p)
        rows.append({"criterion": criterion, "variant": variant, "parameters": source,
                     "n_fitted": int(criterion == "mlj_tuned"),
                     **{c: m[c] for c in ("n", "TP", "FP", "TN", "FN", "MCC", "recall", "specificity")},
                     **dmcc[(criterion, variant, source)]})
    return pd.DataFrame(rows)


def classes_table(df, predicted, classes):
    y = df[pairs.QUENCHES].to_numpy(bool)
    rows = []
    for (criterion, variant, source), p in predicted.items():
        for cls in CLASS_ORDER:
            inside = classes == cls
            m = scoring.metrics(y[inside], p[inside])
            rows.append({"criterion": criterion, "variant": variant, "parameters": source,
                         "quencher_class": cls, **{c: m[c] for c in ("n", "TP", "FP", "TN", "FN")}})
    return pd.DataFrame(rows)


def parity_table(df, rates):
    k_q = df[pairs.K_Q].to_numpy(float)
    on = df[pairs.QUENCHES].to_numpy(bool) & (k_q > 0)
    measured = np.log10(k_q[on])
    rows = []
    for (shape, prefactor, source), log_k in rates.items():
        d = log_k[on] - measured
        rows.append({"line_shape": shape, "prefactor": prefactor, "parameters": source,
                     "quantum_mode": "projected", "n": int(on.sum()),
                     "median bias / decades": np.median(d),
                     "RMSE / decades": np.sqrt(np.mean(d ** 2)),
                     "Spearman rho": spearmanr(log_k[on], measured).statistic,
                     "within 1 decade": np.mean(np.abs(d) <= 1.0)})
    return pd.DataFrame(rows)


def barriers_table(I, classes):
    effective = -RT * log_fc(I["dG"], I["ls_f"], I["ls_r"], I["lam_v"], I["hw"], "asymmetric")
    rows = []
    for cls in (*CLASS_ORDER, "all"):
        inside = classes == cls if cls != "all" else np.ones(len(classes), bool)
        rows.append({"parameters": "dft", "quencher_class": cls, "n": int(inside.sum()),
                     "median classical barrier / kcal mol-1": np.median(I["barrier"][inside]),
                     "median MLJ effective barrier / kcal mol-1": np.median(effective[inside])})
    return pd.DataFrame(rows)


def summary_table(df, I, ratios, t_pc):
    quenching = df[pairs.QUENCHES].to_numpy(bool) & (df[pairs.K_Q].to_numpy(float) > 0)
    ln_fc = log_fc(I["dG"], I["ls_f"], I["ls_r"], I["lam_v"], I["hw"], "symmetric")
    speed_up = (ln_fc + I["barrier"] / RT) / np.log(10)
    shift = t_pc["Eyring"] - t_pc["k_-d/nu = 0.25"]
    rows = [
        ("median lambda_v / kcal mol-1", np.median(I["lam_v"]), 2),
        ("median hw / cm-1", np.median(I["hw"] / KCAL_PER_CM), 0),
        ("median S", np.median(I["lam_v"] / I["hw"]), 2),
        ("median lambda_s,r / kcal mol-1", np.median(I["ls_r"]), 2),
        ("median log10 of 0.25 over the Eyring k_-d/nu / decades",
         np.median(np.log10(ESCAPE_OVER_NU_LOW / ratios["Eyring"])), 3),
        ("median log10 of F (symmetric) over exp(-barrier/RT), quenching pairs / decades",
         np.median(speed_up[quenching]), 3),
        ("t_PC at k_-d/nu = 0.25, minimum / kcal mol-1", t_pc["k_-d/nu = 0.25"].min(), 2),
        ("t_PC at k_-d/nu = 0.25, maximum / kcal mol-1", t_pc["k_-d/nu = 0.25"].max(), 2),
        ("t_PC (Eyring) minus t_PC at 0.25, minimum / kcal mol-1", shift.min(), 2),
        ("t_PC (Eyring) minus t_PC at 0.25, maximum / kcal mol-1", shift.max(), 2),
    ]
    return pd.DataFrame([{"parameters": "dft", "quantity": q, "value": round(float(v), d)}
                         for q, v, d in rows])


def tuned_table(chosen):
    rows = []
    for evaluation, g in chosen.groupby("evaluation", sort=False):
        rows.append({"evaluation": evaluation, "n_choices": len(g),
                     "median lambda_s / kcal mol-1": g.lambda_s.median(),
                     "min lambda_s / kcal mol-1": g.lambda_s.min(),
                     "max lambda_s / kcal mol-1": g.lambda_s.max()})
    return pd.DataFrame(rows)


def rounded(table):
    """kcal/mol to 2 decimals, every other float to 3."""
    table = table.copy()
    for column in table.columns:
        if table[column].dtype.kind == "f":
            table[column] = table[column].round(2 if "kcal mol-1" in column else 3)
    return table


def main():
    df = pairs.scored(pairs.load())
    split, medians = molecule_split(pd.read_csv(MODES), four_point_lambdas(pd.read_csv(pairs.DATASET)))
    classes_by_name = pd.read_csv(CLASSES).set_index("id")["proposed_class"]
    classes = df[pairs.Q_NAME].map(classes_by_name).to_numpy()
    if pd.isna(classes).any():
        raise ValueError("quencher without a class in " + CLASSES.name)

    inp = {s: inputs(df, s, split, medians) for s in pairs.SOURCES}
    ratios = {"Eyring": theory.escape_over_frequency(pairs.contact_distances(df)),
              "k_-d/nu = 0.25": np.full(len(df), ESCAPE_OVER_NU_LOW)}
    t_pc = {p: theory.lifetime_barrier(pairs.tau0_seconds(df), K_SV_CRIT, r) for p, r in ratios.items()}

    predicted, rates = criteria(df, inp, ratios, t_pc)
    predicted_tuned, chosen = tuned(df, inp)
    predicted |= predicted_tuned
    dmcc = bootstrap_dmcc(df, predicted)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = {"metrics": metrics_table(df, predicted, dmcc),
              "classes": classes_table(df, predicted, classes),
              "rate_parity": parity_table(df, rates),
              "barriers": barriers_table(inp["dft"], classes),
              "summary": summary_table(df, inp["dft"], ratios, t_pc),
              "tuned": tuned_table(chosen)}
    for name, table in tables.items():
        rounded(table).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")

    m = tables["metrics"]
    free = m[m.criterion.str.startswith("mlj_") & (m.n_fitted == 0)]
    print(m[m.criterion.isin(["two_gate", "mlj_tuned"])][["criterion", "variant", "parameters", "MCC",
                                                         "recall", "dMCC"]].round(3).to_string(index=False))
    print(f"parameter-free MLJ criteria: MCC {free.MCC.min():.3f}-{free.MCC.max():.3f}")
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
