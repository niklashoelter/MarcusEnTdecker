# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Shape of the DvTE vertical-gap distributions and the harmonic picture of the two-gate barrier.

Reads   ../../data/computational_data_raw/quencher_adiabatic_DvTE/quencher_dvte_samples.csv.gz
        ../../data/computational_data_processed/quencher_geometry_S0_T1.csv
        ../../data/exp_dataset_ent.csv (through pairs.py)
        results/quencher_classes/quencher_classes.csv
Writes  results/anharmonicity/gap_shape.csv          per quencher: moments of the sampled gaps,
                                                     skewness with its trajectory-bootstrap 95%
                                                     interval, 0.008 quantiles and the S0 -> T1
                                                     C=C/C=N torsion
        results/anharmonicity/gap_shape_summary.csv  quenchers with a significantly negative or
                                                     positive skewness
        results/anharmonicity/skew_vs_error.csv      Spearman rho of the skewness with the signed
                                                     error of two-gate DFT, non-dissociative quenchers
        results/anharmonicity/tail_correction.csv    two-gate DFT with dG_EnT shifted by the
                                                     empirical minus the normal 0.008 quantile
        results/anharmonicity/crossing.csv           crossing point of the asymmetric barrier and the
                                                     quencher S0 energy there, by quencher group
        results/anharmonicity/torsional_curves.csv   two-gate DFT with torsional instead of harmonic
                                                     quencher curves for the twisted-T1 quenchers

    uv run benchmark/further_analysis/anharmonicity.py

Gap shape (all 48 quenchers). The gap of every sample is (E_T1 - E_S0) x 627.509
kcal/mol. Skewness is the bias-corrected sample skewness. Its 95% interval comes from
B_SKEW resamples of whole trajectories with replacement (the samples of one trajectory are
5 fs apart and correlated); after them, an i.i.d. resample of the samples is drawn from the
same random stream. A skewness is significant when the interval excludes 0. The tail shift
is the empirical 0.008 quantile minus the normal one, mu + Phi^-1(0.008) sd. A quencher
has a twisted T1 when its C=C or C=N torsion changes by at least TWIST_DEG between the S0
and T1 minima; dissociative quenchers are those of quencher_classes.py.

Scored pairs. Two-gate DFT is the benchmark criterion (r from DFT sizes). The signed error
of a quencher is (FN - FP) / n over its pairs. The tail correction replaces dG_EnT by
dG_EnT + tail shift of the quencher in both gates. The crossing point x (0 at the reactant,
1 at the product minimum) of the asymmetric barrier and the quencher S0 energy there,
lambda_S0-T1(Q) x^2 / RT, are given for the exergonic pairs (DFT). The torsional curves
replace the quencher part of the reactant curve, lambda_S0-T1(Q) x^2, by
lambda_S0-T1(Q) [w_S0 sin^2(pi x / 2) + (1 - w_S0) x^2] and that of the product curve,
lambda_T1-S0(Q) (1 - x)^2, by lambda_T1-S0(Q) [w_T1 cos^2(pi x / 2) + (1 - w_T1) (1 - x)^2],
for the twisted-T1 quenchers only; w_S0 = w_T1 = 0 is the benchmark barrier.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

DATA = HERE.parent.parent / "data"
SAMPLES = DATA / "computational_data_raw" / "quencher_adiabatic_DvTE" / "quencher_dvte_samples.csv.gz"
GEOMETRY = DATA / "computational_data_processed" / "quencher_geometry_S0_T1.csv"
CLASSES = HERE / "results" / "quencher_classes" / "quencher_classes.csv"
RESULTS = HERE / "results" / "anharmonicity"

H_TO_KCAL = 627.509
P_TAIL = 0.008
B_SKEW = 2000           # trajectory resamples for the skewness interval
N_IID = 400             # i.i.d. resamples drawn after them
SEED = 20261009
TWIST_DEG = 45.0
TORSION_WEIGHTS = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0))
TWO_GATE = classification.candidates("two_gate", "r from DFT sizes")[0]


# ----------------------------------------------------------------------------- gap shape
def skewness(x):
    return float(stats.skew(x, bias=False))


def gap_shape(samples, seed=SEED):
    """Per quencher: moments, skewness with its trajectory-bootstrap interval, 0.008 quantiles."""
    samples = samples.assign(gap=(samples.E_triplet_Eh - samples.E_singlet_Eh) * H_TO_KCAL)
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(P_TAIL)
    rows = []
    for qid, g in samples.groupby("quencher_id"):
        gap = g.gap.to_numpy()
        mu, sd = gap.mean(), gap.std(ddof=1)
        runs = [run.gap.to_numpy() for _, run in g.groupby("md_run")]
        boot = []
        for _ in range(B_SKEW):
            pick = rng.integers(0, len(runs), len(runs))
            boot.append(skewness(np.concatenate([runs[i] for i in pick])))
        rng.choice(gap, (N_IID, len(gap)))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        emp, normal = np.quantile(gap, P_TAIL), mu + z * sd
        rows.append({"quencher": f"Q{qid}", "quencher_id": qid, "n_samples": len(gap),
                     "n_trajectories": len(runs), "mu / kcal mol-1": mu, "sd / kcal mol-1": sd,
                     "skewness": skewness(gap), "skewness_lo95": lo, "skewness_hi95": hi,
                     "skewness_significant": "negative" if hi < 0 else "positive" if lo > 0 else "n.s.",
                     f"q{P_TAIL}_empirical / kcal mol-1": emp, f"q{P_TAIL}_normal / kcal mol-1": normal,
                     "tail_shift / kcal mol-1": emp - normal})
    return pd.DataFrame(rows)


def add_groups(shape, geometry, classes):
    """Torsion, twisted-T1 flag and dissociative flag of every quencher."""
    torsion = shape.quencher.map(geometry.set_index("id")["dtors_double_deg"])
    dissociative = shape.quencher.map(classes.set_index("id")["proposed_class"]) == "dissociative"
    return shape.assign(**{"dtors_double / deg": torsion, "twisted_T1": torsion >= TWIST_DEG,
                           "dissociative": dissociative})


def shape_summary(shape):
    return pd.DataFrame([
        {"quantity": "quenchers", "value": len(shape)},
        {"quantity": "skewness significantly negative",
         "value": int((shape.skewness_significant == "negative").sum())},
        {"quantity": "skewness significantly positive",
         "value": int((shape.skewness_significant == "positive").sum())}])


# ----------------------------------------------------------------------------- scored pairs
def lifetime_threshold(df):
    return theory.lifetime_barrier(pairs.tau0_seconds(df), TWO_GATE.k_sv_crit,
                                   theory.escape_over_frequency(pairs.contact_distances(df)))


def two_gate_shifted(df, shift):
    """Two-gate DFT with dG_EnT + shift in both gates."""
    dG = pairs.driving_force(df, "dft") + shift
    forward, reverse = theory.pair_lambdas(*pairs.lambdas(df, "dft"))
    barrier = theory.marcus_barrier_asymmetric(dG, forward, reverse)
    return (dG <= 0) & (lifetime_threshold(df) - barrier >= 0)


def signed_errors(df, predicted):
    y = df[pairs.QUENCHES].to_numpy(bool)
    t = pd.DataFrame({"quencher": df[pairs.Q_NAME], "n": 1, "FN": ~predicted & y, "FP": predicted & ~y})
    t = t.groupby("quencher", sort=False).sum()
    t["signed_error"] = (t.FN - t.FP) / t.n
    return t


def skew_vs_error(df, shape):
    predicted = classification.predict(df, TWO_GATE, "dft")[0]
    err = signed_errors(df, predicted)
    sh = shape.set_index("quencher")
    keep = [q for q in sh.index if q in err.index and not sh.loc[q, "dissociative"]]
    r = stats.spearmanr(sh.loc[keep, "skewness"], err.loc[keep, "signed_error"])
    return pd.DataFrame([{"criterion": "two_gate", "parameters": "dft",
                          "quenchers": "scored, non-dissociative", "n_quenchers": len(keep),
                          "spearman_rho": float(r.statistic), "p": float(r.pvalue)}])


def tail_correction(df, shape):
    y = df[pairs.QUENCHES].to_numpy(bool)
    reference = two_gate_shifted(df, 0.0)
    assert np.array_equal(reference, classification.predict(df, TWO_GATE, "dft")[0])
    shift = df[pairs.Q_NAME].map(shape.set_index("quencher")["tail_shift / kcal mol-1"]).to_numpy(float)
    corrected = two_gate_shifted(df, shift)
    rows = []
    for version, p in (("benchmark", reference), (f"dG_EnT + tail shift (p = {P_TAIL})", corrected)):
        rows.append({"criterion": "two_gate", "parameters": "dft", "version": version,
                     **scoring.metrics(y, p)})
    out = pd.DataFrame(rows)
    out["dMCC"] = out.MCC - out.MCC.iloc[0]
    out["changed_pairs"] = [0, int((corrected != reference).sum())]
    return out


def quencher_groups(df, shape):
    sh = shape.set_index("quencher")
    names = df[pairs.Q_NAME]
    diss = names.map(sh.dissociative).to_numpy(bool)
    twisted = names.map(sh.twisted_T1).to_numpy(bool) & ~diss
    return np.where(diss, "dissociative", np.where(twisted, "twisted T1", "planar T1"))


def crossing_point(dG, forward, reverse):
    """x of the crossing of the two asymmetric parabolas; symmetric crossing where none exists."""
    rad = forward * reverse + (reverse - forward) * dG
    cross = (rad >= 0) & ~np.isclose(forward, reverse)
    return np.where(cross, (np.sqrt(np.clip(rad, 0, None)) - forward) / np.where(cross, reverse - forward, 1.0),
                    (1 + dG / forward) / 2)


def crossing(df, shape):
    dG = pairs.driving_force(df, "dft")
    pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0 = pairs.lambdas(df, "dft")
    forward, reverse = theory.pair_lambdas(pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0)
    x = crossing_point(dG, forward, reverse)
    t = pd.DataFrame({"group": quencher_groups(df, shape), "x": x, "q_S0_kT": q_s0_t1 * x ** 2 / theory.RT})
    t = t[dG <= 0]
    return (t.groupby("group").agg(n_pairs=("x", "size"), x_cross_median=("x", "median"),
                                   q_S0_at_crossing_over_RT_median=("q_S0_kT", "median"))
            .reset_index().assign(parameters="dft", pairs="exergonic")
            [["parameters", "pairs", "group", "n_pairs", "x_cross_median", "q_S0_at_crossing_over_RT_median"]])


def barrier_torsional(dG, pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0, w_s0, w_t1):
    """(barrier, x) at the crossing of the mixed harmonic/torsional curves of one pair;
    0 where the product curve lies below at x = 0, NaN where the curves do not cross."""
    def reactant(x):
        return pc_t1_s0 * x ** 2 + q_s0_t1 * (w_s0 * np.sin(np.pi * x / 2) ** 2 + (1 - w_s0) * x ** 2)

    def product(x):
        return dG + pc_s0_t1 * (1 - x) ** 2 + q_t1_s0 * (w_t1 * np.cos(np.pi * x / 2) ** 2
                                                        + (1 - w_t1) * (1 - x) ** 2)

    def f(x):
        return product(x) - reactant(x)

    if f(0) <= 0:
        return 0.0, 0.0
    if f(1) > 0:
        return np.nan, np.nan
    x = brentq(f, 0, 1, xtol=1e-10)
    return reactant(x), x


def torsional_curves(df, shape):
    y = df[pairs.QUENCHES].to_numpy(bool)
    dG = pairs.driving_force(df, "dft")
    lam = pairs.lambdas(df, "dft")
    forward, reverse = theory.pair_lambdas(*lam)
    harmonic = theory.marcus_barrier_asymmetric(dG, forward, reverse)
    t_pc = lifetime_threshold(df)
    twisted = quencher_groups(df, shape) == "twisted T1"
    rows = []
    for w_s0, w_t1 in TORSION_WEIGHTS:
        barrier = harmonic.copy()
        for i in np.flatnonzero(twisted):
            barrier[i] = barrier_torsional(dG[i], *(a[i] for a in lam), w_s0, w_t1)[0]
        predicted = (dG <= 0) & (barrier <= t_pc)
        rows.append({"parameters": "dft", "w_S0": w_s0, "w_T1": w_t1, "n_twisted_pairs": int(twisted.sum()),
                     "dB_median / kcal mol-1": float(np.nanmedian((barrier - harmonic)[twisted])),
                     **scoring.metrics(y, predicted)})
    return pd.DataFrame(rows)


def rounded(table):
    """kcal/mol to 2 decimals, everything else to 3."""
    table = table.copy()
    for c in table.select_dtypes(float).columns:
        table[c] = table[c].round(2 if "kcal" in c or c.endswith("/ deg") else 3)
    return table


def main():
    shape = gap_shape(pd.read_csv(SAMPLES))
    shape = add_groups(shape, pd.read_csv(GEOMETRY), pd.read_csv(CLASSES))
    df = pairs.scored(pairs.load())
    tables = {"gap_shape": shape, "gap_shape_summary": shape_summary(shape),
              "skew_vs_error": skew_vs_error(df, shape), "tail_correction": tail_correction(df, shape),
              "crossing": crossing(df, shape), "torsional_curves": torsional_curves(df, shape)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        rounded(table).to_csv(RESULTS / f"{name}.csv", index=False, lineterminator="\n")
    for name in ("gap_shape_summary", "skew_vs_error", "tail_correction", "crossing", "torsional_curves"):
        print(rounded(tables[name]).to_string(index=False), "\n")
    print(f"-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
