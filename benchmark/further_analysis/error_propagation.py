# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Errors in the computed parameters, propagated into the two_gate classification and the
predicted rate constants.

Reads   ../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/error_propagation/monte_carlo.csv        one row per error scenario and
                                                         source: spread of the predicted
                                                         log10 k_q of the quenching pairs,
                                                         its RMSE and Spearman rho against
                                                         the measured k_q, and MCC and class
                                                         changes of two_gate and adiabatic
        results/error_propagation/sensitivity.csv        local sensitivity of log10 k_q to
                                                         the quencher E_T and lambda

    uv run benchmark/further_analysis/error_propagation.py

Rate law: k_q = k_d / (1 + exp(dG/RT) + (k_-d/nu) exp(B/RT)) (theory.rate), with B the
asymmetric Marcus barrier and k_-d/nu = (k_d/K_A)/(k_B T/h) at the contact distance
r_PC + r_Q of each pair. two_gate: dG_EnT <= 0 and B <= t_PC(tau0, r), K_SV,crit = 10 M-1,
as in classification.py; adiabatic: dG_EnT <= 0.

Monte Carlo: N_DRAWS draws per scenario from one generator seeded with SEED, the
scenarios in the order of SCENARIOS. An error is drawn once per
molecule and applied to all of its pairs. A quencher lambda with an added error is floored
at LAMBDA_MIN kcal/mol, a multiplicative lambda factor at LAMBDA_MIN.

Q E_T                 normal error with sigma in kcal/mol on the quencher E_T (dG_EnT)
Q and PC E_T          independent normal errors on the quencher and photocatalyst E_T
Q params +- ML uncertainty
                      E_T, lambda_S0_T1 and lambda_T1_S0 of each quencher + N(0, the model's
                      uncertainty of that value)
Q params + resampled ML-DFT differences
                      the per-quencher (E_T, lambda_S0_T1, lambda_T1_S0) differences ML - DFT,
                      resampled jointly with replacement; added to the DFT parameters and
                      subtracted from the ML parameters
lambda multiplicative every lambda of every photocatalyst and quencher x (1 + N(0, sigma)),
                      sigma relative

Per scenario, over the quenching pairs: the width of the 5-95 % interval of each pair's
log10 k_q over the draws (median and 5th/95th percentiles over the pairs), its SD, the
fraction of pairs wider than two decades, and the shift of the median draw. Per draw, over
the quenching pairs: RMSE and Spearman rho against the measured log10 k_q. Per draw, over
all scored pairs: MCC and the fraction of pairs whose class changes; and the fraction of
pairs that change class in at least half of the draws.

Sensitivity: central quantities of the derivative of log10 k_q, by a forward difference of
SENSITIVITY_STEP kcal/mol in dG_EnT (decades per kcal/mol) and of 1 % in both quencher
lambdas (decades per decade).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import classification  # noqa: E402
import pairs  # noqa: E402
import scoring  # noqa: E402
import theory  # noqa: E402

RESULTS = HERE / "results" / "error_propagation"

SEED = 20261009
N_DRAWS = 1000
LAMBDA_MIN = 0.1                # kcal/mol, or a factor for relative errors
SENSITIVITY_STEP = 0.01         # kcal/mol
SCENARIOS = ([("Q E_T", s) for s in (1.0, 2.0, 3.0)]
             + [("Q and PC E_T", s) for s in (2.0, 3.0)]
             + [("Q params +- ML uncertainty", np.nan),
                ("Q params + resampled ML-DFT differences", np.nan)]
             + [("lambda multiplicative", s) for s in (0.1, 0.2)])
SWEEP_SIGMAS = np.round(np.arange(0.25, 5.0001, 0.25), 2)     # kcal/mol
ML_TARGETS = ("E_T", "lambda_S0_T1", "lambda_T1_S0")


class Pairs:
    """The scored pairs, their unperturbed parameters and their molecule indices."""

    def __init__(self, df):
        self.df = df
        self.n = len(df)
        self.y = df[pairs.QUENCHES].to_numpy(bool)
        self.escape_over_nu = theory.escape_over_frequency(pairs.contact_distances(df))
        self.t_pc = theory.lifetime_barrier(pairs.tau0_seconds(df), classification.K_SV_CRIT,
                                            self.escape_over_nu)
        k_q = df[pairs.K_Q].to_numpy(float)
        self.measured = np.log10(np.where(k_q > 0, k_q, np.nan))
        quencher_ids, self.q_index = np.unique(df[pairs.Q_ID].to_numpy(), return_inverse=True)
        pc_names, self.pc_index = np.unique(df[pairs.PC_NAME].to_numpy(), return_inverse=True)
        self.n_q, self.n_pc = len(quencher_ids), len(pc_names)
        self.base = {s: self.parameters(s) for s in pairs.SOURCES}
        self.unperturbed = {s: self.evaluate(self.base[s]) for s in pairs.SOURCES}

        quenchers = df.groupby(pairs.Q_ID).first().loc[quencher_ids]
        self.ml_uncertainty = {t: quenchers[pairs.q_ml_uncertainty(t)].to_numpy(float)
                               for t in ML_TARGETS}
        self.ml_minus_dft = np.column_stack([
            quenchers[f"q_{t}_ml / kcal mol-1"].to_numpy(float)
            - quenchers[f"q_{t}_dft / kcal mol-1"].to_numpy(float) for t in ML_TARGETS])

    def parameters(self, source):
        pc_s0_t1, pc_t1_s0, q_s0_t1, q_t1_s0 = pairs.lambdas(self.df, source)
        return {"dG": pairs.driving_force(self.df, source), "pc_s0_t1": pc_s0_t1,
                "pc_t1_s0": pc_t1_s0, "q_s0_t1": q_s0_t1, "q_t1_s0": q_t1_s0}

    def evaluate(self, p):
        """(log10 k_q, two_gate prediction, adiabatic prediction), arrays of any leading shape."""
        forward, reverse = theory.pair_lambdas(p["pc_s0_t1"], p["pc_t1_s0"], p["q_s0_t1"], p["q_t1_s0"])
        barrier = theory.marcus_barrier_asymmetric(p["dG"], forward, reverse)
        log_k = np.log10(theory.rate(p["dG"], barrier, self.escape_over_nu))
        exergonic = p["dG"] <= 0
        return log_k, exergonic & (barrier <= self.t_pc), exergonic


def draw(rng, data: Pairs, scenario, source, sigma, n):
    """n perturbed parameter sets as (n x pairs) arrays."""
    p = {k: np.broadcast_to(v, (n, data.n)).copy() for k, v in data.base[source].items()}
    qi, pi = data.q_index, data.pc_index
    if scenario == "Q E_T":
        p["dG"] += rng.normal(0, sigma, (n, data.n_q))[:, qi]
    elif scenario == "Q and PC E_T":
        p["dG"] += (rng.normal(0, sigma, (n, data.n_q))[:, qi]
                    - rng.normal(0, sigma, (n, data.n_pc))[:, pi])
    elif scenario == "Q params +- ML uncertainty":
        u = data.ml_uncertainty
        p["dG"] += rng.normal(0, 1, (n, data.n_q))[:, qi] * u["E_T"][qi]
        p["q_s0_t1"] = np.maximum(p["q_s0_t1"] + rng.normal(0, 1, (n, data.n_q))[:, qi]
                                  * u["lambda_S0_T1"][qi], LAMBDA_MIN)
        p["q_t1_s0"] = np.maximum(p["q_t1_s0"] + rng.normal(0, 1, (n, data.n_q))[:, qi]
                                  * u["lambda_T1_S0"][qi], LAMBDA_MIN)
    elif scenario == "Q params + resampled ML-DFT differences":
        sign = 1.0 if source == "dft" else -1.0
        e = sign * data.ml_minus_dft[rng.integers(0, data.n_q, (n, data.n_q))]   # (n, quenchers, 3)
        p["dG"] += e[:, qi, 0]
        p["q_s0_t1"] = np.maximum(p["q_s0_t1"] + e[:, qi, 1], LAMBDA_MIN)
        p["q_t1_s0"] = np.maximum(p["q_t1_s0"] + e[:, qi, 2], LAMBDA_MIN)
    elif scenario == "lambda multiplicative":
        def factor(m, index):
            return np.maximum(1.0 + rng.normal(0, sigma, (n, m)), LAMBDA_MIN)[:, index]
        p["pc_s0_t1"] *= factor(data.n_pc, pi)
        p["pc_t1_s0"] *= factor(data.n_pc, pi)
        p["q_s0_t1"] *= factor(data.n_q, qi)
        p["q_t1_s0"] *= factor(data.n_q, qi)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return p


def run(rng, data: Pairs, scenario, source, sigma, n=None):
    n = N_DRAWS if n is None else n
    log_k, two_gate, adiabatic = data.evaluate(draw(rng, data, scenario, source, sigma, n))
    y, measured = data.y, data.measured[data.y]
    quenching = log_k[:, y]
    width = np.percentile(quenching, 95, axis=0) - np.percentile(quenching, 5, axis=0)
    shift = np.abs(np.median(quenching, axis=0) - data.unperturbed[source][0][y])
    rmse = np.sqrt(np.nanmean((quenching - measured) ** 2, axis=1))
    rho = np.array([stats.spearmanr(row, measured)[0] for row in quenching])
    out = {"scenario": scenario, "sigma": sigma, "start": source, "draws": n,
           "quenching pairs: median per-pair 5-95% width of log10 k_q / decades": np.median(width),
           "per-pair width 5% (over pairs) / decades": np.percentile(width, 5),
           "per-pair width 95% (over pairs) / decades": np.percentile(width, 95),
           "quenching pairs: median per-pair SD of log10 k_q / decades":
               np.median(quenching.std(axis=0)),
           "fraction quenching pairs with 5-95% width > 2 decades": float(np.mean(width > 2)),
           "quenching pairs: median |median draw - unperturbed| / decades": np.median(shift),
           "RMSE vs measured median / decades": np.median(rmse),
           "RMSE 5%": np.percentile(rmse, 5), "RMSE 95%": np.percentile(rmse, 95),
           "Spearman vs measured median": np.median(rho),
           "Spearman 5%": np.percentile(rho, 5), "Spearman 95%": np.percentile(rho, 95)}
    for label, predicted, reference in (("two-gate", two_gate, data.unperturbed[source][1]),
                                        ("adiabatic", adiabatic, data.unperturbed[source][2])):
        changed = predicted != reference
        per_draw = changed.mean(axis=1)
        mcc = scoring.mcc_rows(y, predicted)
        out |= {f"[{label}] MCC unperturbed": scoring.metrics(y, reference)["MCC"],
                f"[{label}] MCC median": np.median(mcc),
                f"[{label}] MCC 5%": np.percentile(mcc, 5),
                f"[{label}] MCC 95%": np.percentile(mcc, 95),
                f"[{label}] fraction pairs changing class per draw (median)": np.median(per_draw),
                f"[{label}] fraction changing class 5%": np.percentile(per_draw, 5),
                f"[{label}] fraction changing class 95%": np.percentile(per_draw, 95),
                f"[{label}] fraction pairs changing class in >=50% of draws":
                    float(np.mean(changed.mean(axis=0) >= 0.5))}
    return out


def monte_carlo(data: Pairs):
    """The scenario table, drawn in this order from one generator."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame([run(rng, data, scenario, source, sigma)
                         for scenario, sigma in SCENARIOS for source in pairs.SOURCES])


def sensitivity(data: Pairs):
    decade = theory.RT * np.log(10.0)
    rows = []
    for source in pairs.SOURCES:
        p = data.base[source]
        base = data.unperturbed[source][0]
        shifted = dict(p, dG=p["dG"] + SENSITIVITY_STEP)
        scaled = dict(p, q_s0_t1=p["q_s0_t1"] * 1.01, q_t1_s0=p["q_t1_s0"] * 1.01)
        d_energy = np.abs((data.evaluate(shifted)[0] - base) / SENSITIVITY_STEP)
        d_lambda = np.abs((data.evaluate(scaled)[0] - base) / np.log10(1.01))
        for label, inside in (("quenching", data.y), ("all", np.ones(data.n, bool))):
            rows.append({"parameters": source, "pairs": label,
                         "median |dlog k / dE_T(Q)| / dec per kcal": np.median(d_energy[inside]),
                         "q90 |dlog k / dE_T(Q)|": np.percentile(d_energy[inside], 90),
                         "max |dlog k / dE_T(Q)|": d_energy[inside].max(),
                         "theoretical max 1/(RT ln10)": 1.0 / decade,
                         "median |dlog k / dlog lambda_Q|": np.median(d_lambda[inside]),
                         "kcal per decade (RT ln10)": decade})
    return pd.DataFrame(rows)


def write(table, path):
    """kcal/mol and sigma to 2 decimals, everything else to 3."""
    table = table.copy()
    for column in table.columns:
        if table[column].dtype.kind == "f":
            table[column] = table[column].round(2 if ("kcal" in column and "per kcal" not in column)
                                                or column == "sigma" else 3)
    table.to_csv(path, index=False, lineterminator="\n")


def main():
    data = Pairs(pairs.scored(pairs.load()))
    scenarios = monte_carlo(data)
    sens = sensitivity(data)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write(scenarios, RESULTS / "monte_carlo.csv")
    write(sens, RESULTS / "sensitivity.csv")

    shown = ["scenario", "sigma", "start", "[two-gate] MCC median", "[two-gate] MCC 5%",
             "[two-gate] MCC 95%", "[two-gate] fraction pairs changing class per draw (median)",
             "quenching pairs: median per-pair 5-95% width of log10 k_q / decades"]
    print(scenarios[shown].round(3).rename(columns=lambda c: c[:30]).to_string(index=False))
    print()
    print(sens.round(3).to_string(index=False))
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
