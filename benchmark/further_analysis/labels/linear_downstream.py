# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Benchmark metrics under the labels of every linear-range variant of linear_variants.py.

Reads   results/labels/linear/pair_variants.csv.gz (linear_variants.py)
        ../../../data/exp_dataset_ent.csv (through pairs.py)
        ../../../data/computational_data_processed/photocatalyst_properties_calc.xlsx
        ../../../data/experimental_data_processed/quencher_properties_exp.xlsx
Writes  results/labels/linear/downstream_mcc.csv   one row per variant, label rule and
                                                   criterion: n, quenching pairs, MCC,
                                                   recall, specificity

    uv run benchmark/further_analysis/labels/linear_downstream.py

The pairs of a variant are those of the benchmark under that variant's labels: PC22, Q31
and Q38 left out, every input of every criterion present (pairs.INPUTS), and either not
quenching or with every single-electron-transfer channel with a measured quencher
potential more than 0.25 V uphill, the rule of build_exp_dataset_ent.py applied to the
variant's label. Pair properties are taken per molecule from exp_dataset_ent.csv, so pairs
that the data set does not carry (quenching under a variant, not under the data set) are
scored too. Criteria: adiabatic (DFT, ML), DvTE p = 0.008 normal quantile, two-gate (DFT,
ML) with the per-pair contact distance (theory.py, K_SV,crit = 10 M-1). Written with three
decimals.
"""

import sys

import numpy as np
import pandas as pd

import common  # puts benchmark/ on the path
import classification
import pairs
import scoring
import theory

sys.dont_write_bytecode = True
sys.path.insert(0, str(common.DATA))
from build_exp_dataset import load, number  # noqa: E402

SET_EXCLUSION_V = 0.25
ORDER = ["adiabatic dG<=0 (dft)", "two-gate asym t_PC Eyring (dft)",
         "adiabatic dG<=0 (ml)", "two-gate asym t_PC Eyring (ml)", "DvTE p=0.008 normal"]


def universe(V):
    """Pair properties of every series of the benchmark molecules, rebuilt per molecule."""
    ent = pd.read_csv(pairs.DATASET)
    pc_cols = ["pc_new_id", "pc_old_id", "pc_name", "tau0 / ns", pairs.PC_E_T, pairs.PC_LAMBDA_S0_T1,
               pairs.PC_LAMBDA_T1_S0, pairs.PC_RADIUS]
    q_cols = ["quencher_id", "quencher_name"] + [c for c in ent.columns if c.startswith("q_")]
    PCT = ent[pc_cols].drop_duplicates("pc_new_id").set_index("pc_new_id")
    QT = ent[q_cols].drop_duplicates("quencher_id").set_index("quencher_id")
    pc_calc = load(common.DATA / "computational_data_processed" / "photocatalyst_properties_calc.xlsx", "new_id")
    q_exp = load(common.PROCESSED / "quencher_properties_exp.xlsx", "quencher_id")

    U = V[["pc_new_id", "quencher_id"]].join(PCT, on="pc_new_id").join(QT, on="quencher_id")
    U[pairs.CONTACT] = pairs.contact_distance(U)

    def r2(x):
        return None if x is None else round(x, 2)

    ruled_out = []
    for pc, q in zip(U.pc_new_id, U.quencher_id):
        p, qe = pc_calc.get(int(pc)), q_exp.get(int(q))
        ox = r2(None if number(qe, "E_Ox_vs_SCE_exp / V") is None or number(p, "E_PCstar_PCminus / V") is None
                else number(qe, "E_Ox_vs_SCE_exp / V") - number(p, "E_PCstar_PCminus / V"))
        red = r2(None if number(p, "E_PCplus_PCstar / V") is None or number(qe, "E_Red_vs_SCE_exp / V") is None
                 else number(p, "E_PCplus_PCstar / V") - number(qe, "E_Red_vs_SCE_exp / V"))
        m = [v for v in (ox, red) if v is not None]
        ruled_out.append(bool(m) and all(v > SET_EXCLUSION_V for v in m))
    U["set_ruled_out"] = ruled_out
    for s in pairs.SOURCES:
        U[pairs.dg_ent(s)] = (U[f"q_E_T_{s} / kcal mol-1"] - U[pairs.PC_E_T]).round(2)
    U["inputs_ok"] = ~U[pairs.INPUTS].isna().any(axis=1)
    return U


def predictions(U):
    tau = U["tau0 / ns"].to_numpy(float) * 1e-9
    r_ij = U[pairs.CONTACT].to_numpy(float)
    pred = {}
    for s in pairs.SOURCES:
        dG = U[pairs.dg_ent(s)].to_numpy(float)
        lf, lr = theory.pair_lambdas(U[pairs.PC_LAMBDA_S0_T1].to_numpy(float), U[pairs.PC_LAMBDA_T1_S0].to_numpy(float),
                                     U[pairs.q_lambda_s0_t1(s)].to_numpy(float), U[pairs.q_lambda_t1_s0(s)].to_numpy(float))
        barrier = theory.marcus_barrier_asymmetric(dG, lf, lr)
        with np.errstate(invalid="ignore"):
            t_pc = theory.lifetime_barrier(tau, classification.K_SV_CRIT, theory.escape_over_frequency(r_ij))
        pred[f"adiabatic dG<=0 ({s})"] = dG <= 0
        pred[f"two-gate asym t_PC Eyring ({s})"] = (dG <= 0) & (barrier <= t_pc)
    pred["DvTE p=0.008 normal"] = (U[pairs.PC_E_T] - U[pairs.q_dvte("0.008", "normal")]).to_numpy(float) >= 0
    return {k: pred[k] for k in ORDER}


def main():
    V = pd.read_csv(common.LINEAR / "pair_variants.csv.gz")
    V = V[(V.pc_new_id != int(common.LEFT_OUT_PC[2:]))
          & ~V.quencher_id.isin([int(q[1:]) for q in pairs.LEFT_OUT_QUENCHERS])].reset_index(drop=True)
    U = universe(V)
    pred = predictions(U)

    # the labels of the data set give the benchmark's pairs and predictions
    base = (U.inputs_ok & ((V.quenches == 0) | U.set_ruled_out)).to_numpy()
    bench = pairs.scored(pairs.load())
    two_gate = classification.Setting("two_gate", barrier_model="marcus_asymmetric",
                                      k_sv_crit=classification.K_SV_CRIT, contact_distance=None)
    chk = bench.assign(**{f"b_{s}": classification.predict(bench, two_gate, s)[0] for s in pairs.SOURCES}).merge(
        U[base].assign(**{f"p_{s}": pred[f"two-gate asym t_PC Eyring ({s})"][base] for s in pairs.SOURCES}),
        on=["pc_new_id", "quencher_id"], suffixes=("", "_u"))
    assert len(chk) == len(bench) == base.sum()
    assert all((chk[f"b_{s}"] == chk[f"p_{s}"]).all() for s in pairs.SOURCES)

    rows = []
    for c in [c for c in V.columns if c.startswith("lab|")]:
        _, name, rule = c.split("|")
        y = V[c].to_numpy(bool)
        keep = (U.inputs_ok & (~V[c].astype(bool) | U.set_ruled_out)).to_numpy()
        for crit, p in pred.items():
            m = scoring.metrics(y[keep], p[keep])
            rows.append(dict(variant=name, label_rule=rule, criterion=crit, n=m["n"], n_quench=int(y[keep].sum()),
                             MCC=m["MCC"], recall=m["recall"], specificity=m["specificity"]))
    M = pd.DataFrame(rows)
    common.write(M.round(3), common.LINEAR / "downstream_mcc.csv")
    print(M.pivot_table(index=["variant", "label_rule"], columns="criterion", values="MCC").round(3).to_string())


if __name__ == "__main__":
    main()
