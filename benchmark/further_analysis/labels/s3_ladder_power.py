# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1"]
# ///
"""
Stage 3 of the label analysis: how often the label ladder of the data set calls a pair of
known K_SV quenching, under the noise of each photocatalyst.

Reads   results/labels/noise/noise_model_pc.csv (s1_noise_model.py)
        results/labels/noise/detection_limits_pc.csv (s2_posteriors.py)
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
        ../../../data/experimental_data_scripts/stern_volmer_fits.py (imported, unchanged)
Writes  results/labels/noise/ladder_power_curves.csv   one row per photocatalyst and true
                                                       K_SV: fraction of synthetic series
                                                       labelled quenching, and the fitted
                                                       K_SV

    uv run benchmark/further_analysis/labels/s3_ladder_power.py

For each photocatalyst, 300 synthetic series per true K_SV are drawn at its median I0 and
its concentrations: well noise of the stage-1 model with outliers, and a pseudo-slope beta
~ N(mu_block, tau^2) of the empirical null of stage 2 for a random plate half. Each series
runs through analyse, finalise and rung of stern_volmer_fits.py. True K_SV x [Q]max = 0,
0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32, 0.64.
"""

import numpy as np
import pandas as pd

import common

SEED = 7
N_SIM = 300
KT = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32, 0.64])   # K * x_max


def main():
    rng = np.random.default_rng(SEED)
    svf = common.sv_fits_module()
    sv = common.intensities()
    nm = pd.read_csv(common.NOISE / "noise_model_pc.csv").set_index("pc_old_id")
    dl = pd.read_csv(common.NOISE / "detection_limits_pc.csv").set_index("pc_old_id")
    b, pout, kout = nm["b_additive / counts"].iloc[0], nm.p_outlier_well.iloc[0], nm.k_outlier.iloc[0]
    I0 = sv[sv.pos == 0].groupby("pc_old_id").intensity_au.median()

    rows = []
    for pc in dl.index:
        x = np.sort(sv[sv.pc_old_id == pc].quencher_conc_M.unique())
        xm = x.max()
        xt = x / xm
        sm = dl.loc[pc, "sigma_m_used"]
        for kt in KT:
            K = kt / xm
            lab = np.zeros(N_SIM, bool)
            ks = np.zeros(N_SIM)
            for j in range(N_SIM):
                blk = rng.integers(1, 3)
                beta = rng.normal(dl.loc[pc, f"mu_block{blk}_span"], dl.loc[pc, "tau_span"])
                mean = np.log(I0[pc]) - np.log1p(K * x) - beta * xt
                s = np.sqrt(sm ** 2 + (b / np.exp(mean)) ** 2)
                s = s * np.where(rng.random(len(x)) < pout, kout, 1.0)
                I = np.exp(mean + rng.normal(0, s))
                idx, note, conc, ratio = svf.analyse(x, I)
                slope, intercept, r2, method = svf.finalise(idx, conc, ratio)
                lab[j] = bool(svf.rung(slope, r2))
                ks[j] = slope
            rows.append({"pc_old_id": pc, "pc": dl.loc[pc, "pc"], "K_true / M-1": K, "K_true_x_xmax": kt,
                         "P_label_1": lab.mean(), "K_SV_fit_median": np.median(ks),
                         "K_SV_fit_sd": ks.std(), "K_SV_fit_q05": np.quantile(ks, .05),
                         "K_SV_fit_q95": np.quantile(ks, .95)})
        print(dl.loc[pc, "pc"], flush=True)
    common.write(pd.DataFrame(rows), common.NOISE / "ladder_power_curves.csv")


if __name__ == "__main__":
    main()
