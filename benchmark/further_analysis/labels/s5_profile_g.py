# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3", "scipy==1.18.1", "openpyxl==3.1.5"]
# ///
"""
Stage 5 of the label analysis: profile likelihood of the slope g of the censored fit.

Reads   results/labels/noise/pair_posteriors.csv.gz (s2_posteriors.py)
        results/labels/noise/ladder_power_curves.csv (s3_ladder_power.py)
        ../../../data/exp_dataset_ent.csv (through pairs.py)
Writes  results/labels/noise/profile_g.csv   one row per parameter source and g: the
                                             minimum of -ln L over a and sigma, and its
                                             distance dnll to the minimum over g

    uv run benchmark/further_analysis/labels/s5_profile_g.py

The likelihood is that of s4_downstream.py for the two-gate rate, mu = a + g * pred.
g = 0.20-0.64 in steps of 0.02; at each g, Nelder-Mead from ten starts over a and sigma.
"""

import numpy as np
import pandas as pd
from scipy import optimize

import common
import s4_downstream

G_GRID = np.round(np.arange(0.20, 0.66, 0.02), 2)


def profile(cl, mu):
    rows = []
    for g in G_GRID:
        best = None
        for a0 in (2, 3, 4, 5, 6):
            for s0 in (np.log(0.55), np.log(0.8)):
                o = optimize.minimize(lambda t: cl.nll(np.array([t[0], g, t[1]]), mu, True), [a0, s0],
                                      method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-6, fatol=1e-7))
                if best is None or o.fun < best.fun:
                    best = o
        rows.append({"g": g, "nll": best.fun, "a": best.x[0], "sigma": np.exp(best.x[1])})
    return rows


def main():
    cl, V = s4_downstream.censored_setup()
    rows = []
    for source in ("dft", "ml"):
        for row in profile(cl, V[f"log10 k_q pred {source}"].to_numpy()):
            rows.append({"model": source, **row})
            print(source, row["g"], round(row["nll"], 3), flush=True)
    P = pd.DataFrame(rows)
    P["dnll"] = P.nll - P.groupby("model").nll.transform("min")
    common.write(P.round(4), common.NOISE / "profile_g.csv")


if __name__ == "__main__":
    main()
