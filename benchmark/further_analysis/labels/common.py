"""
Loaders and paths shared by the Stern-Volmer label scripts in this folder.

Reads   ../../../data/experimental_data_raw/stern_volmer/PC*_QBatch*.csv   raw plate reads
        ../../../data/experimental_data_processed/stern_volmer_intensities.csv
        ../../../data/experimental_data_processed/stern_volmer_fits.csv
        ../../../data/experimental_data_processed/photocatalyst_properties_exp.xlsx
        ../../../data/exp_dataset.csv

Photocatalysts carry the old id of the raw files (`pc_old_id`) in the Stern-Volmer
tables; `pc` is the name PC<new id> used everywhere else.

Not a script; imported by the scripts of this folder.
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent.parent
DATA = BENCHMARK.parent / "data"
RAW = DATA / "experimental_data_raw" / "stern_volmer"
PROCESSED = DATA / "experimental_data_processed"
INTENSITIES = PROCESSED / "stern_volmer_intensities.csv"
FITS = PROCESSED / "stern_volmer_fits.csv"
PC_PROPERTIES = PROCESSED / "photocatalyst_properties_exp.xlsx"
EXP_DATASET = DATA / "exp_dataset.csv"
SV_FITS_SCRIPT = DATA / "experimental_data_scripts" / "stern_volmer_fits.py"

RESULTS = HERE.parent / "results" / "labels"
NOISE = RESULTS / "noise"          # s1-s5
ROBUST = RESULTS / "robust"        # robust_fits, robust_labels
LINEAR = RESULTS / "linear"        # linear_shapes, linear_variants, linear_downstream

DETECTOR_CEILING = 260000          # counts; reads at or above it are saturated
LEFT_OUT_PC = "PC22"               # in the data set, not in the benchmark
WELL = re.compile(r"[A-H](?:[1-9]|1[0-2])")

sys.path.insert(0, str(BENCHMARK))


def sv_fits_module():
    """data/experimental_data_scripts/stern_volmer_fits.py, imported unchanged (analyse,
    finalise, rung, ...)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("stern_volmer_fits", SV_FITS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_plate_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def raw_reads():
    """One row per (batch, well, read index): raw counts and read time."""
    rows = []
    for f in sorted(RAW.glob("PC*_QBatch*.csv")):
        times = None
        for line in read_plate_text(f).splitlines():
            parts = [p.strip() for p in line.split(";")]
            if parts[0].startswith("Time [s]"):
                times = [float(p.replace(",", ".")) for p in parts[1:] if p]
            if WELL.fullmatch(parts[0] or ""):
                values = [int(p) for p in parts[1:] if p]
                for k, v in enumerate(values):
                    rows.append((f.stem, parts[0], k, v,
                                 times[k] if times and k < len(times) else np.nan))
    return pd.DataFrame(rows, columns=["batch", "well", "read", "counts", "time_s"])


def intensities():
    """The well intensities with plate block (1: columns 1-6, 2: columns 7-12) and the
    position of the well in its series (0 = the [Q] = 0 well)."""
    sv = pd.read_csv(INTENSITIES)
    sv["row"] = sv.well.str[0]
    sv["col"] = sv.well.str[1:].astype(int)
    sv["block"] = np.where(sv.col <= 6, 1, 2)
    sv["pos"] = (sv.col - 1) % 6
    sv["plate"] = sv.batch
    return sv


def fits():
    return pd.read_csv(FITS)


def lifetimes():
    """pc_new_id, pc_old_id, compound, tau0 and its error (ns) per photocatalyst."""
    x = pd.read_excel(PC_PROPERTIES, sheet_name="Photocatalysts", header=1)
    x = x.rename(columns={"new_id": "pc_new_id", "old_id": "pc_old_id"})
    return x[["pc_new_id", "pc_old_id", "compound", "tau0 / ns", "tau0_err / ns"]].dropna(
        subset=["pc_new_id"])


def pc_names():
    """pc_old_id -> PC<new id>."""
    lt = lifetimes()
    return {int(o): f"PC{int(n)}" for o, n in zip(lt.pc_old_id, lt.pc_new_id)}


def write(df, path, gzip=False):
    """A result table with LF line endings; .csv.gz with a fixed gzip header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip:
        df.to_csv(path, index=False, lineterminator="\n",
                  compression={"method": "gzip", "mtime": 0})
    else:
        df.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path.relative_to(BENCHMARK)} ({len(df)} rows)")
