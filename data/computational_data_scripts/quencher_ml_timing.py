"""
Inference time of the graph MDN for the 48 quenchers, through `ml.prediction.predict`.

Reads   ../compound_identification.xlsx                (ids, SMILES)
        ../../ml/data/model.pt                         (the trained checkpoint)
Writes  ../computational_data_processed/quencher_timings_ml.csv

    uv sync --extra ml
    uv run --extra ml python data/computational_data_scripts/quencher_ml_timing.py

Runs on the CPU, the device `ml.prediction` uses. Wall time is `time.perf_counter`, CPU
time `time.process_time`, which sums over all threads of the process. The output starts
with `# key: value` lines recording the machine and versions; read it with
`pd.read_csv(path, comment="#")`.

    import            importing ml.prediction and its dependencies (torch, rdkit, ...)
    checkpoint load   torch.load of model.pt, timed inside the first call
    first call        the first predict call, one SMILES (Q1): checkpoint load, model
                      construction, lazy initialisation and the first prediction
    individual        one predict call per quencher with that SMILES alone, REPEATS times
    batch             one predict call with all 48 SMILES, REPEATS times, after one
                      untimed batch call

Individual and batch rows hold the median over the repeats, with the fastest and
slowest repeat. Batch rows also give the median divided by the number of molecules.
`time.process_time` advances in ticks (15.625 ms on Windows), so single calls get a
quantised CPU median; `cpu_mean / s` is the mean over the repeats.
Every call goes through the whole API: SMILES to graph with positional encodings, the
forward pass and the DataFrame of results. The model's console output is discarded.
"""

import contextlib
import csv
import io
import os
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
ROOT = DATA.parent
sys.path.insert(0, str(DATA))

from compound_identification import QUENCHERS  # noqa: E402

CHECKPOINT = ROOT / "ml" / "data" / "model.pt"
OUT = DATA / "computational_data_processed" / "quencher_timings_ml.csv"

REPEATS = 5
SIGFIGS = 6

COLUMNS = ["quencher_id", "quencher_name", "mode", "n_molecules", "n_heavy_atoms",
           "repeats", "wall / s", "wall_min / s", "wall_max / s",
           "cpu / s", "cpu_mean / s",
           "wall_per_molecule / s", "cpu_per_molecule / s"]


def sig(x, n=SIGFIGS):
    """Round to six significant figures; None -> empty cell."""
    if x is None:
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def cpu_name():
    """Processor name from the registry (Windows) or /proc/cpuinfo (Linux)."""
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def timed(fn, *args):
    """(wall seconds, process CPU seconds, result) of one call, its stdout discarded."""
    with contextlib.redirect_stdout(io.StringIO()):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        result = fn(*args)
        wall1, cpu1 = time.perf_counter(), time.process_time()
    return wall1 - wall0, cpu1 - cpu0, result


def repeated(fn, *args):
    """{wall, wall_min, wall_max, cpu, cpu_mean} over REPEATS calls, wall and cpu as medians."""
    walls, cpus = [], []
    for _ in range(REPEATS):
        wall, cpu, _ = timed(fn, *args)
        walls.append(wall)
        cpus.append(cpu)
    return {"wall": statistics.median(walls), "wall_min": min(walls),
            "wall_max": max(walls), "cpu": statistics.median(cpus),
            "cpu_mean": statistics.fmean(cpus)}


def out_row(**values):
    """A CSV row with every column present."""
    row = {c: "" for c in COLUMNS}
    row.update({k: (sig(v) if isinstance(v, float) else v) for k, v in values.items()})
    return row


def main():
    if not CHECKPOINT.exists():
        sys.exit(f"missing checkpoint: {CHECKPOINT}")
    ids = sorted(QUENCHERS)
    smiles = [QUENCHERS[q]["smiles"] for q in ids]

    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import torch
            from ml.prediction import predict
    except ImportError as exc:
        sys.exit(f"{exc}\n\nthis script needs the `ml` package and its environment:\n"
                 f"    uv sync --extra ml\n"
                 f"    uv run --extra ml python "
                 f"{Path(__file__).resolve().relative_to(ROOT).as_posix()}")
    import_wall, import_cpu = time.perf_counter() - wall0, time.process_time() - cpu0

    import rdkit
    import torch_geometric
    from rdkit import Chem

    heavy = {q: Chem.MolFromSmiles(QUENCHERS[q]["smiles"]).GetNumAtoms() for q in ids}

    # time torch.load inside the first call
    load_times = []
    torch_load = torch.load

    def timed_load(*args, **kwargs):
        w, c = time.perf_counter(), time.process_time()
        try:
            return torch_load(*args, **kwargs)
        finally:
            load_times.append((time.perf_counter() - w, time.process_time() - c))

    torch.load = timed_load
    try:
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
        first_wall, first_cpu, _ = timed(predict, [smiles[0]])
    finally:
        torch.load = torch_load
    if len(load_times) != 1:
        sys.exit(f"expected one torch.load in the first call, saw {len(load_times)}")
    load_wall, load_cpu = load_times[0]

    rows = [
        out_row(mode="import", repeats=1,
                **{"wall / s": import_wall, "cpu / s": import_cpu}),
        out_row(mode="checkpoint load", repeats=1,
                **{"wall / s": load_wall, "cpu / s": load_cpu}),
        out_row(quencher_id=ids[0], quencher_name=f"Q{ids[0]}", mode="first call",
                n_molecules=1, n_heavy_atoms=heavy[ids[0]], repeats=1,
                **{"wall / s": first_wall, "cpu / s": first_cpu}),
    ]
    print(f"import {import_wall:.2f} s, checkpoint load {load_wall:.2f} s, "
          f"first call {first_wall:.2f} s", flush=True)

    print(f"individual calls, {REPEATS} repeats each ...", flush=True)
    individual = {}
    for q, s in zip(ids, smiles):
        t = repeated(predict, [s])
        individual[q] = t
        rows.append(out_row(quencher_id=q, quencher_name=f"Q{q}", mode="individual",
                            n_molecules=1, n_heavy_atoms=heavy[q], repeats=REPEATS,
                            **{"wall / s": t["wall"], "wall_min / s": t["wall_min"],
                               "wall_max / s": t["wall_max"], "cpu / s": t["cpu"],
                               "cpu_mean / s": t["cpu_mean"],
                               "wall_per_molecule / s": t["wall"],
                               "cpu_per_molecule / s": t["cpu"]}))

    print(f"batch of {len(smiles)}, {REPEATS} repeats ...", flush=True)
    _, _, frame = timed(predict, smiles)          # untimed first batch call
    if len(frame) != len(smiles):
        sys.exit(f"batch returned {len(frame)} rows for {len(smiles)} SMILES")
    t = repeated(predict, smiles)
    rows.append(out_row(quencher_id="all", quencher_name="all", mode="batch",
                        n_molecules=len(smiles), n_heavy_atoms=sum(heavy.values()),
                        repeats=REPEATS,
                        **{"wall / s": t["wall"], "wall_min / s": t["wall_min"],
                           "wall_max / s": t["wall_max"], "cpu / s": t["cpu"],
                           "cpu_mean / s": t["cpu_mean"],
                           "wall_per_molecule / s": t["wall"] / len(smiles),
                           "cpu_per_molecule / s": t["cpu"] / len(smiles)}))

    header = {
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "rdkit": rdkit.__version__,
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="\n", encoding="utf-8") as fh:
        for key, value in header.items():
            fh.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    walls = [v["wall"] for v in individual.values()]
    cpus = [v["cpu_mean"] for v in individual.values()]
    print(f"\n{len(rows)} rows -> {OUT.name}")
    print(f"  {header['cpu']}, {header['torch_threads']} torch threads, torch {torch.__version__}")
    print(f"  individual, per quencher: wall median {statistics.median(walls) * 1000:.1f} ms "
          f"(range {min(walls) * 1000:.1f}-{max(walls) * 1000:.1f}), "
          f"CPU (mean over repeats) median {statistics.median(cpus) * 1000:.1f} ms")
    print(f"  batch of {len(smiles)}: wall {t['wall']:.3f} s, CPU {t['cpu']:.3f} s; "
          f"per molecule wall {t['wall'] / len(smiles) * 1000:.1f} ms, "
          f"CPU {t['cpu'] / len(smiles) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
