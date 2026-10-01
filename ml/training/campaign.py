"""
The validation campaign as a fixed list of steps, for running on a GPU machine.

    uv run --extra ml python -m ml.training.campaign check
    uv run --extra ml python -m ml.training.campaign gpus
    uv run --extra ml python -m ml.training.campaign status
    uv run --extra ml python -m ml.training.campaign run <step> --device cuda:0 --commit --push
    uv run --extra ml python -m ml.training.campaign sync

The steps, in dependency order:

    train_trial_55_random_<seed>   five runs: trial 55 on each random split
    evaluate_trial_0               metrics of Leon's trial-0 runs (already in the repository)
    evaluate_trial_55              metrics of the five trial-55 runs
    select                         the set with the lower mean validation loss
    train_oos_styryl               the selected set on the out-of-sample split
    train_full                     the selected set on all data
    evaluate_selected              the selected set's metrics again, now with the OOS split
    ensemble                       records the selected set's five random-split models as
                                   the ensemble and checks that they predict together

`run` refuses a step whose prerequisites are not done, a step that is already running, and
any run while files outside ml/results/ have been changed. A finished step is not run
again. Each run writes its output to ml/results/logs/<step>.log; with --commit it commits
exactly the step's result files and that log, nothing else, and with --push it pushes.

Written for this repository. It adds no training or evaluation logic of its own: every step
calls train.py, evaluate.py, select_hyperparameters.py or ml.ensemble.
"""

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ml.training.splits import RANDOM_SEEDS
from ml.training.test_predictions import RESULTS, run_dir

BASELINE = "trial_0"
CANDIDATE = "trial_55"

LOGS = RESULTS / "logs"
LOCKS = RESULTS / ".locks"
ENSEMBLE_RECORD = RESULTS / "ensemble_members.csv"
PREDICTION_FILES = ("model.pt", "test_predictions_graph.csv", "test_predictions_node.csv")
METRIC_FILES = ("metrics.csv", "metrics_random_splits.csv", "uncertainty_filtering.csv")


def selected():
    from ml.training.select_hyperparameters import selected_set
    return selected_set()


@dataclass
class Step:
    name: str
    needs: list = field(default_factory=list)
    outputs: Callable = None    # () -> list of paths
    done: Callable = None       # () -> bool
    run: Callable = None        # (device) -> None
    message: Callable = None    # () -> commit message


def _all_exist(paths):
    return all(p.exists() for p in paths)


def _train_random(seed):
    split = f"random_{seed}"
    outputs = lambda: [run_dir(CANDIDATE, split) / f for f in PREDICTION_FILES]

    def run(device):
        from ml.training.train import train
        train(CANDIDATE, split, device)

    return Step(f"train_{CANDIDATE}_{split}", [], outputs, lambda: _all_exist(outputs()), run,
                lambda: f"Train {CANDIDATE} on {split}")


def _evaluate(name, step_name, needs, done_extra=None):
    outputs = lambda: [run_dir(name() if callable(name) else name, "") / f for f in METRIC_FILES]

    def run(device):
        from ml.training.evaluate import evaluate
        evaluate(name() if callable(name) else name)

    def done():
        return _all_exist(outputs()) and (done_extra is None or done_extra())

    return Step(step_name, needs, outputs, done, run,
                lambda: f"Evaluate {name() if callable(name) else name}")


def _select():
    from ml.training.select_hyperparameters import RUNS, SELECTION
    outputs = lambda: [RUNS, SELECTION]

    def run(device):
        from ml.training.select_hyperparameters import main
        main()

    return Step("select", [f"evaluate_{BASELINE}", f"evaluate_{CANDIDATE}"], outputs,
                lambda: _all_exist(outputs()), run,
                lambda: f"Select the hyperparameter set by validation loss: {selected()}")


def _train_oos():
    outputs = lambda: [run_dir(selected(), "oos_styryl") / f for f in PREDICTION_FILES]

    def run(device):
        from ml.training.train import train
        train(selected(), "oos_styryl", device)

    return Step("train_oos_styryl", ["select"], outputs, lambda: _all_exist(outputs()), run,
                lambda: f"Train {selected()} on oos_styryl")


def _train_full():
    outputs = lambda: [run_dir(selected(), "full") / "model.pt"]

    def run(device):
        from ml.training.train import train_full
        train_full(selected(), device)

    return Step("train_full", ["select"], outputs, lambda: _all_exist(outputs()), run,
                lambda: f"Train {selected()} on all data")


def _oos_evaluated():
    metrics = run_dir(selected(), "") / "metrics.csv"
    if not metrics.exists():
        return False
    with open(metrics, newline="", encoding="utf-8") as f:
        return any(row["split"] == "oos_styryl" for row in csv.DictReader(f))


def _ensemble():
    outputs = lambda: [ENSEMBLE_RECORD]

    def done():
        if not ENSEMBLE_RECORD.exists():
            return False
        with open(ENSEMBLE_RECORD, newline="", encoding="utf-8") as f:
            return {row["hyperparameters"] for row in csv.DictReader(f)} == {selected()}

    def run(device):
        from ml.ensemble import Ensemble, default_checkpoints
        name = selected()
        paths = default_checkpoints(name)
        if [p.parent.name for p in paths] != [f"random_{s}" for s in RANDOM_SEEDS]:
            raise RuntimeError(f"{name}: expected the five random-split models, found {paths}")
        table = Ensemble(paths, device).predict(["c1ccccc1", "C=Cc1ccccc1"])
        print(table[["smiles", "triplet_energy_adiabatic", "triplet_energy_adiabatic_uncertainty",
                     "triplet_energy_adiabatic_model_spread", "n_models"]].to_string(index=False))
        root = repo_root()
        with open(ENSEMBLE_RECORD, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["hyperparameters", "split", "checkpoint", "md5"])
            for p in paths:
                digest = hashlib.md5(p.read_bytes()).hexdigest()
                writer.writerow([name, p.parent.name, p.relative_to(root).as_posix(), digest])

    return Step("ensemble", ["select"], outputs, done, run,
                lambda: f"Record the {selected()} random-split models as the ensemble")


STEPS = ([_train_random(seed) for seed in RANDOM_SEEDS]
         + [_evaluate(BASELINE, f"evaluate_{BASELINE}", []),
            _evaluate(CANDIDATE, f"evaluate_{CANDIDATE}",
                      [f"train_{CANDIDATE}_random_{s}" for s in RANDOM_SEEDS]),
            _select(), _train_oos(), _train_full(),
            _evaluate(selected, "evaluate_selected", ["train_oos_styryl"], _oos_evaluated),
            _ensemble()])
BY_NAME = {step.name: step for step in STEPS}


# -- git -----------------------------------------------------------------------------------

def repo_root():
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=RESULTS.parent,
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True,
                          check=check)


def changes_outside_results():
    """Tracked files changed anywhere but ml/results/ (a sparse checkout's absent paths aside)."""
    status = git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
    return [line for line in status if not line[3:].startswith("ml/results/")]


@contextlib.contextmanager
def file_lock(path, what):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"host": socket.gethostname(), "pid": os.getpid(), "what": what,
            "started": datetime.datetime.now().isoformat(timespec="seconds")}
    for _ in range(600):
        try:
            with open(path, "x", encoding="utf-8") as f:
                json.dump(info, f)
            break
        except FileExistsError:
            if _stale(path):
                path.unlink(missing_ok=True)
                continue
            if what == "git":
                time.sleep(1)
                continue
            raise SystemExit(f"{what} is already running: {path.read_text(encoding='utf-8')}")
    else:
        raise SystemExit(f"could not take {path}")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _stale(path):
    """A lock left by a process on this host that no longer exists."""
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if info.get("host") != socket.gethostname():
        return False
    try:
        os.kill(info["pid"], 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def commit(step):
    root = repo_root()
    paths = [p for p in step.outputs() if p.exists()]
    log = LOGS / f"{step.name}.log"
    if log.exists():
        paths.append(log)
    relative = [p.relative_to(root).as_posix() for p in paths]
    with file_lock(LOCKS / "git.lock", "git"):
        staged = git("diff", "--cached", "--name-only").stdout.split()
        if staged:
            raise SystemExit(f"refusing to commit: other files are staged: {staged}")
        git("add", "--", *relative)
        staged = git("diff", "--cached", "--name-only").stdout.split()
        if not staged:
            print("nothing new to commit")
            return
        if not set(staged) <= set(relative):
            git("reset", "-q")
            raise SystemExit(f"refusing to commit: unexpected staged files {staged}")
        git("commit", "-q", "-m", step.message())
        print(f"committed: {step.message()} ({len(staged)} files)")


def push():
    with file_lock(LOCKS / "git.lock", "git"):
        if git("push", "-q", "origin", "HEAD", check=False).returncode == 0:
            print("pushed")
            return
        pulled = git("pull", "-q", "--rebase", "--autostash", "origin", "main", check=False)
        if pulled.returncode != 0:
            raise SystemExit(f"push rejected and rebase failed:\n{pulled.stderr}")
        git("push", "-q", "origin", "HEAD")
        print("pushed after rebasing onto origin/main")


# -- steps ---------------------------------------------------------------------------------

def state(step):
    if step.done():
        return "done"
    lock = LOCKS / f"{step.name}.lock"
    if lock.exists() and not _stale(lock):
        return "running"
    if all(BY_NAME[n].done() for n in step.needs):
        return "ready"
    return "blocked"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(name, device, do_commit, do_push):
    step = BY_NAME.get(name)
    if step is None:
        raise SystemExit(f"unknown step {name}; see `status`")
    changed = changes_outside_results()
    if changed:
        raise SystemExit("refusing to run: files outside ml/results/ are changed:\n"
                         + "\n".join(changed))
    current = state(step)
    if current == "blocked":
        raise SystemExit(f"{name} is blocked; it needs " + ", ".join(step.needs))
    if current == "running":
        raise SystemExit(f"{name} is already running")
    if current == "ready":
        LOGS.mkdir(parents=True, exist_ok=True)
        with file_lock(LOCKS / f"{name}.lock", name), \
                open(LOGS / f"{name}.log", "a", encoding="utf-8") as log:
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            log.write(f"\n===== {name} on {socket.gethostname()} {device} {stamp}\n")
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), \
                    contextlib.redirect_stderr(Tee(sys.stderr, log)):
                step.run(device)
        if not step.done():
            raise SystemExit(f"{name} ended without all its outputs")
        print(f"{name}: done")
    else:
        print(f"{name} was already done")
    if do_commit:
        commit(step)
    if do_push:
        push()


def status():
    width = max(len(s.name) for s in STEPS)
    states = {}
    for step in STEPS:
        try:
            states[step.name] = state(step)
        except FileNotFoundError:
            states[step.name] = "blocked"
        print(f"{step.name:<{width}}  {states[step.name]}")
    ready = [n for n, s in states.items() if s == "ready"]
    if all(s == "done" for s in states.values()):
        print("\nall steps done")
    elif ready:
        print("\nready: " + " ".join(ready))


def free_gpus(max_memory_mib=1000):
    """Indices of GPUs that run no compute process and hold less than `max_memory_mib`."""
    query = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]
    gpus = [line.split(", ") for line in
            subprocess.run(query, capture_output=True, text=True, check=True).stdout.splitlines()]
    busy = set(subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              check=True).stdout.split())
    return [int(i) for i, uuid, memory, _ in gpus
            if uuid not in busy and int(memory) < max_memory_mib]


def gpus():
    query = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"]
    print(subprocess.run(query, capture_output=True, text=True, check=True).stdout.rstrip())
    free = free_gpus()
    print("free: " + (" ".join(f"cuda:{i}" for i in free) if free else "none"))


def check():
    import torch
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, "
          f"devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
    changed = changes_outside_results()
    print("files outside ml/results/ changed: " + (", ".join(changed) if changed else "none"))
    print(f"git: {git('log', '-1', '--format=%h %s').stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("check")
    commands.add_parser("gpus")
    commands.add_parser("sync")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("step")
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--commit", action="store_true")
    run_parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.command == "status":
        status()
    elif args.command == "check":
        check()
    elif args.command == "gpus":
        gpus()
    elif args.command == "sync":
        out = git("pull", "--ff-only", check=False)
        print(out.stdout or out.stderr)
        if out.returncode != 0:
            raise SystemExit("sync failed")
    else:
        run(args.step, args.device, args.commit, args.push)


if __name__ == "__main__":
    main()
