# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Computational cost of the quencher calculations: wall time, cores and CPU hours per job.

Reads   ../computational_data_raw/quencher_adiabatic_DFT/Q<n>/<state>/
        ../computational_data_raw/quencher_adiabatic_DvTE/Gaussian_Freq/gaussian_s0_q<n>/
        ../computational_data_raw/quencher_adiabatic_DvTE/Milo_MD/milo_q<n>/md_<k>/
        ../computational_data_raw/quencher_adiabatic_DvTE/quenchers_vertical_gaps.tar.xz
Writes  ../computational_data_processed/quencher_timings.csv
        ../computational_data_raw/quencher_adiabatic_DvTE/quencher_timings_vertical_sp.csv.gz

    uv run data/computational_data_scripts/quencher_timings.py

One row per job in quencher_timings.csv, except the vertical single points of the DvTE
trajectories, which are summed per quencher, trajectory and multiplicity. Their per-output
table is quencher_timings_vertical_sp.csv.gz.

    dft <state> <job>     ORCA, `<job>_output.out`. Wall time from `TOTAL RUN TIME`, summed
                          when a file holds several runs; cores from `%pal nprocs`.
    dvte opt_freq         Gaussian optimisation and frequencies, two linked jobs in
                          `g16.log`. Wall time is the sum of `Elapsed time`, program CPU
                          time the sum of `Job cpu time`; cores from `%nproc`.
    dvte md               Milo trajectory. Its Gaussian logs are not kept, so wall time and
                          cores come from the Slurm job report `milo_q<n>_md_<k>.out`.
    dvte sp singlet/triplet
                          ORCA single points `output_1.out` / `output_3.out` at every
                          sampled step (step_5 ... step_1000), read from the archive as a
                          stream. Only outputs with a `TOTAL RUN TIME` are summed.
    dvte sp failed        Steps where an output is missing (the Slurm job hit its time
                          limit before the outputs were copied back) or ended without a
                          run time (ORCA error termination). Their cost is the step's Slurm
                          wall time minus the ORCA-timed output of the same step, if any.

    cpu / h               wall / s x cores / 3600
    program_cpu / h       Gaussian's own `Job cpu time`

Where a job has both a program timing and a Slurm report (`output.dat`), the timing is the
program's and the Slurm wall time and cores are recorded next to it. One Slurm report per
DvTE step covers both ORCA runs of that step, so it is compared with their sum, and only
for steps where both have a run time. Slurm reports whole seconds and its wall time also
covers copying files to and from the node. A job is flagged when

    slurm wall - program wall  <  -SLURM_BELOW_S      Slurm shorter than the program
    slurm wall - program wall  >  SLURM_ABOVE_S       Slurm longer than copy overhead
    slurm cores  !=  program cores

The flag names which of these apply. MD jobs have no program timing (source = slurm) and
are not cross-checked. Expected but absent DFT and Gaussian jobs are rows without a timing,
counted in n_missing; Q10 and Q11 have no `red` jobs.
"""

import csv
import gzip
import io
import re
import statistics
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA))

from compound_identification import QUENCHERS  # noqa: E402

RAW = DATA / "computational_data_raw"
DFT = RAW / "quencher_adiabatic_DFT"
DVTE = RAW / "quencher_adiabatic_DvTE"
GAUSSIAN = DVTE / "Gaussian_Freq"
MILO = DVTE / "Milo_MD"
ARCHIVE = DVTE / "quenchers_vertical_gaps.tar.xz"
OUT = DATA / "computational_data_processed" / "quencher_timings.csv"
OUT_STEPS = DVTE / "quencher_timings_vertical_sp.csv.gz"

SIGFIGS = 6

# jobs expected in each DFT state folder, as in quencher_dft.py
DFT_JOBS = {
    "S0": ("opt", "sp", "sp_vert_spin_change", "sp_vert_ox", "sp_vert_red"),
    "T1": ("opt", "sp", "sp_vert_spin_change"),
    "ox": ("opt", "sp", "sp_vert_red"),
    "red": ("opt", "sp", "sp_vert_ox"),
}
KNOWN_MISSING = {(10, "red"), (11, "red")}

N_TRAJECTORIES = 25
SAMPLED_STEPS = tuple(range(5, 1001, 5))      # step_5 ... step_1000
MULTIPLICITY = {"1": "singlet", "3": "triplet"}

# Slurm vs program flag rule, in seconds
SLURM_BELOW_S = 2.0
SLURM_ABOVE_S = 30.0

COLUMNS = ["quencher_id", "quencher_name", "calculation", "trajectory",
           "n_jobs", "n_runs", "cores", "wall / s", "cpu / h", "source",
           "program_cpu / h", "slurm_wall / s", "slurm_cores", "slurm_state",
           "n_missing", "n_failed", "n_flagged", "flag", "note"]

STEP_COLUMNS = ["quencher_id", "trajectory", "step", "multiplicity", "cores", "n_runs",
                "wall / s", "cpu / h", "terminated_normally", "step_complete",
                "step_program_wall / s", "slurm_wall / s", "slurm_cores", "slurm_state",
                "flag"]

ORCA_RUN_TIME = re.compile(
    rb"TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds"
    rb"\s*(\d+)\s*msec")
ORCA_NORMAL = re.compile(rb"ORCA TERMINATED NORMALLY")
ORCA_NPROCS = re.compile(rb"%pal\s+nprocs\s+(\d+)", re.I)
ORCA_MPI = re.compile(rb"Program running with\s+(\d+)\s+parallel MPI-processes")

G16_TIME = r"(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes\s+([\d.]+)\s+seconds"
G16_CPU = re.compile(r"Job cpu time:\s*" + G16_TIME)
G16_ELAPSED = re.compile(r"Elapsed time:\s*" + G16_TIME)
G16_NPROC = re.compile(r"%nproc(?:shared)?\s*=\s*(\d+)", re.I)

MEMBER = re.compile(r"milo_q(\d+)/md_(\d+)/step_(\d+)/"
                    r"(output_[13]\.out|output\.dat|input_[13]\.inp|submit\.cmd)$")


def sig(x, n=SIGFIGS):
    """Round to six significant figures; None -> empty cell."""
    if x is None:
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def slurm_seconds(text):
    """Slurm duration `[D-]HH:MM:SS` (or MM:SS) -> seconds."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def parse_slurm(text):
    """Slurm job report -> {wall_s, cores, state, cpu_s, n_reports}, or None.

    Two layouts occur: the `JOB REPORT` block (`Job Wall-clock time`, `Nodes`,
    `Cores per node`, `State`) and the `SLURM JOB STATISTICS` block (`WallTime`,
    `Allocated CPUs`). The last report in the file is used. Without a `State` field, a
    slurmstepd cancellation line gives the state, e.g. CANCELLED DUE TO TIME LIMIT.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    n_reports = text.count("JOB REPORT") + text.count("SLURM JOB STATISTICS")
    if not n_reports:
        return None

    def last(pattern):
        found = re.findall(pattern, text, flags=re.M)
        return found[-1].strip() if found else None

    wall = last(r"^Job Wall-clock time:\s*(\S+)") or last(r"^WallTime:\s*(\S+)")
    cpu = last(r"^CPU Utilized:\s*(\S+)") or last(r"^CPU Time \(CPUTime\):\s*(\S+)")
    per_node = last(r"^Cores per node:\s*(\d+)")
    nodes = last(r"^Nodes:\s*(\d+)")
    allocated = last(r"^Allocated CPUs:\s*(\d+)")
    state = last(r"^State:\s*(.+)$")
    cancelled = re.findall(r"\*\*\* JOB \d+ ON \S+ CANCELLED(?: AT \S+)?(.*?) \*\*\*", text)
    if state is None and cancelled:
        state = ("CANCELLED " + cancelled[-1].strip()).strip()
    if per_node is not None:
        cores = int(per_node) * int(nodes or 1)
    elif allocated is not None:
        cores = int(allocated)
    else:
        cores = None
    return {"wall_s": slurm_seconds(wall) if wall else None,
            "cpu_s": slurm_seconds(cpu) if cpu else None,
            "cores": cores,
            "state": state,
            "n_reports": n_reports}


def parse_orca(raw, input_text=None):
    """ORCA output bytes -> {wall_s, n_runs, n_normal, cores}."""
    runs = ORCA_RUN_TIME.findall(raw)
    wall = sum(int(d) * 86400 + int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
               for d, h, m, s, ms in runs)
    nprocs = ORCA_NPROCS.findall(raw)
    if not nprocs and input_text is not None:
        nprocs = ORCA_NPROCS.findall(input_text)
    if not nprocs:
        nprocs = ORCA_MPI.findall(raw)
    return {"wall_s": wall if runs else None,
            "n_runs": len(runs),
            "n_normal": len(ORCA_NORMAL.findall(raw)),
            "cores": int(nprocs[-1]) if nprocs else None}


def orca_failed(parsed):
    """True when not every run in the file ended with ORCA TERMINATED NORMALLY."""
    return parsed["n_normal"] == 0 or parsed["n_normal"] < parsed["n_runs"]


def compare(program_wall, program_cores, slurm):
    """Flag text for a program timing against its Slurm report ('' when they agree)."""
    if slurm is None or program_wall is None:
        return ""
    reasons = []
    if slurm["wall_s"] is not None:
        diff = slurm["wall_s"] - program_wall
        if diff < -SLURM_BELOW_S:
            reasons.append("slurm wall shorter")
        elif diff > SLURM_ABOVE_S:
            reasons.append("slurm wall longer")
    if slurm["cores"] is not None and program_cores is not None \
            and slurm["cores"] != program_cores:
        reasons.append("cores differ")
    return "; ".join(reasons)


def row(qid, calculation, **values):
    """A quencher_timings.csv row with every column present."""
    out = {c: "" for c in COLUMNS}
    out.update({"quencher_id": qid, "quencher_name": f"Q{qid}", "calculation": calculation,
                "n_missing": 0, "n_failed": 0, "n_flagged": 0})
    out.update({k: v for k, v in values.items() if v is not None})
    return out


def dft_rows(ids):
    """One row per expected ORCA job in Q<n>/<state>/."""
    rows = []
    for qid in ids:
        for state, jobs in DFT_JOBS.items():
            for job in jobs:
                name = "opt_freq" if job == "opt" else job
                calculation = f"dft {state} {name}"
                out = DFT / f"Q{qid}" / state / f"{job}_output.out"
                if not out.exists():
                    note = "no red jobs" if (qid, state) in KNOWN_MISSING else "output missing"
                    rows.append(row(qid, calculation, n_jobs=0, n_missing=1, note=note))
                    continue
                inp = out.with_name(f"{job}_input.inp")
                parsed = parse_orca(out.read_bytes(),
                                    inp.read_bytes() if inp.exists() else None)
                failed = orca_failed(parsed)
                notes = []
                if parsed["n_runs"] > 1:
                    notes.append(f"{parsed['n_runs']} runs in one output, summed")
                if failed:
                    notes.append(f"{parsed['n_normal']} of {parsed['n_runs']} runs "
                                 f"terminated normally")
                cpu_h = (None if parsed["wall_s"] is None or parsed["cores"] is None
                         else parsed["wall_s"] * parsed["cores"] / 3600)
                rows.append(row(qid, calculation, n_jobs=1, n_runs=parsed["n_runs"],
                                cores=parsed["cores"], **{"wall / s": sig(parsed["wall_s"]),
                                                          "cpu / h": sig(cpu_h)},
                                source="orca", n_failed=int(failed),
                                note="; ".join(notes)))
    return rows


def gaussian_rows(ids):
    """One row per quencher for the Gaussian optimisation + frequency job."""
    rows = []
    for qid in ids:
        folder = GAUSSIAN / f"gaussian_s0_q{qid}"
        log = folder / "g16.log"
        if not log.exists():
            rows.append(row(qid, "dvte opt_freq", n_jobs=0, n_missing=1,
                            note="g16.log missing"))
            continue
        text = log.read_text(encoding="utf-8", errors="replace")

        def seconds(match):
            d, h, m, s = match
            return int(d) * 86400 + int(h) * 3600 + int(m) * 60 + float(s)

        elapsed = [seconds(m) for m in G16_ELAPSED.findall(text)]
        cpu = [seconds(m) for m in G16_CPU.findall(text)]
        nproc = G16_NPROC.findall(text)
        if not nproc and (folder / "g16.com").exists():
            nproc = G16_NPROC.findall((folder / "g16.com").read_text(errors="replace"))
        cores = int(nproc[0]) if nproc else None
        n_normal = text.count("Normal termination of Gaussian")
        failed = "Error termination" in text or n_normal < len(elapsed) or not elapsed
        wall = sum(elapsed) if elapsed else None

        slurm = None
        if (folder / "output.dat").exists():
            slurm = parse_slurm((folder / "output.dat").read_text(errors="replace"))
        flag = compare(wall, cores, slurm)
        notes = [f"{len(elapsed)} linked jobs"]
        if len(set(nproc)) > 1:
            notes.append(f"%nproc values {sorted(set(nproc))}")
        if failed:
            notes.append(f"{n_normal} normal terminations")
        rows.append(row(qid, "dvte opt_freq", n_jobs=1, n_runs=len(elapsed), cores=cores,
                        source="gaussian", n_failed=int(failed), n_flagged=int(bool(flag)),
                        flag=flag, note="; ".join(notes),
                        **{"wall / s": sig(wall),
                           "cpu / h": sig(None if wall is None or cores is None
                                          else wall * cores / 3600),
                           "program_cpu / h": sig(sum(cpu) / 3600 if cpu else None),
                           "slurm_wall / s": slurm and sig(slurm["wall_s"]),
                           "slurm_cores": slurm and slurm["cores"],
                           "slurm_state": slurm and slurm["state"]}))
    return rows


def md_rows(ids):
    """One row per Milo trajectory, timed by its Slurm report."""
    rows = []
    for qid in ids:
        for k in range(1, N_TRAJECTORIES + 1):
            folder = MILO / f"milo_q{qid}" / f"md_{k}"
            report = folder / f"milo_q{qid}_md_{k}.out"
            slurm = parse_slurm(report.read_text(errors="replace")) \
                if report.exists() else None
            if slurm is None or slurm["wall_s"] is None:
                rows.append(row(qid, "dvte md", trajectory=k, n_jobs=0, n_missing=1,
                                source="slurm", note="no Slurm report"))
                continue
            normal = False
            job_out = folder / "job.out"
            if job_out.exists():
                with job_out.open("rb") as fh:
                    fh.seek(max(0, job_out.stat().st_size - 2000))
                    normal = b"Normal termination." in fh.read()
            processors = None
            if (folder / "job.in").exists():
                m = re.search(r"^\s*processors\s+(\d+)",
                              (folder / "job.in").read_text(errors="replace"), re.M)
                processors = int(m.group(1)) if m else None
            notes = []
            if not normal:
                notes.append("job.out without Milo normal termination")
            if processors is not None and processors != slurm["cores"]:
                notes.append(f"milo processors {processors}")
            if slurm["n_reports"] > 1:
                notes.append(f"{slurm['n_reports']} Slurm reports, last used")
            rows.append(row(qid, "dvte md", trajectory=k, n_jobs=1, cores=slurm["cores"],
                            source="slurm", n_failed=int(not normal), note="; ".join(notes),
                            **{"wall / s": sig(slurm["wall_s"]),
                               "cpu / h": sig(slurm["wall_s"] * slurm["cores"] / 3600
                                              if slurm["cores"] else None),
                               "slurm_wall / s": sig(slurm["wall_s"]),
                               "slurm_cores": slurm["cores"],
                               "slurm_state": slurm["state"]}))
    return rows


def read_archive():
    """(qid, md, step) -> {name: parsed}, streamed from the vertical-gap archive."""
    steps = defaultdict(dict)
    inputs = {}
    seen = 0
    started = time.time()
    with tarfile.open(ARCHIVE, mode="r|xz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            m = MEMBER.search(member.name)
            if not m:
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            name = m.group(4)
            if name.startswith("output_"):
                steps[key][name] = parse_orca(raw)
            elif name == "output.dat":
                steps[key][name] = parse_slurm(raw)
            elif name.startswith("input_"):
                found = ORCA_NPROCS.findall(raw)
                inputs[key + (name,)] = int(found[-1]) if found else None
            else:
                found = re.findall(rb"--ntasks-per-node[ =](\d+)", raw)
                steps[key]["ntasks"] = int(found[-1]) if found else None
            seen += 1
            if seen % 50000 == 0:
                print(f"    {seen} files read, {len(steps)} steps, "
                      f"{time.time() - started:.0f} s elapsed", flush=True)
    for (qid, md, step, name), nprocs in inputs.items():
        out = steps[(qid, md, step)].get(name.replace("input_", "output_").replace(".inp",
                                                                                  ".out"))
        if out is not None and out["cores"] is None:
            out["cores"] = nprocs
    print(f"    {seen} files read in {time.time() - started:.0f} s", flush=True)
    return steps


def step_table(ids, steps):
    """Per-output rows of the vertical single points, including expected but absent ones."""
    rows = []
    for qid in ids:
        for md in range(1, N_TRAJECTORIES + 1):
            for step in SAMPLED_STEPS:
                found = steps.get((qid, md, step), {})
                outputs = {mult: found.get(f"output_{mult}.out") for mult in MULTIPLICITY}
                slurm = found.get("output.dat")
                if slurm is not None and slurm["cores"] is None:
                    slurm["cores"] = found.get("ntasks")      # from submit.cmd
                timed = [o for o in outputs.values()
                         if o is not None and o["wall_s"] is not None]
                complete = len(timed) == len(outputs)
                program_wall = sum(o["wall_s"] for o in timed)
                cores = {o["cores"] for o in timed}
                flag = compare(program_wall, cores.pop() if len(cores) == 1 else None,
                               slurm) if complete else ""
                for mult, label in MULTIPLICITY.items():
                    o = outputs[mult]
                    r = {c: "" for c in STEP_COLUMNS}
                    r.update({"quencher_id": qid, "trajectory": md, "step": step,
                              "multiplicity": label, "step_complete": int(complete),
                              "step_program_wall / s": sig(program_wall)})
                    if o is not None:
                        r.update({"cores": o["cores"], "n_runs": o["n_runs"],
                                  "wall / s": sig(o["wall_s"]),
                                  "cpu / h": sig(None if o["wall_s"] is None
                                                 or o["cores"] is None
                                                 else o["wall_s"] * o["cores"] / 3600),
                                  "terminated_normally": int(not orca_failed(o))})
                    if slurm is not None:
                        r.update({"slurm_wall / s": sig(slurm["wall_s"]),
                                  "slurm_cores": slurm["cores"],
                                  "slurm_state": slurm["state"] or "", "flag": flag})
                    rows.append(r)
    return rows


def step_rows(step_table_rows):
    """quencher_timings.csv rows for the vertical single points, per trajectory.

    `dvte sp singlet` and `dvte sp triplet` sum the ORCA-timed outputs. `dvte sp failed`
    sums, over the steps with an output that is missing or has no run time, the Slurm
    wall time of the step minus the ORCA-timed outputs of that step.
    """
    groups = defaultdict(list)
    for r in step_table_rows:
        groups[(r["quencher_id"], r["trajectory"])].append(r)
    rows = []
    for (qid, md), members in sorted(groups.items()):
        for label in MULTIPLICITY.values():
            outputs = [r for r in members if r["multiplicity"] == label]
            timed = [r for r in outputs if r["wall / s"] != ""]
            cores = Counter(r["cores"] for r in timed if r["cores"] != "")
            n_restarted = sum(1 for r in timed if r["n_runs"] > 1)
            notes = [f"{len(outputs)} sampled steps"]
            if len(cores) > 1:
                notes.append("cores " + ", ".join(f"{c}: {n} jobs"
                                                  for c, n in sorted(cores.items())))
            if n_restarted:
                notes.append(f"{n_restarted} outputs with several runs, summed")
            rows.append(row(
                qid, f"dvte sp {label}", trajectory=md, n_jobs=len(timed),
                n_runs=sum(r["n_runs"] for r in timed),
                cores=cores.most_common(1)[0][0] if cores else None, source="orca",
                n_missing=sum(1 for r in outputs if r["n_runs"] == ""),
                n_failed=sum(1 for r in outputs if r["terminated_normally"] == 0),
                n_flagged=sum(1 for r in outputs if r["flag"]),
                flag="; ".join(sorted({r["flag"] for r in outputs if r["flag"]})),
                note="; ".join(notes),
                **{"wall / s": sig(sum(r["wall / s"] for r in timed)),
                   "cpu / h": sig(sum(r["cpu / h"] for r in timed if r["cpu / h"] != ""))}))

        incomplete = {}
        for r in members:
            if not r["step_complete"]:
                incomplete.setdefault(r["step"], r)
        wall = cpu = 0.0
        states = Counter()
        no_report = 0
        for r in incomplete.values():
            if r["slurm_wall / s"] == "":
                no_report += 1
                continue
            rest = max(0.0, r["slurm_wall / s"] - r["step_program_wall / s"])
            wall += rest
            cpu += rest * r["slurm_cores"] / 3600
            if r["slurm_state"]:
                states[r["slurm_state"]] += 1
        cores = {r["slurm_cores"] for r in incomplete.values() if r["slurm_cores"] != ""}
        notes = [f"{len(incomplete)} steps with a missing or failed output"]
        notes += [f"{n} Slurm {s}" for s, n in sorted(states.items())]
        if no_report:
            notes.append(f"{no_report} without Slurm report, not timed")
        rows.append(row(qid, "dvte sp failed", trajectory=md,
                        n_jobs=len(incomplete) - no_report,
                        cores=cores.pop() if len(cores) == 1 else None, source="slurm",
                        n_missing=no_report, note="; ".join(notes),
                        **{"wall / s": sig(wall), "cpu / h": sig(cpu)}))
    return rows


GROUPS = [
    ("DFT opt_freq S0+T1", lambda c: c in ("dft S0 opt_freq", "dft T1 opt_freq")),
    ("DFT sp S0+T1", lambda c: c in ("dft S0 sp", "dft T1 sp")),
    ("DFT vert spin", lambda c: c.endswith("sp_vert_spin_change")),
    ("DFT redox", lambda c: c.startswith("dft")),
    ("DvTE opt_freq", lambda c: c == "dvte opt_freq"),
    ("DvTE md", lambda c: c == "dvte md"),
    ("DvTE vert sp", lambda c: c in ("dvte sp singlet", "dvte sp triplet")),
    ("DvTE sp failed", lambda c: c == "dvte sp failed"),
]


def kind(calculation):
    """Calculation without the DFT state or the multiplicity, for the job counts."""
    words = calculation.split()
    if words[0] == "dft":
        return f"dft {words[2]}"
    if calculation in ("dvte sp singlet", "dvte sp triplet"):
        return "dvte sp"
    return calculation


def summary(ids, rows):
    """Print CPU hours per quencher and group, then missing, failed and flagged counts."""
    table = defaultdict(float)
    for r in rows:
        if r["cpu / h"] == "":
            continue
        for name, test in GROUPS:
            if test(r["calculation"]):
                table[(r["quencher_id"], name)] += r["cpu / h"]
                break
    names = [g for g, _ in GROUPS]
    width = max(len(n) for n in names) + 2
    print("\nCPU hours per quencher (DFT redox: the ox and red jobs and the S0 vertical "
          "ox/red single points)")
    print(f"  {'':<7}" + "".join(f"{n:>{width}}" for n in names) + f"{'total':>12}")
    for qid in ids:
        values = [table[(qid, n)] for n in names]
        print(f"  {'Q' + str(qid):<7}" + "".join(f"{v:>{width}.1f}" for v in values)
              + f"{sum(values):>12.1f}")
    for label, fn in (("median", statistics.median), ("min", min), ("max", max)):
        values = [fn([table[(q, n)] for q in ids]) for n in names]
        totals = fn([sum(table[(q, n)] for n in names) for q in ids])
        print(f"  {label:<7}" + "".join(f"{v:>{width}.1f}" for v in values)
              + f"{totals:>12.1f}")
    print(f"  all quenchers: {sum(table.values()):.0f} CPU hours")

    print("\njobs per calculation type")
    counts = defaultdict(Counter)
    for r in rows:
        c = counts[kind(r["calculation"])]
        c["jobs"] += r["n_jobs"]
        c["missing"] += r["n_missing"]
        c["failed"] += r["n_failed"]
        c["flagged"] += r["n_flagged"]
        if r["source"] == "orca" and r["n_runs"] != "":
            c["restarts"] += r["n_runs"] - r["n_jobs"]
    for name, c in counts.items():
        print(f"  {name:<26} {c['jobs']:>7} timed, {c['missing']:>4} missing, "
              f"{c['failed']:>5} failed, {c['restarts']:>2} restarts, "
              f"{c['flagged']:>2} flagged")


def slurm_differences(label, pairs):
    """Print the distribution of slurm wall - program wall."""
    diffs = sorted(s - p for p, s in pairs)
    if not diffs:
        return
    q = statistics.quantiles(diffs, n=100, method="inclusive")
    rel = sorted((s - p) / p for p, s in pairs if p > 0)
    print(f"  {label}: n {len(diffs)}, slurm - program wall / s: min {diffs[0]:.1f}, "
          f"p1 {q[0]:.1f}, p50 {q[49]:.1f}, p99 {q[98]:.1f}, max {diffs[-1]:.1f}; "
          f"relative p50 {statistics.median(rel):.4f}, max {rel[-1]:.4f}")


def main():
    for path in (DFT, GAUSSIAN, MILO, ARCHIVE):
        if not path.exists():
            sys.exit(f"missing input: {path}")
    ids = sorted(QUENCHERS)

    print("DFT jobs ...", flush=True)
    rows = dft_rows(ids)
    print("Gaussian frequency jobs ...", flush=True)
    rows += gaussian_rows(ids)
    print("Milo trajectories ...", flush=True)
    rows += md_rows(ids)
    print(f"streaming {ARCHIVE.name} ({ARCHIVE.stat().st_size / 2**30:.1f} GB) ...",
          flush=True)
    steps = read_archive()
    unexpected = sorted({k for k in steps if k[0] not in QUENCHERS
                         or not 1 <= k[1] <= N_TRAJECTORIES or k[2] not in SAMPLED_STEPS})
    if unexpected:
        print(f"  {len(unexpected)} step folders outside the expected grid, not counted: "
              f"{unexpected[:5]}")
    table = step_table(ids, steps)
    rows += step_rows(table)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # mtime=0 and an empty filename make the gzip file identical between runs.
    with open(OUT_STEPS, "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STEP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    print(f"\n{len(rows)} rows -> {OUT.name}, {len(table)} single points -> {OUT_STEPS.name}")

    print("\nSlurm vs program wall time")
    slurm_differences("dvte opt_freq", [(r["wall / s"], r["slurm_wall / s"]) for r in rows
                                        if r["calculation"] == "dvte opt_freq"
                                        and r["wall / s"] != "" and r["slurm_wall / s"] != ""])
    slurm_differences("dvte sp steps, singlet + triplet",
                      [(r["step_program_wall / s"], r["slurm_wall / s"]) for r in table
                       if r["multiplicity"] == "singlet" and r["step_complete"]
                       and r["slurm_wall / s"] != ""])

    summary(ids, rows)


if __name__ == "__main__":
    main()
