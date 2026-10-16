# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.2"]
# ///
"""
TCSPC decay traces -> excited-state lifetime tau0 of each photocatalyst.

Reads   ../experimental_data_raw/pc_tcspc/PC<old id>.txt   tab-separated, header block then
                                                         time (ns) vs counts, 2048 channels
        ../compound_identification.xlsx   (old -> new photocatalyst id)
Writes  ../experimental_data_raw/pc_tcspc/lifetimes_tcspc.csv   keyed by pc_new_id

    uv run data/experimental_data_scripts/lifetimes_tcspc.py

One trace per photocatalyst (MeCN, Ar, 10 uM, 3 mL). The background is the median of
the trace tail. The decay is fitted with a single exponential with 1/y weights; tau0
and its error are the fitted value and standard error. The instrument export
`PC<old id>.FL` is stored beside each trace.

The fit starts at the first channel after the pulse at or below 1000 counts. For PC4
it starts at 1500 ns on the instrument time axis instead (FIT_START_NS), so that its fast
early component is excluded and only the long-lived decay is fitted.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compound_identification import NEW_ID  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "experimental_data_raw" / "pc_tcspc"
OUT = RAW / "lifetimes_tcspc.csv"

FIT_OFFSET_POINTS = 3                 # channels skipped after the pulse peak before looking for the start
FIT_START_COUNT_THRESHOLD = 1000.0    # start at the first post-peak channel at or below this
PULSE_END_FRACTION = 0.5              # fallback: pulse is over below this fraction of the peak
STABILIZATION_POINTS = 4              # ... and once this many consecutive channels are non-rising
MIN_FIT_POINTS = 15                   # fewer usable channels than this and the trace is not fitted
MIN_SPAN_IN_LIFETIMES = 5             # the fit window must span this many lifetimes
# new id -> fit start time (ns, instrument axis) used instead of the count threshold
FIT_START_NS = {4: 1500.0}
TAU_GRID_POINTS = 320
TAU_REFINE_POINTS = 200
SIGFIGS = 6

COLUMNS = [
    "pc_new_id", "tau0_ns", "tau0_err_ns", "amplitude",
    "background_counts", "pulse_channel", "fit_channel_start", "fit_channel_end",
    "fit_start_ns", "fit_end_ns", "n_fit_points", "span_ns", "note",
]


def sig(x, n=SIGFIGS):
    if x is None or not np.isfinite(x):
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def read_trace(path):
    """time_ns, counts."""
    t, c = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        t.append(a)
        c.append(b)
    if not t:
        raise ValueError(f"no numeric data in {path.name}")
    return np.asarray(t, float), np.asarray(c, float)


def estimate_background(counts):
    """Median of the last 5% of the trace (at least 10 channels), negatives excluded."""
    finite = counts[np.isfinite(counts)]
    if finite.size == 0:
        return 0.0
    tail = finite[-max(10, int(round(0.05 * finite.size))):]
    tail = tail[tail >= 0]
    return float(np.median(tail)) if tail.size else 0.0


def detect_decay_start(signal, pulse_index):
    """First channel after the peak at or below FIT_START_COUNT_THRESHOLD.

    If none is found, the first channel where STABILIZATION_POINTS consecutive channels
    are below PULSE_END_FRACTION of the peak and not rising.
    """
    n = signal.size
    if n == 0:
        return 0
    peak = signal[pulse_index] if 0 <= pulse_index < n else np.nan
    if not np.isfinite(peak) or peak <= 0:
        return min(n - 1, max(0, pulse_index + FIT_OFFSET_POINTS))

    base = max(0, pulse_index + FIT_OFFSET_POINTS)
    for idx in range(base, n):
        v = signal[idx]
        if not np.isfinite(v) or v <= 0 or v > FIT_START_COUNT_THRESHOLD:
            continue
        if idx == base:
            return idx
        prev = signal[idx - 1]
        if np.isfinite(prev) and prev > FIT_START_COUNT_THRESHOLD:
            return idx

    threshold = peak * PULSE_END_FRACTION
    stab = max(1, STABILIZATION_POINTS)
    for idx in range(base, n - stab + 1):
        w = signal[idx:idx + stab]
        if np.all(np.isfinite(w)) and np.all(w <= threshold) and np.all(np.diff(w) <= 0):
            return idx
    return min(n - 1, base)


def fit_single_exponential(x, y):
    """y = a*exp(-(t-t0)/tau), fitted in linear space on a geometric tau grid with 1/y weights."""
    if x.size < 2:
        return None
    dt = x - float(x[0])
    span = float(np.ptp(dt))
    if not np.isfinite(span) or span <= 0:
        return None

    step = np.diff(x)
    step = step[np.isfinite(step) & (step > 0)]
    min_step = float(np.median(step)) if step.size else max(span / 100.0, 1e-6)
    tau_min = max(min_step * 0.5, 1e-6)
    tau_max = max(span * 5.0, tau_min * 20.0)

    w = 1.0 / np.clip(y, 1.0, None)

    def scan(grid, best):
        for tau in grid:
            g = np.exp(-dt / tau)
            denom = float(np.sum(w * g * g))
            if denom <= 0 or not np.isfinite(denom):
                continue
            a = float(np.sum(w * y * g) / denom)
            if not np.isfinite(a) or a <= 0:
                continue
            sse = float(np.sum(w * (y - a * g) ** 2))
            if sse < best[0]:
                best = (sse, float(tau), a)
        return best

    best = scan(np.geomspace(tau_min, tau_max, TAU_GRID_POINTS), (np.inf, None, None))
    if best[1] is None:
        return None
    best = scan(np.geomspace(max(best[1] / 3.0, tau_min), min(best[1] * 3.0, tau_max),
                             TAU_REFINE_POINTS), best)
    sse, tau, a = best

    stderr = None
    if x.size > 2:
        g = np.exp(-dt / tau)
        jac = np.column_stack((g, a * g * (dt / tau ** 2)))
        dof = x.size - 2
        sigma2 = sse / dof if dof > 0 else np.nan
        if np.isfinite(sigma2) and sigma2 >= 0:
            cov = np.linalg.pinv(jac.T @ (w[:, None] * jac)) * sigma2
            if cov.shape == (2, 2) and np.isfinite(cov[1, 1]) and cov[1, 1] >= 0:
                stderr = float(np.sqrt(cov[1, 1]))
    return a, tau, stderr


def analyse(time_ns, counts, fit_start_ns=None):
    background = estimate_background(counts)
    signal = counts - background
    pulse_index = int(np.nanargmax(counts)) if np.any(np.isfinite(counts)) else 0
    if fit_start_ns is None:
        start = detect_decay_start(signal, pulse_index)
    else:
        late = np.where(time_ns >= fit_start_ns)[0]
        if late.size == 0:
            return {"background": background, "pulse_channel": pulse_index, "n_fit_points": 0,
                    "note": f"not fitted: window ends at {time_ns[-1]:g} ns, "
                            f"before the fixed fit start at {fit_start_ns:g} ns"}
        start = int(late[0])

    finite_positive = np.isfinite(signal) & (signal > 0)
    peak = signal[pulse_index] if 0 <= pulse_index < signal.size else np.nan
    if not np.isfinite(peak) or peak <= 0:
        peak = np.nanmax(signal[finite_positive]) if np.any(finite_positive) else np.nan

    mask = np.zeros_like(signal, dtype=bool)
    if np.isfinite(peak):
        cand = np.where((np.arange(signal.size) >= start) & finite_positive & (signal <= peak))[0]
        if cand.size:
            mask[int(cand[0]):] = True
    mask &= signal >= 1.0          # skip channels within one count of background
    mask &= finite_positive
    if np.count_nonzero(mask) < MIN_FIT_POINTS:
        mask = np.zeros_like(signal, dtype=bool)
        mask[start:] = True
        mask &= finite_positive

    idx = np.where(mask)[0]
    result = {"background": background, "pulse_channel": pulse_index,
              "n_fit_points": int(idx.size)}
    if idx.size < MIN_FIT_POINTS:
        result["note"] = f"not fitted: only {idx.size} usable channels, minimum {MIN_FIT_POINTS}"
        return result

    x, y = time_ns[mask], signal[mask]
    fit = fit_single_exponential(x, y)
    if fit is None:
        result["note"] = "not fitted: exponential fit did not converge"
        return result

    a, tau, stderr = fit
    result.update({
        "amplitude": a, "tau_ns": tau, "tau_stderr_ns": stderr,
        "fit_channel_start": int(idx[0]), "fit_channel_end": int(idx[-1]),
        "fit_start_ns": float(x[0]), "fit_end_ns": float(x[-1]),
        "span_ns": float(x[-1] - x[0]),
        "note": "" if fit_start_ns is None else f"fit start fixed at {fit_start_ns:g} ns",
    })
    return result


def main():
    files = sorted(RAW.glob("*.txt"))
    if not files:
        sys.exit(f"no TCSPC traces found in {RAW}")
    stray = [f.name for f in files if not re.fullmatch(r"PC\d+", f.stem)]
    if stray:
        sys.exit(f"traces must be named PC<n>.txt: {stray}")
    unknown = [f.name for f in files if int(f.stem[2:]) not in NEW_ID]
    if unknown:
        sys.exit(f"traces of photocatalysts not in the data: {unknown}")
    files.sort(key=lambda f: NEW_ID[int(f.stem[2:])])

    rows, problems = [], []
    for f in files:
        pc_id = NEW_ID[int(f.stem[2:])]
        time_ns, counts = read_trace(f)
        r = analyse(time_ns, counts, FIT_START_NS.get(pc_id))
        tau, span = r.get("tau_ns"), r.get("span_ns")
        if tau is None:
            problems.append(f"PC{pc_id}: {r.get('note', 'not fitted')}")
        elif span < MIN_SPAN_IN_LIFETIMES * tau:
            problems.append(f"PC{pc_id}: the window spans {span / tau:.1f} lifetimes, "
                            f"fewer than {MIN_SPAN_IN_LIFETIMES}")
        rows.append({
            "pc_new_id": pc_id,
            "tau0_ns": sig(tau),
            "tau0_err_ns": sig(r.get("tau_stderr_ns")),
            "amplitude": sig(r.get("amplitude")),
            "background_counts": sig(r["background"]),
            "pulse_channel": r["pulse_channel"],
            "fit_channel_start": r.get("fit_channel_start", ""),
            "fit_channel_end": r.get("fit_channel_end", ""),
            "fit_start_ns": sig(r.get("fit_start_ns")),
            "fit_end_ns": sig(r.get("fit_end_ns")),
            "n_fit_points": r["n_fit_points"],
            "span_ns": sig(span),
            "note": r.get("note", ""),
        })
    if problems:
        sys.exit("every photocatalyst needs one usable trace:\n  " + "\n  ".join(problems))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} photocatalysts -> {OUT.name}")
    for r in rows:
        if r["note"]:
            print(f"  PC{r['pc_new_id']}: {r['note']}")


if __name__ == "__main__":
    main()
