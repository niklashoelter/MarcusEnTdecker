# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Quencher redox potentials from cyclic voltammograms, with literature values where no
voltammogram exists.

Reads   ../experimental_data_raw/quencher_cyclic_voltammetry/   (sample traces and blanks)
        ../experimental_data_raw/quencher_cyclic_voltammetry/quencher_literature_data.csv
Writes  cyclic_voltammetry_redox.csv    one row per measured sample
        cyclic_voltammetry_traces.csv   blank-corrected traces
        quencher_redox_potentials.csv   one row per quencher (48)

Usage: uv run cyclic_voltammetry_redox.py

Sources. A potential comes from the voltammogram wherever one exists. The literature file
fills in the rest, and its `source` column gives a DOI, `CV` for a value measured in this
lab without a deposited trace, or `estimated`. The literature file must not repeat a value
the voltammogram already gives.

Files. The three-digit sample number in the file name is the quencher id (`021_2.csv` is
the second run of Q21); the `i400sN1_1#...` line inside is the instrument's run counter,
not the sample. Each file is semicolon-separated with three header lines, then
potential in V and current in uA, for one cycle 0 -> +2.5 -> -2.5 -> 0 V at 0.05 V or
0.005 V steps. Each sample has a blank `<sample>_blank.csv` on the same potential axis,
which is subtracted point by point. Potentials are used as recorded, vs SCE, without
ferrocene correction. The export has no scan rate, so `scan_rate_V_s` is empty.

Reported value. On each branch the first resolvable wave is taken, which comes before
electrolyte breakdown. A wave is resolvable when the blank-corrected current rises by
MIN_WAVE_UA, or NOISE_K robust standard deviations if larger, above its trailing minimum,
at least EDGE_MARGIN_V inside the vertex. Onset (tangent at the steepest rise, extended
back to the baseline), peak, half-peak and half-wave potentials are written for each
branch. The reported value is the half-wave potential when a return peak lies within
REVERSIBLE_WINDOW_V of the forward peak, and the onset otherwise; `E_ox_basis` and
`E_red_basis` say which. A branch without a wave is reported as a bound at EDGE_MARGIN_V
inside the vertex (> +2.4 V or < -2.4 V), with the bound sign in its own column.

Repeats are averaged only when they agree to REPEAT_TOLERANCE_V; otherwise that value is
left empty and the note says why.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compound_identification import QUENCHERS  # noqa: E402

RAW = HERE.parent / "experimental_data_raw" / "quencher_cyclic_voltammetry"
LITERATURE = RAW / "quencher_literature_data.csv"
# Outputs sit next to the traces, so the sample scan in main() skips them.
OUT = RAW / "cyclic_voltammetry_redox.csv"
TRACES_OUT = RAW / "cyclic_voltammetry_traces.csv"
REDOX_OUT = RAW / "quencher_redox_potentials.csv"
WRITTEN = {OUT, TRACES_OUT, REDOX_OUT, LITERATURE}

SIGFIGS = 6

# Moving-average window (V) applied before differentiating.
SMOOTH_V = 0.02

# A wave starts where the smoothed current rises by max(MIN_WAVE_UA, NOISE_K * noise)
# above its minimum over the preceding TRAIL_V.
TRAIL_V = 0.50
MIN_WAVE_UA = 4.0
NOISE_K = 6.0

# A rise starting this close to the vertex is not counted as a wave. A branch without a
# wave is reported as a bound at this distance from the vertex.
EDGE_MARGIN_V = 0.10

# Baseline: median over BASE_WINDOW_V of sweep, ending BASE_GAP_V before the wave foot.
BASE_GAP_V = 0.05
BASE_WINDOW_V = 0.40

# The peak is the first maximum after which the current falls by this fraction of the
# wave height.
PEAK_DROP = 0.10

# A return peak within this distance of the forward peak marks a reversible couple.
REVERSIBLE_WINDOW_V = 0.30

# Repeats are averaged when they agree to this; otherwise the value is left empty.
REPEAT_TOLERANCE_V = 0.15

SAMPLE = re.compile(r"^(?P<sample>\d{3})(?:_(?P<repeat>\d+))?$")

COLUMNS = [
    "quencher_id", "sample", "n_repeats", "traces",
    "E_ox_V", "E_ox_bound", "E_ox_basis",
    "E_ox_onset_V", "E_ox_peak_V", "E_ox_half_peak_V", "i_ox_peak_uA",
    "E_ox_onset_spread_V", "ox_reversible", "E_ox_half_wave_V",
    "E_red_V", "E_red_bound", "E_red_basis",
    "E_red_onset_V", "E_red_peak_V", "E_red_half_peak_V", "i_red_peak_uA",
    "E_red_onset_spread_V", "red_reversible", "E_red_half_wave_V",
    "window_min_V", "window_max_V", "step_V", "n_points", "scan_rate_V_s",
    "note",
]

TRACE_COLUMNS = [
    "quencher_id", "sample", "repeat", "point", "segment",
    "potential_V", "current_uA", "blank_uA", "corrected_uA",
]

# Same columns as the literature file.
REDOX_COLUMNS = [
    "quencher_id", "E_Ox_vs_SCE_V", "E_Ox_bound", "E_Red_vs_SCE_V", "E_Red_bound",
    "source", "comment",
]

def sig(x, n=SIGFIGS):
    if x is None:
        return ""
    return 0.0 if x == 0 else float(f"{float(x):.{n}g}")


def read_export(path):
    """-> (potential, current). Three header lines, then two quoted, semicolon columns."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    if len(rows) < 4 or rows[2][:2] != ["Potential (V)", "Current (uA)"]:
        sys.exit(f"{path.name}: unexpected header, got {rows[2] if len(rows) > 2 else rows}")
    data = [r for r in rows[3:] if len(r) == 2 and r[0].strip()]
    if not data:
        sys.exit(f"{path.name}: no data rows")
    return (np.array([float(r[0]) for r in data]),
            np.array([float(r[1]) for r in data]))


def smooth(y, width):
    if width < 3:
        return y.astype(float).copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(padded, np.ones(width) / width, mode="valid")




def find_wave(u, y, step):
    """First resolvable wave on one branch.

    `u` increases along the sweep and `y` is the corrected current with the wave pointing
    up; the caller flips signs for the cathodic branch. Returns a dict with `detected`
    False when there is no wave inside the window.
    """
    n = len(u)
    if n < 5:
        return {"detected": False, "reason": "branch too short"}

    ys = smooth(y, max(1, round(SMOOTH_V / step)))
    slope = np.gradient(ys, u)
    noise = float(1.4826 * np.median(np.abs(y - ys)))
    threshold = max(MIN_WAVE_UA, NOISE_K * noise)

    trail = max(2, round(TRAIL_V / step))
    foot = None
    for k in range(1, n):
        if ys[k] - np.min(ys[max(0, k - trail):k + 1]) >= threshold:
            foot = k
            break

    if foot is None:
        return {"detected": False,
                "reason": f"no rise of {threshold:.1f} uA anywhere in the window"}
    if u[foot] > u[-1] - EDGE_MARGIN_V:
        return {"detected": False,
                "reason": f"the only rise begins {EDGE_MARGIN_V} V or less from the vertex"}

    gap = round(BASE_GAP_V / step)
    win = round(BASE_WINDOW_V / step)
    hi = max(1, foot - gap)
    lo = max(0, hi - win)
    base = float(np.median(ys[lo:hi])) if hi > lo else float(ys[0])

    peak = None
    best, best_k = ys[foot], foot
    for k in range(foot, n):
        if ys[k] > best:
            best, best_k = ys[k], k
        if best - base > 0 and (best - ys[k]) > PEAK_DROP * (best - base) and best_k > foot:
            peak = best_k
            break

    end = peak if peak is not None else n - 1
    search_lo = max(1, foot - round(0.30 / step))
    steep = search_lo + int(np.argmax(slope[search_lo:max(search_lo + 1, end + 1)]))
    onset = (float(u[steep] - (ys[steep] - base) / slope[steep])
             if slope[steep] > 1e-9 else None)

    height = float(ys[end] - base)
    half = None
    if peak is not None and height > 0:
        target = base + 0.5 * height
        for k in range(peak, foot - 1, -1):
            if ys[k] <= target:
                y0, y1 = ys[k], ys[k + 1]
                half = float(u[k] + (target - y0) / (y1 - y0) * (u[k + 1] - u[k])
                             if y1 != y0 else u[k])
                break

    return {
        "detected": True,
        "onset": onset,
        "peak": float(u[peak]) if peak is not None else None,
        "peak_current": float(ys[peak] - base) if peak is not None else None,
        "half": half,
        "reason": "" if peak is not None else "wave runs into the vertex, no peak",
    }


def return_peak(u, y, step, forward_peak):
    """Potential of the return peak within REVERSIBLE_WINDOW_V of the forward peak, or None.

    `u` and `y` are oriented so the return peak points up.
    """
    if forward_peak is None or len(u) < 5:
        return None
    ys = smooth(y, max(1, round(SMOOTH_V / step)))
    near = np.abs(u - forward_peak) <= REVERSIBLE_WINDOW_V
    if near.sum() < 5:
        return None
    idx = np.flatnonzero(near)
    k = int(idx[int(np.argmax(ys[idx]))])
    if k in (idx[0], idx[-1]):
        return None                      # maximum at the window edge: a shoulder
    # The current must rise into the peak and fall after it. The fall is measured over
    # the rest of the sweep, since a return peak decays slowly. Requiring the fall stops
    # the steep return climb after an irreversible wave from counting as a peak.
    if ys[k] - ys[idx[0]] < MIN_WAVE_UA or ys[k] - np.min(ys[k:]) < MIN_WAVE_UA:
        return None
    return float(u[k])


def analyze(path, blank_path):
    """Analyse one trace: oxidation and reduction branches, metadata and corrected trace."""
    potential, current = read_export(path)
    blank_potential, blank_current = read_export(blank_path)
    if len(potential) != len(blank_potential) or not np.allclose(potential, blank_potential):
        sys.exit(f"{path.name}: blank is on a different potential axis than the sample")

    corrected = current - blank_current
    step = float(abs(potential[1] - potential[0]))
    top = int(np.argmax(potential))
    bottom = int(np.argmin(potential))
    if not (0 < top < bottom < len(potential) - 1):
        sys.exit(f"{path.name}: not the expected 0 -> +max -> -max -> 0 cycle")

    # Anodic sweep, 0 -> +max. Oxidation shows here.
    ox = find_wave(potential[:top + 1], corrected[:top + 1], step)

    # Cathodic sweep, +max -> -max, negative half only. Reduction shows here. Both axes
    # are flipped so the sweep runs forward and the wave points up.
    cath_e, cath_i = potential[top:bottom + 1], corrected[top:bottom + 1]
    negative = cath_e <= 0.0
    red = find_wave(-cath_e[negative], -cath_i[negative], step)
    for key in ("onset", "peak", "half"):
        if red.get(key) is not None:
            red[key] = -red[key]

    # Reverse of the oxidation: the start of the cathodic sweep, coming back down.
    ox["reverse"] = return_peak(-cath_e[~negative], -cath_i[~negative], step,
                                -ox["peak"] if ox.get("peak") is not None else None)
    if ox["reverse"] is not None:
        ox["reverse"] = -ox["reverse"]
    # Reverse of the reduction: the return sweep, -max -> 0.
    red["reverse"] = return_peak(potential[bottom:], corrected[bottom:], step,
                                 red.get("peak"))

    meta = {
        "window_min_V": float(potential.min()),
        "window_max_V": float(potential.max()),
        "step_V": step,
        "n_points": len(potential),
    }
    segments = (["anodic"] * (top + 1) + ["cathodic"] * (bottom - top)
                + ["return"] * (len(potential) - bottom - 1))
    return ox, red, meta, (potential, current, blank_current, corrected, segments)


def combine(values, label, notes, tolerance=REPEAT_TOLERANCE_V, unit="V"):
    """Mean of the repeats when they agree, otherwise nothing and a note saying so."""
    usable = [v for v in values if v is not None]
    if not usable:
        return None, None
    spread = max(usable) - min(usable) if len(usable) > 1 else None
    if spread is not None and tolerance is not None and spread > tolerance:
        notes.append(f"repeats disagree on {label}: "
                     + ", ".join(f"{v:.3f}" for v in usable)
                     + f" (spread {spread:.2f} {unit}), value withheld")
        return None, spread
    return sum(usable) / len(usable), spread



def read_literature():
    """quencher_id -> row of quencher_literature_data.csv."""
    if not LITERATURE.exists():
        sys.exit(f"missing input: {LITERATURE}")
    with LITERATURE.open(encoding="utf-8", newline="") as fh:
        rows = {int(row["quencher_id"]): row for row in csv.DictReader(fh)}
    for qid, row in rows.items():
        for side, sign in (("Ox", ">"), ("Red", "<")):
            if row[f"E_{side}_bound"] not in ("", sign):
                sys.exit(f"Q{qid}: E_{side}_bound must be '{sign}' or blank, "
                         f"got {row[f'E_{side}_bound']!r}")
            if row[f"E_{side}_bound"] and not row[f"E_{side}_vs_SCE_V"]:
                sys.exit(f"Q{qid}: a bound on E_{side} without a value")
    return rows


def extract(samples):
    """One row per sample, plus the blank-corrected traces."""
    rows, trace_rows = [], []
    for sample in sorted(samples):
        quencher_id = int(sample)
        if quencher_id not in QUENCHERS:
            sys.exit(f"sample {sample} is not a quencher in compound_identification.xlsx")
        blank = RAW / f"{sample}_blank.csv"
        if not blank.exists():
            sys.exit(f"sample {sample} has no blank ({blank.name})")

        notes = []
        per_repeat, meta = [], None
        for repeat, path in sorted(samples[sample]):
            ox, red, meta, trace = analyze(path, blank)
            per_repeat.append((repeat, path.name, ox, red))
            potential, current, blank_current, corrected, segments = trace
            for k in range(len(potential)):
                trace_rows.append({
                    "quencher_id": quencher_id, "sample": sample, "repeat": repeat,
                    "point": k, "segment": segments[k],
                    "potential_V": sig(potential[k]), "current_uA": sig(current[k]),
                    "blank_uA": sig(blank_current[k]), "corrected_uA": sig(corrected[k]),
                })

        row = {"quencher_id": quencher_id, "sample": sample,
               "n_repeats": len(per_repeat),
               "traces": " | ".join(name for _, name, _, _ in per_repeat)}
        row.update(meta)
        row["scan_rate_V_s"] = ""      # not in the export

        for side, key, edge in (("ox", 2, meta["window_max_V"] - EDGE_MARGIN_V),
                                ("red", 3, meta["window_min_V"] + EDGE_MARGIN_V)):
            branches = [r[key] for r in per_repeat]
            found = [b for b in branches if b["detected"]]
            onset, spread = combine([b.get("onset") for b in found],
                                    f"the {side} onset", notes)
            peak, _ = combine([b.get("peak") for b in found], f"the {side} peak", notes)
            half, _ = combine([b.get("half") for b in found],
                              f"the {side} half-peak", notes)
            height, _ = combine([b.get("peak_current") for b in found],
                                f"the {side} peak current", notes,
                                tolerance=None, unit="uA")
            reverse, _ = combine([b.get("reverse") for b in found],
                                 f"the {side} return peak", notes)
            reversible = any(b.get("reverse") is not None for b in found)
            half_wave = (0.5 * (peak + reverse)
                         if reversible and peak is not None and reverse is not None
                         else None)

            row[f"E_{side}_onset_V"] = sig(onset)
            row[f"E_{side}_peak_V"] = sig(peak)
            row[f"E_{side}_half_peak_V"] = sig(half)
            row[f"i_{side}_peak_uA"] = sig(height)
            row[f"E_{side}_onset_spread_V"] = sig(spread)
            row[f"{side}_reversible"] = int(reversible)
            row[f"E_{side}_half_wave_V"] = sig(half_wave)

            # Reported value: half-wave for a reversible couple, onset for an
            # irreversible wave, a bound when there is no wave.
            if not found:
                reported, bound, basis = edge, (">" if side == "ox" else "<"), "no wave"
            else:
                reported = half_wave if half_wave is not None else onset
                bound = ""
                basis = ("" if reported is None else
                         "half-wave" if half_wave is not None else "onset")
            row[f"E_{side}_V"] = sig(reported)
            row[f"E_{side}_bound"] = bound
            row[f"E_{side}_basis"] = basis
            if half_wave is not None:
                notes.append(f"{side} couple is quasi-reversible: forward peak "
                             f"{peak:+.3f} V, return peak {reverse:+.3f} V, "
                             f"E_1/2 {half_wave:+.3f} V")

            if not found:
                whys = sorted({b.get("reason", "") for b in branches if b.get("reason")})
                notes.append(f"no {side} wave: " + "; ".join(whys))
            elif len(found) < len(branches):
                notes.append(f"{len(found)} of {len(branches)} repeats show a {side} wave")
            else:
                whys = sorted({b.get("reason", "") for b in found if b.get("reason")})
                if whys:
                    notes.append(f"{side}: " + "; ".join(whys))

        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows, trace_rows


def redox_table(cv_rows, literature):
    """Per quencher: the voltammogram's value where it gives one, the literature's otherwise."""
    cv = {row["quencher_id"]: row for row in cv_rows}
    missing = sorted(set(QUENCHERS) - set(cv) - set(literature))
    if missing:
        sys.exit(f"no voltammogram and no literature entry for quenchers {missing}")
    stray = sorted(set(literature) - set(QUENCHERS))
    if stray:
        sys.exit(f"literature entries for quenchers not in compound_identification.xlsx: {stray}")

    table = []
    for qid in sorted(QUENCHERS):
        lit = literature.get(qid, {})
        out = {"quencher_id": qid}
        sources, comments = [], []
        for side, cv_side, word in (("Ox", "ox", "oxidation"), ("Red", "red", "reduction")):
            value, bound = "", ""
            if qid in cv and cv[qid][f"E_{cv_side}_V"] != "":
                if lit.get(f"E_{side}_vs_SCE_V"):
                    sys.exit(f"Q{qid}: E_{side} is given by both the voltammogram and "
                             f"the literature file; remove it from the literature file")
                # Rounded to 1 mV; the traces have 5 or 50 mV steps.
                value = round(float(cv[qid][f"E_{cv_side}_V"]), 3)
                bound = cv[qid][f"E_{cv_side}_bound"]
                sources.append("CV")
                basis = cv[qid][f"E_{cv_side}_basis"]
                comments.append(f"E_{side}: no {word} wave inside the window" if bound
                                else f"E_{side}: {basis}")
            elif lit.get(f"E_{side}_vs_SCE_V"):
                value, bound = lit[f"E_{side}_vs_SCE_V"], lit[f"E_{side}_bound"]
                sources.append(lit["source"])
                if lit["comment"] and lit["comment"] not in comments:
                    comments.append(lit["comment"])
            out[f"E_{side}_vs_SCE_V"] = value
            out[f"E_{side}_bound"] = bound
        if not sources and lit:
            sources.append(lit["source"])
            if lit["comment"]:
                comments.append(lit["comment"])
        out["source"] = "; ".join(dict.fromkeys(sources))
        out["comment"] = "; ".join(comments)
        table.append(out)
    return table


def write(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{c: row.get(c, "") for c in columns} for row in rows])


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input directory: {RAW}")
    literature = read_literature()

    samples = {}
    for path in sorted(RAW.glob("*.csv")):
        if path.stem.endswith("_blank") or path in WRITTEN:
            continue
        match = SAMPLE.match(path.stem)
        if not match:
            sys.exit(f"cannot read a sample number from {path.name}")
        samples.setdefault(match["sample"], []).append((int(match["repeat"] or 1), path))
    if not samples:
        sys.exit(f"no CV traces found in {RAW}")

    rows, trace_rows = extract(samples)
    table = redox_table(rows, literature)
    write(OUT, COLUMNS, rows)
    write(TRACES_OUT, TRACE_COLUMNS, trace_rows)
    write(REDOX_OUT, REDOX_COLUMNS, table)
    report(rows, len(trace_rows), table)


def report(rows, n_points, table):
    print(f"{len(rows)} samples -> {OUT.name}")
    print(f"{n_points} trace points -> {TRACES_OUT.name}")
    print(f"{len(table)} quenchers -> {REDOX_OUT.name}")

    show = lambda row, side: ((row[f"E_{side}_bound"] or " ")
                              + (f"{float(row[f'E_{side}_V']):+.3f}"
                                 if row[f"E_{side}_V"] != "" else "     .")
                              + f" {row[f'E_{side}_basis']:<9}")
    print(f"\n  {'q':>3} | {'oxidation':<17} | {'reduction':<17} | note")
    for row in rows:
        print(f"  {row['quencher_id']:>3} | {show(row, 'ox')} | {show(row, 'red')} | "
              f"{row['note'][:70]}")

    kinds = {}
    for row in table:
        kind = ("CV (voltammogram)" if row["quencher_id"] in {r["quencher_id"] for r in rows}
                else "DOI" if row["source"].startswith("https://doi.org/")
                else f"{row['source']} (no voltammogram)")
        kinds.setdefault(kind, []).append(row["quencher_id"])
    print("\n  where the 48 quenchers' potentials come from:")
    for kind, ids in kinds.items():
        print(f"    {kind:<28} {len(ids):>2}   {ids}")


if __name__ == "__main__":
    main()
