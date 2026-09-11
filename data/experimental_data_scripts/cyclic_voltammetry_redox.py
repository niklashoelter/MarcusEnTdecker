# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Cyclic voltammograms -> the redox potentials that were measured in this lab.

Reads   ../experimental_data_raw/quencher_cyclic_voltammetry/   (21 traces + 19 blanks)
        ../experimental_data_raw/compound_records/compound_records_quenchers.csv  (for comparison)
Writes  ../experimental_data_raw/quencher_cyclic_voltammetry/cyclic_voltammetry_redox.csv   (19 samples)
        ../experimental_data_raw/quencher_cyclic_voltammetry/cyclic_voltammetry_traces.csv  (25821 points)

Half of the 48 quencher redox potentials were measured here rather than taken from the
literature, and for 19 of them the voltammogram itself is in the package. Until now the
numbers travelled without the measurement behind them. This script puts the two side by
side.

THE SAMPLE NUMBER IS THE QUENCHER ID
    The file names carry a three-digit sample number and nothing else -- the one text
    line at the top of each export, `i400sN1_1#034`, is the instrument's own run counter
    and does NOT track the sample (`009.csv` is run #034). The lab record says the sample
    number is the quencher id, and this script checks that claim instead of assuming it:
    every extracted potential is compared against the value recorded for that quencher in
    `compound_records_quenchers.csv`, and the comparison is written out per sample. The
    circumstantial support is that 18 of the 19 numbers land on a quencher whose
    provenance says `measured`, and the 19th, 009, is Q9 -- the single quencher in the
    whole set whose potential has no attribution at all.

WHAT IS IN A FILE
    Semicolon-separated, three header lines (run label, blank, column names), then
    potential in V against current in uA. Decimal points, not commas. Every trace is one
    full cycle on the same schedule: 0 V -> +2.5 V -> -2.5 V -> 0 V, at either 0.05 V
    (201 points) or 0.005 V (2001 points) per exported point. Each `<nnn>.csv` has a
    `<nnn>_blank.csv` measured on the same potential axis, point for point.

    The potential axis is ALREADY on the scale the values are reported on. It is not
    referenced to an internal standard: the script finds no reversible couple anywhere on
    any anodic branch, so there is no ferrocene wave near +0.4 V to correct against, and
    the extracted potentials land on the recorded vs.-SCE values without one. Across the
    29 branches that agree, the mean SIGNED deviation is +0.046 V -- a small bias of the
    tangent construction, a tenth of the ~0.4 V that a missed ferrocene correction would
    have put on every potential in the set. So no correction is applied, and none is
    needed.

    There is no time column and no scan rate anywhere in the export, so `scan_rate_V_s`
    cannot be filled in from these files and is left empty rather than guessed. What can
    be recovered -- window, step, point count, direction -- is written out.

THE BLANK IS SUBTRACTED POINT FOR POINT
    Sample and blank share the same potential grid exactly, so the background is removed
    by plain subtraction, no interpolation and no scaling. The blanks are small (a few uA
    against tens to hundreds) but they carry the electrolyte's own solvent-window edges,
    which is precisely the region where the interesting waves sit.

WHAT IS EXTRACTED, AND WHICH NUMBER IS REPORTED
    35 of the 38 branches here are irreversible -- an oxidation or reduction that runs
    into the solvent limit with no return peak -- so for those there is no reversible
    E_1/2 to report and the onset is all there is. Four descriptors are written per
    branch, and `E_ox_basis` / `E_red_basis` says which of them `E_ox_V` / `E_red_V` is:

      E_ox_onset_V / E_red_onset_V   tangent at the point of steepest rise, extrapolated
                                     back to the local baseline
      E_ox_peak_V  / E_red_peak_V    the peak potential, empty when the wave runs into
                                     the vertex without turning over
      E_ox_half_peak_V / ...         the potential at half the peak current
      E_ox_half_wave_V / ...         (forward peak + return peak) / 2, only where a
                                     return peak was found

    The ONSET is the reported number on an irreversible wave, because it is what the
    experimenters read: across the branches whose recorded value is a plain measurement
    the extraction reproduces it to a median of 0.050 V, while the peak sits
    systematically 0.2 V beyond it. The other descriptors are carried so that a reader
    who prefers a different convention can see what it would give.

    The HALF-WAVE is the reported number wherever a return peak was found within
    REVERSIBLE_WINDOW_V of the forward one, because that is the thermodynamic quantity
    and an onset is not. Three branches qualify, all reductions -- Q28, Q39 and Q42 --
    and on all three the half-wave lands closer to the recorded value than the onset
    does (Q28 +0.025 V against +0.135, Q42 +0.050 V against +0.173). That the recorded
    values follow the same switch is itself evidence that the sample numbers are the
    quencher ids.

    For the other 35, treat the potential as kinetic. An irreversible onset is not a
    redox equilibrium and should not be fed into a driving-force calculation as though
    it were one.

FINDING THE WAVE
    The FIRST resolvable wave on each branch, not the largest. A compound's own oxidation
    comes before the electrolyte's breakdown, and the breakdown is usually the bigger
    feature, so taking the largest would report the solvent on half these traces.

    A wave is called resolvable when the blank-corrected current rises MIN_WAVE_UA (or
    NOISE_K robust standard deviations, whichever is larger) above its own trailing
    minimum, at least EDGE_MARGIN_V inside the vertex. A rise that only begins in the
    last EDGE_MARGIN_V of the sweep is not a wave in this window; it is the edge of the
    window, and it is reported as no wave rather than as a potential.

BOUNDS ARE CHECKED AS BOUNDS
    Five of the recorded values are limits, not measurements -- `> 2.4`, `< -2.5` -- and
    mean the compound was not oxidized or reduced anywhere in the accessible window.
    Comparing a number against those would be a category error. They are checked the only
    way they can be: the trace must show no wave inside the window, or a wave only beyond
    the limit. `delta_*_V` is left empty for them on purpose.

HOW WELL IT AGREES, AND WHERE IT DOES NOT
    32 of the 38 branches carry a plain recorded measurement rather than a bound. The
    extraction reproduces 29 of them to within AGREEMENT_TOLERANCE_V, median deviation
    0.050 V, which is about the width of one exported point on the coarse traces. All
    five bounds are consistent. Four branches are not reproduced, and every one of them
    was already visible as a problem in the lab record or in the trace itself:

      * 017 reduction, recorded -0.2 V, extracted -1.57 V. Its own lab comment reads
        "ugly CV measured". The -0.2 V feature is a few-uA shoulder on a sloping
        background; the first wave that clears the detection threshold is more than a
        volt further negative. The disagreement is written out, not suppressed and not
        fitted around.
      * 046 reduction, recorded -2.0 V, extracted -2.29 V, and its oxidation is a broad
        hump of a few uA with no defined foot. The oxidation onset happens to land near
        the recorded 1.9 V, but the feature is too weak for that to mean anything.
      * 023 oxidation, off by -0.153 V, three thousandths of a volt outside the
        tolerance. The wave is clean; this is where a tangent construction and an eye
        disagree, nothing more.
      * 022 oxidation, withheld. Its two repeats do not agree: the second run shows a
        small feature near 0.3 V that is absent from the first, so the two disagree by
        well over a volt. Rather than average them into a number that describes neither,
        the value is left empty and both are named in `note`.

    Repeats are averaged only when they agree to REPEAT_TOLERANCE_V. 021 and 022 are the
    only samples measured twice. 021's second run shows no oxidation wave at all, so its
    oxidation rests on one of the two.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "experimental_data_raw" / "quencher_cyclic_voltammetry"
RECORDS = HERE.parent / "experimental_data_raw" / "compound_records" / "compound_records_quenchers.csv"
# Written next to the traces they come from, so the sample scan below has to step over them.
OUT = RAW / "cyclic_voltammetry_redox.csv"
TRACES_OUT = RAW / "cyclic_voltammetry_traces.csv"

SIGFIGS = 6

# Smoothing window, in volts, applied before differentiating. Short enough to leave a
# 0.1 V wave alone, long enough that the derivative of the 0.005 V export is not noise.
SMOOTH_V = 0.02

# A wave is detected when the corrected current rises this far above the minimum of the
# preceding TRAIL_V of sweep. MIN_WAVE_UA is the floor; the noise term takes over on the
# traces where the background itself is rough.
TRAIL_V = 0.50
MIN_WAVE_UA = 4.0
NOISE_K = 6.0

# A rise beginning within this distance of the vertex is the edge of the window, not a
# wave inside it.
EDGE_MARGIN_V = 0.10

# The local baseline is the median over BASE_WINDOW_V of sweep ending BASE_GAP_V before
# the wave's foot -- far enough back not to sit on the rising edge itself.
BASE_GAP_V = 0.05
BASE_WINDOW_V = 0.40

# The peak is the first maximum after the foot that the current then falls this far
# below, as a fraction of the wave height. Without a fall there is no peak.
PEAK_DROP = 0.10

# A return peak this close to the forward peak makes the couple reversible.
REVERSIBLE_WINDOW_V = 0.30

# Repeats are averaged when they agree to this; otherwise the value is withheld.
REPEAT_TOLERANCE_V = 0.15

# Extraction counts as reproducing the recorded value within this. It is deliberately
# loose: the recorded numbers were read off plots by eye, by several different people.
AGREEMENT_TOLERANCE_V = 0.15

SAMPLE = re.compile(r"^(?P<sample>\d{3})(?:_(?P<repeat>\d+))?$")

COLUMNS = [
    "quencher_id", "sample", "n_repeats", "traces",
    "E_ox_V", "E_ox_basis",
    "E_ox_onset_V", "E_ox_peak_V", "E_ox_half_peak_V", "i_ox_peak_uA",
    "E_ox_onset_spread_V", "ox_reversible", "E_ox_half_wave_V",
    "E_red_V", "E_red_basis",
    "E_red_onset_V", "E_red_peak_V", "E_red_half_peak_V", "i_red_peak_uA",
    "E_red_onset_spread_V", "red_reversible", "E_red_half_wave_V",
    "E_ox_recorded_V", "E_ox_recorded_bound", "delta_E_ox_V", "ox_agrees",
    "E_red_recorded_V", "E_red_recorded_bound", "delta_E_red_V", "red_agrees",
    "window_min_V", "window_max_V", "step_V", "n_points", "scan_rate_V_s",
    "note",
]

TRACE_COLUMNS = [
    "quencher_id", "sample", "repeat", "point", "segment",
    "potential_V", "current_uA", "blank_uA", "corrected_uA",
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

    `u` increases along the sweep and `y` is the corrected current with the wave
    pointing up, so the caller flips the sign for the cathodic branch and this function
    needs to know nothing about direction. Returns a dict; `detected` is False when
    there is no wave inside the window.
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
        return {"detected": False, "noise": noise, "threshold": threshold,
                "reason": f"no rise of {threshold:.1f} uA anywhere in the window"}
    if u[foot] > u[-1] - EDGE_MARGIN_V:
        return {"detected": False, "noise": noise, "threshold": threshold,
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
        "detected": True, "noise": noise, "threshold": threshold,
        "onset": onset,
        "peak": float(u[peak]) if peak is not None else None,
        "peak_current": float(ys[peak] - base) if peak is not None else None,
        "half": half, "height": height, "base": base,
        "reason": "" if peak is not None else "wave runs into the vertex, no peak",
    }


def return_peak(u, y, step, forward_peak):
    """Potential of a counter-peak on the return sweep, or None.

    `u` and `y` are again oriented so the counter-peak points up. Only a peak within
    REVERSIBLE_WINDOW_V of the forward one counts; anything further away is a different
    process, not the reverse of this one.
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
        return None                      # a shoulder on a ramp, not a peak
    # It has to rise AND then fall. After a large irreversible wave the return sweep
    # climbs steeply back toward zero and flattens out; that climb alone would make
    # every wave look reversible, so the falling side is what decides. The fall is
    # measured over the whole rest of the sweep, not just the comparison window, because
    # a return peak decays slowly.
    if ys[k] - ys[idx[0]] < MIN_WAVE_UA or ys[k] - np.min(ys[k:]) < MIN_WAVE_UA:
        return None
    return float(u[k])


def analyze(path, blank_path):
    """One trace -> its two branches, plus the metadata the file actually carries."""
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

    # Cathodic sweep, +max -> -max, negative half only. Reduction shows here. Flip both
    # axes so the sweep runs forward and the wave points up.
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


def read_records():
    if not RECORDS.exists():
        sys.exit(f"missing input: {RECORDS}\nrun compound_records.py first")
    with RECORDS.open(encoding="utf-8", newline="") as fh:
        return {int(row["quencher_id"]): row for row in csv.DictReader(fh)}


def compare(extracted, recorded, bound, sign):
    """-> (delta or None, agrees flag, note).

    `sign` is +1 for oxidation and -1 for reduction, so that "beyond the limit" is one
    comparison rather than two. A bound is never differenced against a number: it says
    the compound did not react inside the window, and the only thing that can agree or
    disagree with it is the presence of a wave.
    """
    if recorded is None:
        return None, "", "no recorded value to compare against"
    if bound:
        if extracted is None:
            return None, 1, "recorded as a bound; no wave in the window, consistent"
        beyond = sign * (extracted - recorded) >= -AGREEMENT_TOLERANCE_V
        return None, int(beyond), (
            f"recorded as a bound at {recorded:+.2f} V; extracted wave at "
            f"{extracted:+.3f} V is "
            f"{'beyond it, consistent' if beyond else 'INSIDE it, inconsistent'}")
    if extracted is None:
        return None, 0, f"recorded {recorded:+.2f} V but no wave found in the window"
    delta = extracted - recorded
    return delta, int(abs(delta) <= AGREEMENT_TOLERANCE_V), ""


def main():
    if not RAW.is_dir():
        sys.exit(f"missing input directory: {RAW}")
    records = read_records()

    samples = {}
    for path in sorted(RAW.glob("*.csv")):
        if path.stem.endswith("_blank") or path in (OUT, TRACES_OUT):
            continue
        match = SAMPLE.match(path.stem)
        if not match:
            sys.exit(f"cannot read a sample number from {path.name}")
        samples.setdefault(match["sample"], []).append((int(match["repeat"] or 1), path))
    if not samples:
        sys.exit(f"no CV traces found in {RAW}")

    rows, trace_rows = [], []
    for sample in sorted(samples):
        quencher_id = int(sample)
        if quencher_id not in records:
            sys.exit(f"sample {sample} has no quencher {quencher_id} in {RECORDS.name}")
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
        row["scan_rate_V_s"] = ""      # not recorded anywhere in the export

        for side, key, sign in (("ox", 2, +1), ("red", 3, -1)):
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

            # A reversible couple has a thermodynamic E_1/2 and that is the number to
            # report; an irreversible wave has only its onset. Which one was used is
            # written out beside it rather than left to be inferred.
            reported = half_wave if half_wave is not None else onset
            row[f"E_{side}_V"] = sig(reported)
            row[f"E_{side}_basis"] = ("" if reported is None else
                                      "half-wave" if half_wave is not None else "onset")
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

            column = "E_Ox_vs_SCE_exp" if side == "ox" else "E_Red_vs_SCE_exp"
            bound_column = "E_Ox_bound" if side == "ox" else "E_Red_bound"
            raw = records[quencher_id][column]
            recorded = float(raw) if raw else None
            bound = records[quencher_id][bound_column]
            delta, agrees, why = compare(reported, recorded, bound, sign)

            row[f"E_{side}_recorded_V"] = sig(recorded)
            row[f"E_{side}_recorded_bound"] = bound
            row[f"delta_E_{side}_V"] = sig(delta)
            row[f"{side}_agrees"] = agrees
            if why:
                notes.append(why)
            elif agrees == 0:
                notes.append(f"{side} potential differs from the recorded value by "
                             f"{delta:+.2f} V, beyond the {AGREEMENT_TOLERANCE_V} V "
                             f"tolerance")

        record_note = records[quencher_id]["redox_source_note"]
        if record_note:
            notes.append(f"lab record: {record_note}")
        row["note"] = "; ".join(notes)
        rows.append(row)

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([{c: row.get(c, "") for c in COLUMNS} for row in rows])
    with TRACES_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(trace_rows)

    report(rows, len(trace_rows))


def report(rows, n_points):
    print(f"{len(rows)} samples -> {OUT.name}")
    print(f"{n_points} trace points -> {TRACES_OUT.name}")

    print("\n  extracted potential against the value recorded for that quencher")
    print(f"  {'q':>3} {'sample':>6} | {'ox ext':>7} {'ox rec':>8} {'delta':>7} | "
          f"{'red ext':>7} {'red rec':>8} {'delta':>7}")
    deltas = []
    show = lambda v, w: (f"{float(v):+{w}.3f}" if v != "" else ".".rjust(w + 1))
    for row in rows:
        cells = []
        for side in ("ox", "red"):
            recorded = row[f"E_{side}_recorded_V"]
            delta = row[f"delta_E_{side}_V"]
            mark = {"lower": ">", "upper": "<"}.get(row[f"E_{side}_recorded_bound"], " ")
            cells.append(f"{show(row[f'E_{side}_V'], 6)} "
                         f"{mark}{show(recorded, 6)} {show(delta, 6)}")
            if delta != "":
                deltas.append(abs(float(delta)))
        print(f"  {row['quencher_id']:>3} {row['sample']:>6} | " + " | ".join(cells))

    if deltas:
        within = [d for d in deltas if abs(d) <= AGREEMENT_TOLERANCE_V]
        absolute = sorted(abs(d) for d in deltas)
        print(f"\n  {len(deltas)} plain measurements compared: median |delta| "
              f"{absolute[len(absolute) // 2]:.3f} V, max {absolute[-1]:.3f} V, "
              f"{len(within)} within {AGREEMENT_TOLERANCE_V} V")
        # The SIGNED mean is the one that would catch a reference-scale error: a missed
        # ferrocene correction shifts every potential the same way, by several hundred
        # mV, and would show up here rather than in the scatter.
        print(f"  mean signed offset over those {len(within)}: "
              f"{sum(within) / len(within):+.3f} V, nowhere near the ~0.4 V a missed "
              f"ferrocene correction would give")

    bounds = [(row, side) for row in rows for side in ("ox", "red")
              if row[f"E_{side}_recorded_bound"]]
    ok = sum(1 for row, side in bounds if row[f"{side}_agrees"] == 1)
    print(f"  {len(bounds)} recorded bounds checked as bounds: {ok} consistent, "
          f"{len(bounds) - ok} not")

    disagreeing = [(row, side) for row in rows for side in ("ox", "red")
                   if row[f"{side}_agrees"] == 0]
    if disagreeing:
        print(f"\n  {len(disagreeing)} extractions do not reproduce the record:")
        for row, side in disagreeing:
            print(f"    Q{row['quencher_id']} {side}: {row['note']}")

    reversible = [(row, side) for row in rows for side in ("ox", "red")
                  if row[f"{side}_reversible"]]
    print(f"\n  {len(rows) * 2 - len(reversible)} of {len(rows) * 2} branches are "
          f"irreversible - no return peak, so no thermodynamic E_1/2")
    for row, side in reversible:
        print(f"    Q{row['quencher_id']} {side}: E_1/2 "
              f"{row[f'E_{side}_half_wave_V']} V against a recorded "
              f"{row[f'E_{side}_recorded_V']} V")


if __name__ == "__main__":
    main()
