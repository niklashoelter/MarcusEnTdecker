# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Stern-Volmer plate exports -> one QC'd emission intensity per well.

Reads   ../experimental_data_raw/stern_volmer/
        <batch>.csv               BMG plate export, repeat reads per well
        <batch>.well_plate.json   Opentrons pipetting map
Writes  ../experimental_data_processed/stern_volmer_intensities.csv
        ../experimental_data_raw/stern_volmer/plate_reader_filters.csv   one row per batch

81 batches x 96 wells = 7776 rows. Photocatalysts are old ids (`pc_old_id`); see
../photocatalyst_ids.py.

THE WAVELENGTHS COME FROM THE EXPORT, NOT FROM THE MAP
    The plate reader writes the filter pair it used into the export header, e.g.
    `Mode name; FI: Ex: 425-8 Em: 620-50` (centre-bandwidth, nm). That is the record of
    what was measured. The pipetting maps also carry excitation and emission wavelengths,
    but those are planning placeholders that disagree with the export for most batches;
    they are not read.

    Two photocatalysts were measured with a saved protocol whose name, not its filters,
    is all the export carries: old PC1 ("FI: Ir-F") and old PC3 ("FI: Ir(ppy)3"). Their
    wavelengths come from the photocatalyst sheet of the screen
    (Photocatalysts/photocatalysts.xlsx); their bandwidths are not recorded.
"""

import csv
import json
import re
import statistics
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "experimental_data_raw" / "stern_volmer"
OUT = Path(__file__).resolve().parent.parent / "experimental_data_processed" / "stern_volmer_intensities.csv"

# The detector rails at exactly 260000: 75 reads across the 81 plates sit on that
# value and nothing else in the screen exceeds 250000. A railed read is not a
# measurement of anything, so it is discarded before the well is averaged.
DETECTOR_CEILING = 260000
MAX_RSD = 0.05        # a well whose reads spread less than this keeps all of them
CONC_SIGFIGS = 6      # pipetting is good to ~1%; six figures is already generous
INTENSITY_DP = 1      # a mean of integer counts; the fraction below 0.1 is meaningless

FILTERS_OUT = RAW / "plate_reader_filters.csv"
# "FI: Ex: 425-8 Em: 620-50" -> centre and bandwidth of each filter, nm
MODE = re.compile(r"Ex:\s*(\d+)-(\d+)\s*Em:\s*(\d+)-(\d+)")
# Saved protocols whose export names no filters: (excitation, emission) from the
# photocatalyst sheet of the screen. Bandwidths unknown.
PROTOCOL_WAVELENGTHS = {
    "FI: Ir-F": (425, 475),        # old PC1
    "FI: Ir(ppy)3": (425, 520),    # old PC3
}

COLUMNS = [
    "batch", "well", "pc_old_id", "quencher_id", "quencher_label",
    "excitation_nm", "emission_nm", "quencher_conc_M", "intensity_au",
    "n_reads_total", "n_reads_used", "n_saturated", "n_deviant",
]
FILTER_COLUMNS = [
    "batch", "pc_old_id", "mode_name",
    "excitation_nm", "excitation_bandwidth_nm", "emission_nm", "emission_bandwidth_nm",
    "source",
]


def sig(x, n=CONC_SIGFIGS):
    """Round to n significant figures. Returns a float so the CSV carries no trailing zeros."""
    if x is None:
        return ""
    if x == 0:
        return 0.0
    return float(f"{x:.{n}g}")


def well_sort_key(w):
    return (w[0], int(w[1:]))


def read_map(path):
    return json.loads(path.read_text(encoding="utf-8"))["wells"]


def batch_number(plate_name):
    m = re.search(r"QBatch(\d+)", plate_name)
    if not m:
        raise ValueError(f"no QBatch number in plate name {plate_name!r}")
    return int(m.group(1))


def build_batch_layouts(maps):
    """well -> (quencher_id, quencher_name) per QBatch, taken from the QID-carrying plates.

    Quencher identity is a property of the well position within a batch, not of the
    label written on the plate: ten plates label their quenchers with bare names and
    two of those names are misleading. Every plate that does carry _QID_ labels agrees
    on the layout, so position is the unambiguous key.
    """
    layouts = {}
    for plate, wells in maps.items():
        labels = {w: v["quencher"]["compound"] for w, v in wells.items()}
        if not all("_QID_" in q for q in labels.values()):
            continue
        resolved = {}
        for w, label in labels.items():
            name, qid = label.rsplit("_QID_", 1)
            resolved[w] = (int(qid), name)
        b = batch_number(plate)
        if b in layouts and layouts[b] != resolved:
            raise ValueError(f"QBatch{b} layout disagrees between plates; {plate} differs")
        layouts[b] = resolved
    return layouts


def check_bare_plates(maps, layouts):
    """Every bare-name plate must match its batch layout name-for-name, or we stop."""
    for plate, wells in maps.items():
        labels = {w: v["quencher"]["compound"] for w, v in wells.items()}
        if all("_QID_" in q for q in labels.values()):
            continue
        layout = layouts[batch_number(plate)]
        for w, bare in labels.items():
            expected = layout[w][1]
            if bare != expected:
                raise ValueError(
                    f"{plate} well {w}: label {bare!r} does not match "
                    f"QBatch{batch_number(plate)} layout {expected!r}"
                )


def read_plate(path):
    """(well -> [repeat reads], filter record). The export is semicolon-separated with a
    metadata header, which carries the filter pair the plate was read with."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    reads, mode = {}, None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if parts[0] == "Mode name" and len(parts) > 1:
            mode = parts[1]
        if not re.fullmatch(r"[A-H](?:[1-9]|1[0-2])", parts[0] or ""):
            continue
        values = [int(p) for p in parts[1:] if p]
        if values:
            reads[parts[0]] = values
    return reads, filters_from_mode(mode, path.name)


def filters_from_mode(mode, name):
    """The filter record of one export: from its Mode name, or from the photocatalyst
    sheet for the two saved protocols that name no filters."""
    m = MODE.search(mode or "")
    if m:
        ex, ex_bw, em, em_bw = (int(g) for g in m.groups())
        return {"mode_name": mode, "excitation_nm": ex, "excitation_bandwidth_nm": ex_bw,
                "emission_nm": em, "emission_bandwidth_nm": em_bw, "source": "export"}
    if mode in PROTOCOL_WAVELENGTHS:
        ex, em = PROTOCOL_WAVELENGTHS[mode]
        return {"mode_name": mode, "excitation_nm": ex, "excitation_bandwidth_nm": "",
                "emission_nm": em, "emission_bandwidth_nm": "",
                "source": "photocatalyst sheet (protocol names no filters)"}
    raise ValueError(f"{name}: no filter pair in mode {mode!r} and no known protocol")


def qc(reads):
    """Drop rail reads, then one outlier if the well is too noisy. Returns (mean, kept, sat, dev)."""
    kept = [r for r in reads if r < DETECTOR_CEILING]
    n_saturated = len(reads) - len(kept)
    n_deviant = 0

    if len(kept) >= 3:
        mean = statistics.fmean(kept)
        if mean > 0 and statistics.stdev(kept) / mean > MAX_RSD:
            median = statistics.median(kept)
            worst = max(range(len(kept)), key=lambda i: abs(kept[i] - median))
            kept.pop(worst)
            n_deviant = 1

    intensity = round(statistics.fmean(kept), INTENSITY_DP) if kept else None
    return intensity, len(kept), n_saturated, n_deviant


def main():
    map_paths = sorted(RAW.glob("*.well_plate.json"))
    if not map_paths:
        sys.exit(f"no pipetting maps found in {RAW}")

    maps = {p.name.removesuffix(".well_plate.json"): read_map(p) for p in map_paths}
    layouts = build_batch_layouts(maps)
    check_bare_plates(maps, layouts)

    rows, filter_rows = [], []
    for plate in sorted(maps):
        wells = maps[plate]
        reads_by_well, filters = read_plate(RAW / f"{plate}.csv")
        layout = layouts[batch_number(plate)]
        pc_ids = {int(re.fullmatch(r"PC(\d+)", w["photocatalyst"]["compound"]).group(1))
                  for w in wells.values()}
        if len(pc_ids) != 1:
            raise ValueError(f"{plate}: more than one photocatalyst on the plate: {pc_ids}")
        filter_rows.append({"batch": plate, "pc_old_id": next(iter(pc_ids)), **filters})

        missing = set(wells) - set(reads_by_well)
        if missing:
            raise ValueError(f"{plate}.csv has no reads for {sorted(missing)}")

        for well in sorted(wells, key=well_sort_key):
            entry = wells[well]
            pc, q = entry["photocatalyst"], entry["quencher"]
            quencher_id, quencher_name = layout[well]

            # Concentration comes from the pipetted volume and the stock, never from the
            # rounded concentration_mM the map also carries.
            conc_M = q["volume_uL"] * q["stock"]["concentration"] / entry["total_volume_uL"] / 1000

            reads = reads_by_well[well]
            intensity, n_used, n_saturated, n_deviant = qc(reads)

            rows.append({
                "batch": plate,
                "well": well,
                "pc_old_id": int(re.fullmatch(r"PC(\d+)", pc["compound"]).group(1)),
                "quencher_id": quencher_id,
                "quencher_label": quencher_name,
                "excitation_nm": filters["excitation_nm"],
                "emission_nm": filters["emission_nm"],
                "quencher_conc_M": sig(conc_M),
                "intensity_au": "" if intensity is None else intensity,
                "n_reads_total": len(reads),
                "n_reads_used": n_used,
                "n_saturated": n_saturated,
                "n_deviant": n_deviant,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with FILTERS_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FILTER_COLUMNS)
        writer.writeheader()
        writer.writerows(filter_rows)

    per_pc = {}
    for f in filter_rows:
        per_pc.setdefault(f["pc_old_id"], set()).add((f["excitation_nm"], f["emission_nm"]))
    mixed = {pc: pairs for pc, pairs in per_pc.items() if len(pairs) > 1}
    if mixed:
        sys.exit(f"batches of one photocatalyst were read with different filters: {mixed}")

    print(f"{len(map_paths)} batches -> {len(rows)} wells -> {OUT.name}")
    print(f"  filters from the export for {sum(f['source'] == 'export' for f in filter_rows)} "
          f"batches, from the photocatalyst sheet for the rest -> {FILTERS_OUT.name}")
    print(f"  saturated reads dropped:        {sum(r['n_saturated'] for r in rows)}")
    print(f"  wells with an outlier dropped:  {sum(r['n_deviant'] for r in rows)}")
    print(f"  wells with no usable read:      {sum(1 for r in rows if r['intensity_au'] == '')}")


if __name__ == "__main__":
    main()
