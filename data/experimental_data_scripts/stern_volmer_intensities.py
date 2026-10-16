# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Stern-Volmer plate reader exports -> one emission intensity per well.

Reads   ../experimental_data_raw/stern_volmer/
        <batch>.csv               BMG plate export, repeat reads per well
        <batch>.well_plate.json   Opentrons pipetting map
Writes  ../experimental_data_processed/stern_volmer_intensities.csv
        ../experimental_data_raw/stern_volmer/plate_reader_filters.csv   one row per batch

    uv run data/experimental_data_scripts/stern_volmer_intensities.py

81 batches x 96 wells = 7776 rows. The raw files and pipetting maps use the old
photocatalyst ids; they are mapped to `pc_new_id` on reading (../compound_identification.py).

Excitation and emission wavelengths are taken from the export header, e.g.
`Mode name; FI: Ex: 425-8 Em: 620-50` (centre-bandwidth, nm). The wavelengths in the
pipetting maps are planning values and are not read. PC1 ("FI: Ir-F") and PC3
("FI: Ir(ppy)3") were read with saved protocols whose export names no filters; their
settings are in PROTOCOL_WAVELENGTHS.
"""

import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compound_identification import NEW_ID  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "experimental_data_raw" / "stern_volmer"
OUT = Path(__file__).resolve().parent.parent / "experimental_data_processed" / "stern_volmer_intensities.csv"

# The detector saturates at 260000 counts. Reads at or above this value are dropped
# before the well is averaged.
DETECTOR_CEILING = 260000
MAX_RSD = 0.05        # above this relative standard deviation, the worst read is dropped
CONC_SIGFIGS = 6
INTENSITY_DP = 1

FILTERS_OUT = RAW / "plate_reader_filters.csv"
# "FI: Ex: 425-8 Em: 620-50" -> centre and bandwidth of each filter, nm
MODE = re.compile(r"Ex:\s*(\d+)-(\d+)\s*Em:\s*(\d+)-(\d+)")
# Filter settings for the two saved protocols whose export names no filters. Centre
# wavelengths are from the photocatalyst sheet of the screen. Bandwidths were not
# recorded and are set to the instrument's usual values (excitation 8 nm, emission
# 20 nm); the `source` column of plate_reader_filters.csv marks these rows. Bandwidths
# are reported for the methods only and are not used in the analysis.
PROTOCOL_WAVELENGTHS = {
    # mode name: (excitation, excitation bandwidth, emission, emission bandwidth) / nm
    "FI: Ir-F": (425, 8, 475, 20),        # PC1
    "FI: Ir(ppy)3": (425, 8, 520, 20),    # PC3
}

COLUMNS = [
    "batch", "well", "pc_new_id", "quencher_id",
    "excitation_nm", "emission_nm", "quencher_conc_M", "intensity_au",
    "n_reads_total", "n_reads_used", "n_saturated", "n_deviant",
]
FILTER_COLUMNS = [
    "batch", "pc_new_id", "mode_name",
    "excitation_nm", "excitation_bandwidth_nm", "emission_nm", "emission_bandwidth_nm",
    "source",
]


def sig(x, n=CONC_SIGFIGS):
    """Round to n significant figures."""
    if x is None:
        return ""
    if x == 0:
        return 0.0
    return float(f"{x:.{n}g}")


def new_id(label):
    """New photocatalyst id of a pipetting-map label PC<old id>."""
    old = int(re.fullmatch(r"PC(\d+)", label).group(1))
    if old not in NEW_ID:
        raise ValueError(f"pipetting map names old PC{old}, which is not in the data")
    return NEW_ID[old]


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
    """well -> (quencher_id, quencher_name) for each QBatch.

    All plates of a batch share one quencher layout, so the quencher is identified by
    well position. The layout is taken from the plates whose labels carry `_QID_<id>`;
    some plates use bare quencher names only.
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
    """Stop if a plate with bare quencher names does not match its batch layout."""
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
    """Returns (well -> [repeat reads], filter record) from a semicolon-separated export."""
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
    """Filter settings from the Mode name, or from PROTOCOL_WAVELENGTHS."""
    m = MODE.search(mode or "")
    if m:
        ex, ex_bw, em, em_bw = (int(g) for g in m.groups())
        return {"mode_name": mode, "excitation_nm": ex, "excitation_bandwidth_nm": ex_bw,
                "emission_nm": em, "emission_bandwidth_nm": em_bw, "source": "export"}
    if mode in PROTOCOL_WAVELENGTHS:
        ex, ex_bw, em, em_bw = PROTOCOL_WAVELENGTHS[mode]
        return {"mode_name": mode, "excitation_nm": ex, "excitation_bandwidth_nm": ex_bw,
                "emission_nm": em, "emission_bandwidth_nm": em_bw,
                "source": "protocol names no filters: wavelengths from the photocatalyst "
                          "sheet, bandwidths inferred from the instrument's standing values"}
    raise ValueError(f"{name}: no filter pair in mode {mode!r} and no known protocol")


def qc(reads):
    """Drop saturated reads, then the read furthest from the median if the RSD exceeds
    MAX_RSD (needs at least 3 reads). Returns (mean, n_kept, n_saturated, n_deviant)."""
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
        pc_ids = {new_id(w["photocatalyst"]["compound"]) for w in wells.values()}
        if len(pc_ids) != 1:
            raise ValueError(f"{plate}: more than one photocatalyst on the plate: {pc_ids}")
        filter_rows.append({"batch": plate, "pc_new_id": next(iter(pc_ids)), **filters})

        missing = set(wells) - set(reads_by_well)
        if missing:
            raise ValueError(f"{plate}.csv has no reads for {sorted(missing)}")

        for well in sorted(wells, key=well_sort_key):
            entry = wells[well]
            pc, q = entry["photocatalyst"], entry["quencher"]
            quencher_id, _ = layout[well]

            # From pipetted volume and stock concentration, not the rounded concentration_mM.
            conc_M = q["volume_uL"] * q["stock"]["concentration"] / entry["total_volume_uL"] / 1000

            reads = reads_by_well[well]
            intensity, n_used, n_saturated, n_deviant = qc(reads)

            rows.append({
                "batch": plate,
                "well": well,
                "pc_new_id": new_id(pc["compound"]),
                "quencher_id": quencher_id,
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
        per_pc.setdefault(f["pc_new_id"], set()).add((f["excitation_nm"], f["emission_nm"]))
    mixed = {pc: pairs for pc, pairs in per_pc.items() if len(pairs) > 1}
    if mixed:
        sys.exit(f"batches of one photocatalyst were read with different filters: {mixed}")

    print(f"{len(map_paths)} batches -> {len(rows)} wells -> {OUT.name}")
    print(f"  filters from the export for {sum(f['source'] == 'export' for f in filter_rows)} "
          f"batches, from PROTOCOL_WAVELENGTHS for the rest -> {FILTERS_OUT.name}")
    print(f"  saturated reads dropped:        {sum(r['n_saturated'] for r in rows)}")
    print(f"  wells with an outlier dropped:  {sum(r['n_deviant'] for r in rows)}")
    print(f"  wells with no usable read:      {sum(1 for r in rows if r['intensity_au'] == '')}")


if __name__ == "__main__":
    main()
