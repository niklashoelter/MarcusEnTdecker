# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Photocatalyst emission scan -> one intensity per photocatalyst, excitation and wavelength.

Reads   ../experimental_data_raw/pc_emission/
        Emission_Scan_Exc_380_405_425_450_Gain_1500.csv
Writes  ../experimental_data_raw/pc_emission/emission_spectra.csv

This is the only file in pc_emission/ that identifies its samples. Its rows are labelled
with the photocatalyst number directly, so no plate map is needed. Every other file in
that folder is ignored, and deliberately so:

  * six are autofluorescence tests of plate materials (PP, glass-coated PP, quartz)
  * the rest are single-excitation method-development scans labelled by well, with no
    record anywhere of which well held which photocatalyst, plus files marked
    `quick`, `DRY` and `old_`

The scan has four excitation blocks. Emission ranges differ per block because each was
set to start above its own excitation wavelength:

    Ex 450 nm -> Em 475-650      Ex 425 nm -> Em 450-600
    Ex 405 nm -> Em 430-600      Ex 380 nm -> Em 410-550

These scans are what the Stern-Volmer read channels were chosen from. The channel
actually used for each photocatalyst is recorded in the pipetting maps and carried
through to stern_volmer_intensities.csv, so this table documents the choice rather than
driving anything downstream.
"""

import csv
import re
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "experimental_data_raw" / "pc_emission"
OUT = RAW / "emission_spectra.csv"

SCAN = "Emission_Scan_Exc_380_405_425_450_Gain_1500.csv"

# Photocatalysts are numbered 1-29. The 380 nm block carries one extra row labelled 96,
# after PC27, which is not a photocatalyst; it is reported and skipped rather than
# silently dropped.
MAX_PC_ID = 29

COLUMNS = ["pc_id", "excitation_nm", "emission_nm", "intensity_au"]


def parse_blocks(path):
    """Yield (excitation_nm, [wavelengths], {pc_id: [intensities]}) per excitation block."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="latin-1").splitlines()

    excitation = None
    wavelengths = None
    rows = {}
    for line in lines:
        fields = [f.strip() for f in line.split(";")]
        head = fields[0]

        if head == "Mode name":
            if excitation is not None and wavelengths and rows:
                yield excitation, wavelengths, rows
            # e.g. "FI: Ex: 450-8 Em: 475->650"
            m = re.search(r"Ex:\s*(\d+)", fields[1])
            if not m:
                raise ValueError(f"cannot read an excitation wavelength from {fields[1]!r}")
            excitation, wavelengths, rows = int(m.group(1)), None, {}
            continue

        if head == "Wavelength [nm]":
            wavelengths = [int(float(f)) for f in fields[1:] if f]
            continue

        if re.fullmatch(r"\d+", head) and wavelengths:
            values = [int(float(f.replace(",", "."))) for f in fields[1:] if f]
            rows[int(head)] = values[:len(wavelengths)]

    if excitation is not None and wavelengths and rows:
        yield excitation, wavelengths, rows


def main():
    path = RAW / SCAN
    if not path.exists():
        sys.exit(f"missing input: {path}")

    out_rows = []
    peaks = []
    skipped = set()
    for excitation, wavelengths, rows in parse_blocks(path):
        for pc_id in sorted(rows):
            if not 1 <= pc_id <= MAX_PC_ID:
                skipped.add(pc_id)
                continue
            values = rows[pc_id]
            if len(values) != len(wavelengths):
                raise ValueError(
                    f"PC{pc_id} at Ex {excitation} nm has {len(values)} values "
                    f"for {len(wavelengths)} wavelengths")
            for wl, intensity in zip(wavelengths, values):
                out_rows.append({"pc_id": pc_id, "excitation_nm": excitation,
                                 "emission_nm": wl, "intensity_au": intensity})
            peak_wl = wavelengths[max(range(len(values)), key=values.__getitem__)]
            peaks.append((pc_id, excitation, peak_wl, max(values)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    photocatalysts = sorted({r["pc_id"] for r in out_rows})
    excitations = sorted({r["excitation_nm"] for r in out_rows}, reverse=True)
    print(f"{len(out_rows)} points -> {OUT.name}")
    print(f"  {len(photocatalysts)} photocatalysts, {len(peaks)} spectra, "
          f"excitations {excitations}")
    print(f"  photocatalysts present: {photocatalysts}")
    if skipped:
        print(f"  rows skipped, label outside the photocatalyst range 1-{MAX_PC_ID}: "
              f"{sorted(skipped)}")
    print("  brightest excitation per photocatalyst (peak emission):")
    for pc_id in photocatalysts:
        best = max((p for p in peaks if p[0] == pc_id), key=lambda p: p[3])
        print(f"    PC{best[0]:<3d} Ex {best[1]} nm -> peak {best[2]} nm  ({best[3]} counts)")


if __name__ == "__main__":
    main()
