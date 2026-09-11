"""
The layout shared by the four property workbooks, and the one way to read them back.

    experimental_data_processed/photocatalyst_properties_exp.xlsx
    experimental_data_processed/quencher_properties_exp.xlsx
    computational_data_processed/photocatalyst_properties_calc.xlsx
    computational_data_processed/quencher_properties_calc.xlsx

One row per compound. Columns come in blocks -- identity first, then one block per
measured or computed property -- separated by an empty column and named in a band above
the header row. Headers carry their unit as `quantity / unit`. A second sheet, `Column
key`, gives every column its unit, a one-line description and the intermediate table in
`*_raw/` it was taken from.

Nothing in a workbook is computed here. Every value is copied from the per-measurement
table an extraction script wrote into the raw folder beside its instrument files, so a
number can always be followed back one step to where it was derived.

The files are written byte-for-byte reproducibly: the zip timestamps and the document
properties are pinned, so rerunning the pipeline on unchanged data changes nothing that
git can see.

Not a pipeline stage; imported by `properties_exp.py` and `properties_calc.py`.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HEADER_ROW = 2          # row 1 is the block band
FIRST_DATA_ROW = 3

FONT = "Arial"
BAND_FILLS = ["E7E6E6", "DDEBF7", "E2EFDA", "FCE4D6", "FFF2CC", "EDE2F6", "D9F2F2"]
PINNED_TIME = datetime(2000, 1, 1)
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Column:
    header: str                         # "tau0 / ns" -- the unit follows " / "
    description: str
    source: str                         # "pc_tcspc/lifetimes_tau0.csv: tau0_ns"
    value: Callable[[int], object]      # compound id -> cell value


@dataclass(frozen=True)
class Block:
    title: str
    columns: list[Column]


def read_table(path: Path, key: str) -> dict[int, dict[str, str]]:
    """An intermediate CSV as {id: row}. A missing file is fatal, not skipped."""
    if not path.exists():
        sys.exit(f"missing input: {path}\nrun the script that writes it first")
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(row[key]): row for row in csv.DictReader(fh)}


def typed(text):
    """A CSV field as the cell value it stands for: int, float, str, or None when blank.

    Leading zeros are kept as text, so a sample label like `009` is not turned into 9.
    """
    if text is None:
        return None
    text = str(text).strip()
    if text == "":
        return None
    if re.fullmatch(r"-?(0|[1-9]\d*)", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def field(table: dict[int, dict[str, str]], name: str) -> Callable[[int], object]:
    """A column getter: `name` from the row for that id, blank where there is none."""
    return lambda i: typed((table.get(i) or {}).get(name))


def _split_unit(header: str) -> tuple[str, str]:
    name, _, unit = header.partition(" / ")
    return name, unit


def write_workbook(path: Path, sheet: str, ids: list[int], blocks: list[Block],
                   note: str) -> None:
    """Write the property sheet and its column key, reproducibly."""
    wb = Workbook()
    ws = wb.active
    ws.title = sheet

    base = Font(name=FONT, size=10)
    bold = Font(name=FONT, size=10, bold=True)
    rule = Border(bottom=Side(style="thin", color="808080"))

    col = 1
    first_property_col = None
    widths = {}
    for b, block in enumerate(blocks):
        fill = PatternFill("solid", fgColor=BAND_FILLS[b % len(BAND_FILLS)])
        start = col
        for column in block.columns:
            head = ws.cell(HEADER_ROW, col, column.header)
            head.font, head.fill, head.border = bold, fill, rule
            head.alignment = Alignment(vertical="bottom")
            width = len(column.header)
            for r, i in enumerate(ids, start=FIRST_DATA_ROW):
                v = column.value(i)
                cell = ws.cell(r, col, v)
                cell.font = base
                if v is not None:
                    width = max(width, len(str(v)) if not isinstance(v, float) else 10)
            widths[col] = min(max(width + 2, 8), 45)
            col += 1
        if col - 1 > start:     # a one-column block has nothing to merge
            ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=col - 1)
        band = ws.cell(1, start, block.title)
        band.font, band.fill = bold, fill
        band.alignment = Alignment(horizontal="left")
        for c in range(start, col):
            ws.cell(1, c).fill = fill
        if b == 0:
            first_property_col = col + 1
        if b < len(blocks) - 1:
            widths[col] = 2         # the empty separator column
            col += 1

    for c, w in widths.items():
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 18
    ws.freeze_panes = ws.cell(FIRST_DATA_ROW, first_property_col)

    key = wb.create_sheet("Column key")
    key.append(["Block", "Column", "Unit", "Description", "Taken from"])
    for c in key[1]:
        c.font, c.border = bold, rule
    for block in blocks:
        for column in block.columns:
            name, unit = _split_unit(column.header)
            key.append([block.title, column.header, unit, column.description, column.source])
    key.append([])
    for line in note.strip().splitlines():
        key.append([line.strip()])
    for row in key.iter_rows(min_row=2):
        for c in row:
            c.font = base
            c.alignment = Alignment(vertical="top", wrap_text=c.column == 4)
    for letter, w in zip("ABCDE", (34, 30, 14, 80, 48)):
        key.column_dimensions[letter].width = w
    key.freeze_panes = "A2"

    wb.properties.creator = "kinetics-ent"
    wb.properties.created = wb.properties.modified = PINNED_TIME
    buf = io.BytesIO()
    wb.save(buf)
    _write_pinned(buf, path)


def _write_pinned(buf: io.BytesIO, path: Path) -> None:
    """Re-zip with fixed timestamps, so identical content gives an identical file."""
    stamp = PINNED_TIME.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zin, \
            zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:\2>)",
                              rb"\g<1>" + stamp + rb"\g<3>", data)
            out = zipfile.ZipInfo(info.filename, date_time=ZIP_TIME)
            out.compress_type = zipfile.ZIP_DEFLATED
            out.external_attr = info.external_attr
            zout.writestr(out, data)


def read_workbook(path: Path, key: str) -> dict[int, dict[str, object]]:
    """A property workbook as {id: {header: value}}, blanks as "".

    Separator columns are skipped. This is the reader `dataset/build_dataset.py` mirrors.
    """
    ws = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = ws.iter_rows(min_row=HEADER_ROW, values_only=True)
    header = next(rows)
    out = {}
    for row in rows:
        record = {h: ("" if v is None else v) for h, v in zip(header, row) if h}
        if record.get(key) not in ("", None):
            out[int(record[key])] = record
    return out
