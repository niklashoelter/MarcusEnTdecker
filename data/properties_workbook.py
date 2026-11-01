"""
The layout shared by the five property workbooks.

    experimental_data_processed/photocatalyst_properties_exp.xlsx
    experimental_data_processed/quencher_properties_exp.xlsx
    computational_data_processed/photocatalyst_properties_calc.xlsx
    computational_data_processed/quencher_properties_calc.xlsx
    computational_data_processed/quencher_properties_ml.xlsx

One row per compound. Columns grouped in blocks (identity first), separated by an empty
column. Row 1: block names; row 2: headers as `quantity / unit`. Sheet `Column key`:
unit, description and source table of each column. Timestamps are fixed.
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

from openpyxl import Workbook
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
    header: str                         # "tau0 / ns", unit after " / "
    description: str
    source: str                         # "experimental_data_processed/lifetimes_tcspc.csv: tau0_ns"
    value: Callable[[int], object]      # compound id -> cell value


@dataclass(frozen=True)
class Block:
    title: str
    columns: list[Column]


def read_table(path: Path, key: str) -> dict[int, dict[str, str]]:
    """An intermediate CSV as {id: row}. Exits if the file is missing."""
    if not path.exists():
        sys.exit(f"missing input: {path}\nrun the script that writes it first")
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(row[key]): row for row in csv.DictReader(fh)}


def typed(text):
    """A CSV field as int, float, str, or None when blank.

    Numbers with leading zeros, such as the sample label `009`, stay text.
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
    """A column getter: field `name` of the row for an id, None if absent."""
    return lambda i: typed((table.get(i) or {}).get(name))


def _split_unit(header: str) -> tuple[str, str]:
    name, _, unit = header.partition(" / ")
    return name, unit


def write_workbook(path: Path, sheet: str, ids: list[int], blocks: list[Block],
                   note: str) -> None:
    """Write the property sheet and its column key."""
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
            _, unit = _split_unit(column.header)
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
    """Re-zip with fixed timestamps."""
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
