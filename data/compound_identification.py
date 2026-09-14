"""
Compound ids, names, SMILES and molar masses.

Reads compound_identification.xlsx next to this file, which has one sheet per compound
class. All other scripts refer to compounds by id and look them up here.

Photocatalysts have two ids. Raw files and intermediate tables use the old id from the
screen (stern_volmer/PC10_QBatch1.csv, pc_tcspc/PC4.txt,
computational_data_raw/photocatalysts/PC_4/). The new id, PC1-PC27, enters at the
processed stage: properties_exp.py and properties_calc.py write `old_id` and `new_id`
into the workbooks, and build_exp_dataset.py carries both into exp_dataset.

Not a pipeline stage. Uses the standard library only, so importing it adds no dependency.
"""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

WORKBOOK = Path(__file__).resolve().parent / "compound_identification.xlsx"

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _sheets(path):
    """{sheet name: [row dicts keyed by the header row]} for every sheet."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t"))
                       for si in root.findall("m:si", _NS)]
        rels = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels.iter(f"{_REL}Relationship")}
        book = ElementTree.fromstring(z.read("xl/workbook.xml"))
        found = {}
        for s in book.find("m:sheets", _NS):
            part = target[s.get(f"{{{_NS['r']}}}id")].lstrip("/")
            part = part if part.startswith("xl/") else "xl/" + part
            rows = []
            for row in ElementTree.fromstring(z.read(part)).iter(f"{{{_NS['m']}}}row"):
                cells = {}
                for c in row.findall("m:c", _NS):
                    column = re.match(r"[A-Z]+", c.get("r")).group()
                    v = c.find("m:v", _NS)
                    if c.get("t") == "s":
                        cells[column] = strings[int(v.text)]
                    elif c.get("t") == "inlineStr":
                        cells[column] = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
                    elif v is not None:
                        number = float(v.text)
                        cells[column] = int(number) if number.is_integer() else number
                rows.append(cells)
            header = rows[0] if rows else {}
            found[s.get("name")] = [{header[k]: cells.get(k) for k in header}
                                    for cells in rows[1:] if cells]
        return found


_BOOK = _sheets(WORKBOOK)

# old id -> {"new_id", "old_id", "name", "M / g mol-1"}, in new-id order
PHOTOCATALYSTS = {int(r["old_id"]): r for r in sorted(_BOOK["Photocatalysts"],
                                                       key=lambda r: r["new_id"])}
# quencher id -> {"quencher_id", "name", "smiles", "M / g mol-1"}
QUENCHERS = {int(r["quencher_id"]): r for r in sorted(_BOOK["Quenchers"],
                                                      key=lambda r: r["quencher_id"])}

NEW_ID = {old: int(r["new_id"]) for old, r in PHOTOCATALYSTS.items()}
# Old ids of the photocatalysts in the data.
IN_DATA = frozenset(NEW_ID)

# Old ids of photocatalysts that are not in the data. photocatalyst_curate.py skips
# their job folders.
CROSSED_OUT = {
    7: "iPr-TXT",
    26: "[Ir(dFppy)2(MeCN)2]PF6",
    30: "[Ir(p-F-pmb)3]",
    31: "[Ir(pmi)3]",
}

assert not set(NEW_ID) & set(CROSSED_OUT), "a crossed-out id is also in the data"

assert sorted(NEW_ID.values()) == list(range(1, len(NEW_ID) + 1)), "new ids must be 1..n"
assert sorted(QUENCHERS) == list(range(1, len(QUENCHERS) + 1)), "quencher ids must be 1..n"
assert all(r["name"] and r["smiles"] for r in QUENCHERS.values()), "quencher without identity"
