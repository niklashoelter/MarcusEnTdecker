"""
Compound ids, names, SMILES and molar masses.

Reads compound_identification.xlsx next to this file, one sheet per compound class.

Photocatalysts are keyed by their new id. Raw experimental files keep the old id
(stern_volmer/PC10_QBatch1.csv, pc_tcspc/PC4.txt, pipetting maps); their readers map it
to the new id with NEW_ID. No other table carries old ids.
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

# new id -> {"new_id", "old_id", "name", "M / g mol-1"}
PHOTOCATALYSTS = {int(r["new_id"]): r for r in sorted(_BOOK["Photocatalysts"],
                                                       key=lambda r: r["new_id"])}
# quencher id -> {"quencher_id", "name", "smiles", "M / g mol-1"}
QUENCHERS = {int(r["quencher_id"]): r for r in sorted(_BOOK["Quenchers"],
                                                      key=lambda r: r["quencher_id"])}

# old id (raw experimental files and job trees) -> new id
NEW_ID = {int(r["old_id"]): new for new, r in PHOTOCATALYSTS.items()}

assert sorted(PHOTOCATALYSTS) == list(range(1, len(PHOTOCATALYSTS) + 1)), "new ids must be 1..n"
assert len(NEW_ID) == len(PHOTOCATALYSTS), "old ids must be unique"
assert sorted(QUENCHERS) == list(range(1, len(QUENCHERS) + 1)), "quencher ids must be 1..n"
assert all(r["name"] and r["smiles"] for r in QUENCHERS.values()), "quencher without identity"
