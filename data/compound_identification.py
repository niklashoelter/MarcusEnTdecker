"""
Which compounds are in the data, what they are, and how they are numbered.

The one source is compound_identification.xlsx, next to this file: a sheet per compound
class with the ids, the photocatalyst id mapping, the names, the quencher SMILES and the
molecular weights. Everything else in the package refers to a compound by id only.

Photocatalysts have two ids. Up to and including the extraction scripts, a photocatalyst
is its OLD id, the number it had during the screen: raw file names
(stern_volmer/PC10_QBatch1.csv, pc_tcspc/PC4.txt, computational_data_raw/photocatalysts/
PC_4/) and the intermediate tables all use it, and the extraction scripts only ever ask
IN_DATA whether an old id is kept. NEW_ID enters at the end of the pipeline and nowhere
earlier: properties_exp.py and properties_calc.py write `old_id` and `new_id` into the
processed workbooks, and build_exp_dataset.py carries both into exp_dataset. Everything
downstream names a photocatalyst by its new id, PC1-PC27.

Not a pipeline stage; imported by the scripts that need it. Standard library only, so
that importing it adds no dependency to any script -- which is the whole reason this
module exists beside the workbook rather than the workbook being read directly. The data
is the xlsx; this file is the reader, plus the one fact that is not a row of it
(CROSSED_OUT, the photocatalysts that were delivered but are not part of the screen).
Every script asks this module, so there is exactly one answer to what a compound is.
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
OLD_ID = {new: old for old, new in NEW_ID.items()}
# The old ids in the data. All that the extraction scripts use.
IN_DATA = frozenset(NEW_ID)

# Crossed out in the photocatalyst sheet the screen was run from: measured or computed at
# some point, but not part of the screen and not in the data. They are not rows of the
# workbook because the workbook is a table of the compounds that ARE in the data; they are
# here because the curation script has to recognise and skip their delivered job folders.
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
