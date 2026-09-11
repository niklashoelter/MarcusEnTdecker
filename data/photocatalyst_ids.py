"""
Which photocatalysts are in the data, and how they are numbered.

Up to and including the extraction scripts, a photocatalyst is its OLD id, the number it
had during the screen: raw file names (stern_volmer/PC10_QBatch1.csv, pc_tcspc/PC4.txt,
computational_data_raw/photocatalysts/PC_4/) and the intermediate tables all use it, and
the extraction scripts only ever ask IN_DATA whether an old id is kept.

NEW_ID enters at the end of the pipeline and nowhere earlier: properties_exp.py and
properties_calc.py write `old_id` and `new_id` into the processed workbooks, and
build_exp_dataset.py carries both into exp_dataset. Everything downstream of that -- the
benchmark, the text -- names a photocatalyst by its NEW id, PC1-PC27, consecutive and
without gaps.

The kept set and the new order are those of the photocatalyst sheet the screen was run
from (Photocatalysts/photocatalysts.xlsx): every row not crossed out, in row order, which
puts the three Ir(pmb)3-type complexes, measured at 380 nm, last. Old ids 7, 26, 30 and
31 are crossed out there and are not part of the data.

Not a pipeline stage; imported by the scripts that need it.
"""

# old id -> new id, in the row order of the photocatalyst sheet
NEW_ID = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16,
    19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23,
    27: 24,
    8: 25, 28: 26, 29: 27,
}

# Crossed out in the photocatalyst sheet: measured or computed at some point, but not in
# the screen, and not in the data.
CROSSED_OUT = {
    7: "iPr-TXT",
    26: "[Ir(dFppy)2(MeCN)2]PF6",
    30: "[Ir(p-F-pmb)3]",
    31: "[Ir(pmi)3]",
}

# The old ids in the data. All that the extraction scripts use.
IN_DATA = frozenset(NEW_ID)

OLD_ID = {new: old for old, new in NEW_ID.items()}

assert sorted(NEW_ID.values()) == list(range(1, len(NEW_ID) + 1))
assert not set(NEW_ID) & set(CROSSED_OUT)
