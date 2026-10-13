# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.2", "pandas==2.3.3"]
# ///
"""
Rigid, flexible and dissociative quenchers.

Reads   ../../data/computational_data_processed/quencher_geometry_S0_T1.csv
        ../../data/compound_identification.xlsx   (names and SMILES)
        ../../data/exp_dataset_ent.csv            (quencher parameters, all 48 quenchers)
Writes  results/quencher_classes/quencher_classes.csv   one row per quencher

    uv run benchmark/further_analysis/quencher_classes.py

Each quencher is assigned by hand (CLASSES) from its structure and from the change
between its S0 and T1 DFT minima (quencher_geometry_S0_T1.csv): rigid quenchers keep a
planar or nearly unchanged T1 geometry, flexible ones twist about a C=C or C=N bond, or
about the C-N bond of a thioamide, and dissociative ones elongate or break a bond in T1.
A cut at lambda_T1_S0 (DFT) = LAMBDA_CUT, below which a quencher counts as rigid, is the
cross-check; where cut and geometry disagree, the geometry decides and the note starts
with AMBIGUOUS. The note also flags single-bond groups that rotate into the ring plane,
ring puckers, and N-O reagents whose DFT triplet keeps the N-O bond.

Columns
    id, name, smiles      quencher
    functional_class      functional group
    proposed_class        rigid, flexible or dissociative
    evidence              the note, then the DFT parameters, DvTE (p = 0.008, normal)
                          - E_T, E_T,ML - E_T,DFT (kcal/mol) and the geometry change
    ambiguous             1 where the note is flagged AMBIGUOUS
    dtheta / deg          largest change of the deviation from coplanarity about a
                          bond between two heavy atoms, S0 -> T1 (the larger of
                          dtors_double_deg and dtors_any_heavy_deg)
    dr / %                largest relative change of an S0 bond length in T1
                          (max_stretch_pct)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

import pairs  # noqa: E402

DATA = pairs.HERE.parent / "data"
GEOMETRY = DATA / "computational_data_processed" / "quencher_geometry_S0_T1.csv"
RESULTS = HERE / "results" / "quencher_classes"
LAMBDA_CUT = 20.0       # kcal/mol, lambda_T1_S0 (DFT) cross-check
FLAG = "AMBIGUOUS"
CLASS_NAMES = ("rigid", "flexible", "dissociative")

# quencher id: (functional class, class, structural and geometric note)
CLASSES = {
    1: ("aza-arene", "rigid", "planar T1, no torsion"),
    2: ("styrene", "flexible", "T1 C=C twisted 0->88 deg"),
    3: ("nitroarene", "rigid", "AMBIGUOUS: nitro rotates 64->9 deg into the ring plane in T1 (single-bond torsion, not a C=C twist); lS0T1 32 high, lT1S0 below cut"),
    4: ("heteroarene (benzothiophene)", "rigid", "planar T1"),
    5: ("aza-arene", "rigid", "planar T1"),
    6: ("aza-arene", "rigid", "planar T1"),
    7: ("coumarin (cyclic enone/lactone)", "rigid", "C=C held in ring, planar T1"),
    8: ("oxime ester (N-O, oxime carbonate)", "flexible", "AMBIGUOUS: T1 twists about C=N 1->70 deg, N-O intact (1.42->1.37 A); EnT N-O homolysis reagent class, so dissociative by intent but not in DFT"),
    9: ("biaryl (2-arylthiophene)", "rigid", "AMBIGUOUS: aryl-thienyl torsion planarizes 47->0 deg (biphenyl-like; Paton's biphenyl = rigid) but lT1S0 21.7 crosses the 20 cut"),
    10: ("disulfide (S-S)", "dissociative", "T1 S-S 2.05->5.27 A (broken)"),
    11: ("disulfide (S-S)", "dissociative", "T1 S-S 2.03->2.73 A (+35 %)"),
    12: ("1,2-azaborine (BN-arene)", "rigid", "planar T1; ML E_T 9.8 below DFT (ML issue, not a class issue)"),
    13: ("maleimide (cyclic enone/imide)", "rigid", "C=C held in five-membered ring, planar T1"),
    14: ("acrylate (methacrylate)", "flexible", "T1 C=C twisted 0->90 deg"),
    15: ("cyclic alkene (indene)", "rigid", "C=C in five-membered ring, planar T1 (Paton's indene = rigid)"),
    16: ("2-pyridone", "rigid", "planar T1"),
    17: ("diselenide (Se-Se)", "dissociative", "T1 Se-Se 2.30->2.99 A (+30 %); rerun from S0 broke Se-C instead"),
    18: ("4,5-dihydro-1,2,4-oxadiazole (cyclic N-O, C=N)", "rigid", "AMBIGUOUS: ring C=N stretched (+12 %), phenyl planarizes 21->1 deg, N-O intact; lT1S0 21.4 just over the 20 cut"),
    19: ("oxime ester (N-O, oxime thiocarbonate)", "flexible", "AMBIGUOUS: T1 twists about C=N 3->71 deg, N-O intact (1.40->1.38 A); EnT N-O homolysis reagent class, so dissociative by intent but not in DFT"),
    20: ("arene (naphthalene)", "rigid", "planar T1"),
    21: ("alkyne (aryl propiolate)", "rigid", "small lambda; Paton's diphenylacetylene = rigid"),
    22: ("N-acyloxy quinazolinone (N-O ester)", "rigid", "AMBIGUOUS: T1 on quinazolinone C=N (+10 %), N-O intact (1.37->1.38 A), side-chain/phenyl reorientation; lT1S0 27.2 says flexible; N-O reagent by intent"),
    23: ("heteroaryl boronate (Bpin)", "rigid", "planar T1 chromophore"),
    24: ("heteroaryl boronate (Bpin)", "rigid", "planar T1 chromophore"),
    25: ("heteroaryl bromide", "rigid", "AMBIGUOUS: T1 bends out of plane at C-Br (ring torsion 0->37 deg, C-Br +4 %); lS0T1 35.5; lT1S0 20.6 just over cut. Radical anion cleaves C-Br (red state, not T1)"),
    26: ("heteroaryl bromide", "rigid", "AMBIGUOUS: T1 bends out of plane at C-Br (0->36 deg, C-Br 1.85->2.01 A, +8 %); lS0T1 36.4, lT1S0 25.7 over cut. Radical anion cleaves C-Br"),
    27: ("heteroaryl bromide (dibromothiophene)", "rigid", "AMBIGUOUS (mild): T1 ring pucker 0->25 deg, C-Br intact; lS0T1 36.2, lT1S0 19.8 just under cut. Radical anion cleaves C-Br"),
    28: ("heteroaryl aryl ketone", "rigid", "near-planar T1, small lambda"),
    29: ("isoxazol-5(4H)-one (cyclic N-O, C=N)", "rigid", "AMBIGUOUS: ring C=N +14 %, N-O intact (1.41->1.35 A); lT1S0 21.5 just over cut; isoxazolone N-O photolability by intent"),
    30: ("stilbene (E)", "flexible", "T1 C=C twisted 0->85 deg"),
    31: ("isoxazolidine-3,5-dione (cyclic N-O)", "rigid", "AMBIGUOUS: ring puckers 0->37 deg, N-O intact (1.40->1.41 A); lS0T1 37, lT1S0 15; N-O reagent by intent"),
    32: ("thioamide", "flexible", "AMBIGUOUS: T1 twists about the thioamide C-N bond 3->86 deg (torsional), but lT1S0 18.8 below cut; lS0T1 41.8"),
    33: ("thioamide (N-aryl)", "flexible", "AMBIGUOUS: T1 C(S)-N +9 % and N-aryl/C-N reorientation (89->5 deg); lT1S0 16.8 below cut; lS0T1 32.9"),
    34: ("oxime ether (C=N, N-O)", "flexible", "T1 C=N twisted 2->88 deg, N-O intact"),
    35: ("stilbene (Z)", "flexible", "T1 C=C twisted 4->85 deg"),
    36: ("heteroaryl ketone", "rigid", "planar T1, small lambda"),
    37: ("styrene (gem-difluorovinyl, aryl bromide)", "flexible", "T1 C=C twisted 0->64 deg"),
    38: ("1,3-diene", "flexible", "T1 terminal C=C twisted 0->88 deg (allyl-methylene)"),
    39: ("aryl azole (2-phenylthiazole)", "rigid", "phenyl planarizes 18->0 deg, small lambda"),
    40: ("aryl azole (2-phenyloxazole)", "rigid", "planar T1, small lambda"),
    41: ("styrene (alpha-methyl)", "flexible", "T1 C=C twisted 2->86 deg"),
    42: ("heteroaryl aryl ketone", "rigid", "small lambda"),
    43: ("heteroaryl ketone", "rigid", "planar T1, small lambda"),
    44: ("heteroaryl ketone (aza)", "rigid", "planar T1, small lambda"),
    45: ("vinyl selenoxide (Se-C, Se=O)", "dissociative", "T1 Se-C(vinyl) 1.93->4.03 A (broken); rerun from S0 broke Se=O instead"),
    46: ("2H-azirine (C=N in three-membered ring)", "rigid", "AMBIGUOUS: T1 C=N +15 % and twisted 0->25 deg, phenyl 0->45 deg; lT1S0 24.5 over cut; azirine ring opening known (ox state ring C-C +19 %)"),
    47: ("aza-arene ester", "rigid", "AMBIGUOUS (mild): ester rotates 69->0 deg into plane in T1 (conformational), lS0T1 25.7 but lT1S0 8.2; DvTE 9.3 below E_T,DFT and ML 6.4 below DFT"),
    48: ("cyclic enone (cyclohexenone)", "rigid", "DFT T1 near planar (C=C 2->0 deg), small lambda; Paton's cyclohex-2-en-1-one = rigid; note twisted pi,pi* enone triplets are known experimentally"),
}


def evidence(note, parameters, geometry) -> str:
    """The note followed by the DFT parameters and the geometry change of one quencher."""
    et = parameters["q_E_T_dft / kcal mol-1"]
    l1, l2 = parameters[pairs.q_lambda_s0_t1("dft")], parameters[pairs.q_lambda_t1_s0("dft")]
    dvte = parameters[pairs.q_dvte("0.008", "normal")] - et
    et_ml = parameters["q_E_T_ml / kcal mol-1"]
    cut = "flexible/diss" if l2 >= LAMBDA_CUT else "rigid"
    return (f"{note}. DFT: E_T {et:.2f}, lS0T1 {l1:.2f}, lT1S0 {l2:.2f} (cut {LAMBDA_CUT:.0f} -> {cut}); "
            f"DvTE(0.8%)-E_T {dvte:.2f}; E_T,ML-E_T,DFT {et_ml - et:.2f} kcal/mol; "
            f"max S0-bond stretch in T1 {geometry['max_stretch_pct']:.1f} % ({geometry['stretch_bond']}); "
            f"heavy-atom RMSD {geometry['rmsd_heavy']:.2f} A")


def classify(quenchers, parameters, geometry):
    """One row per quencher of `quenchers` ({id: {"name", "smiles"}}); parameters and
    geometry are indexed by quencher id and by "Q<id>"."""
    rows = []
    for qid, record in quenchers.items():
        functional, cls, note = CLASSES[qid]
        g = geometry.loc[f"Q{qid}"]
        rows.append({"id": f"Q{qid}", "name": record["name"], "smiles": record["smiles"],
                     "functional_class": functional, "proposed_class": cls,
                     "evidence": evidence(note, parameters.loc[qid], g),
                     "ambiguous": int(FLAG in note),
                     "dtheta / deg": max(g["dtors_double_deg"], g["dtors_any_heavy_deg"]),
                     "dr / %": g["max_stretch_pct"]})
    return pd.DataFrame(rows)


def cut_disagreements(table, parameters):
    """(rigid quenchers at or above the lambda cut, flexible ones below it)."""
    lam = table["id"].str[1:].astype(int).map(parameters[pairs.q_lambda_t1_s0("dft")])
    over = lam >= LAMBDA_CUT
    return (list(table.loc[(table["proposed_class"] == "rigid") & over, "id"]),
            list(table.loc[(table["proposed_class"] == "flexible") & ~over, "id"]))


def main():
    sys.path.insert(0, str(DATA))
    from compound_identification import QUENCHERS

    if not GEOMETRY.exists():
        raise FileNotFoundError(f"{GEOMETRY} not found; run "
                                "data/computational_data_scripts/quencher_geometry.py first")
    parameters = pd.read_csv(pairs.DATASET).groupby(pairs.Q_ID).first()
    geometry = pd.read_csv(GEOMETRY, keep_default_na=False).set_index("id")
    table = classify(QUENCHERS, parameters, geometry)

    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "quencher_classes.csv", index=False, lineterminator="\n")

    counts = table["proposed_class"].value_counts()
    print(", ".join(f"{counts.get(c, 0)} {c}" for c in CLASS_NAMES))
    for c in CLASS_NAMES:
        print(f"  {c:12s} {' '.join(table.loc[table['proposed_class'] == c, 'id'])}")
    rigid_over, flexible_under = cut_disagreements(table, parameters)
    print(f"  ambiguous    {' '.join(table.loc[table['ambiguous'] == 1, 'id'])}")
    print(f"  rigid with lambda_T1_S0 >= {LAMBDA_CUT:.0f}: {' '.join(rigid_over)}; "
          f"flexible below: {' '.join(flexible_under)}")
    print(f"\n-> {RESULTS.relative_to(pairs.HERE.parent)}")


if __name__ == "__main__":
    main()
