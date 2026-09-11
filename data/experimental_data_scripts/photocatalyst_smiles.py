# /// script
# requires-python = ">=3.11"
# dependencies = ["rdkit", "numpy", "networkx"]
# ///
"""
Photocatalyst structures -> a `smiles` column on the photocatalyst compound records.

Reads   the table below, which is hand-authored and is the only place in this package
        where a structure is asserted rather than measured or computed
        ../computational_data_raw/photocatalysts/PC_<n>/S0/opt_geometry.xyz
        ../computational_data_raw/photocatalysts/PC_<n>/S0/opt_input.inp
Writes  ../experimental_data_raw/compound_records/compound_records_photocatalysts.csv
        (the existing file, with `smiles` appended; no row is added or removed)

WHY THIS FILE EXISTS AT ALL
    Every other column in this package is read out of an instrument file or a
    calculation. A structure is neither. The 31 photocatalysts are recorded in the
    source spreadsheets only as names -- `[Ir(dF(CF3)ppy)2(dtbbpy)][PF6]`,
    `Ir_dCF3_CF3_ppy2_d55_OMe_bpy` -- and a name is not a machine-readable structure.
    The ML model in ../../ml/ takes a SMILES and nothing else, so without this column
    the photocatalyst side of every pair is simply absent.

    So these were written by hand. That is a weaker provenance than the rest of the
    package and it is stated plainly here rather than buried: nobody measured these
    strings. What follows is the check that makes them trustworthy anyway.

VALIDATED AGAINST THE DFT GEOMETRY, NOT AGAINST THE NAME
    A SMILES that is a plausible reading of a ligand abbreviation is worth very little.
    `dF(Me)ppy` and `dF(4-Me)ppy` differ by one ring position; `dF(OMe)ppy` and
    `dF(6-OMe)ppy` likewise. PC12/PC25 and PC20/PC21 are pairs of structural isomers
    with identical formulas, so a formula check alone cannot tell them apart.

    Every SMILES is therefore checked against `S0/opt_geometry.xyz`, the optimized
    ground-state geometry of the species that was actually computed, on four points:

      1. molecular formula, hydrogens included, equals the element counts in the xyz
      2. heavy-atom count equals the heavy-atom count of the xyz
      3. net formal charge equals the charge on the `*xyzfile <charge> <mult>` line of
         the matching opt_input.inp
      4. the element-labeled heavy-atom connectivity graph of the SMILES is isomorphic
         to the connectivity graph derived from the interatomic distances in the xyz

    Check 4 is the one that does the work. It is what separates "a molecule with this
    formula" from "the molecule that was computed": it pins every ring position, every
    substitution pattern and every metal-ligand contact. Bond orders are deliberately
    not compared -- a geometry does not determine them -- but with the hydrogen count
    fixed by check 1 the remaining freedom is negligible.

    Any failure is fatal. A partial file would be worse than none, because a wrong
    photocatalyst structure does not announce itself downstream; it just moves the
    predicted rate.

THE CONVENTION, APPLIED UNIFORMLY
    Donor-pair (ionic) model, which is what RDKit's organometallic guidance recommends
    and what keeps one rule across two metals and three coordination motifs:

      * every metal-ligand bond is a dative bond, written `->` toward the metal
      * the metal carries a formal charge equal to its oxidation state: [Ir+3], [Ru+2]
      * an anionic donor carries its own charge: the cyclometalated aryl carbon is
        [c-], and an N-heterocyclic carbene is written in its imidazolium-ylide form,
        [n+] ... [c-], with the ylide carbon as the donor

    The net charge then falls out rather than being asserted, and check 3 tests it:
    [Ir(C^N)2(N^N)]+ is +3 - 2 = +1, Ir(C^N)3 is +3 - 3 = 0, [Ru(N^N)3]2+ is +2, and
    the tris-carbene complexes are +3 + 3*(+1 -1 -1) = 0.

    The counter-ion is not part of the structure. PF6- is a spectator, it is not in the
    DFT model, and the species the model has to see is the cation.

    Stereochemistry is not encoded. These are octahedral tris-chelates and so are
    chiral (Delta/Lambda), and the homoleptic ones are facial. None of that is written
    into the SMILES: the model consumes a molecular graph, and the experiments were run
    on racemates.

RUN THIS AFTER compound_records.py, NOT BEFORE
    This script adds a column to a file that compound_records.py writes from scratch.
    Re-running that one afterwards drops `smiles` again, silently, because it knows
    nothing about this column. Re-running this one restores it and is idempotent. The
    order is compound_records.py, then this.

TWO PHOTOCATALYSTS HAVE NO ROW TO WRITE TO
    The table holds all 31 and all 31 are validated, but
    compound_records_photocatalysts.csv carries 29. PC30 and PC31 were computed and are
    in the photocatalyst spreadsheet, but they were never screened, so
    properties_photocatalysts.xlsx has no row for them. Rows are not invented here --
    that file's row set belongs to compound_records.py. Their validated SMILES are
    printed at the end so they are not lost.
"""

import collections
import csv
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DFT = ROOT / "computational_data_raw" / "photocatalysts"
RECORDS = ROOT / "experimental_data_raw" / "compound_records" / "compound_records_photocatalysts.csv"

# Covalent radii / Angstrom, Cordero 2008. Used only to decide whether two light atoms
# are bonded, where the distance distribution is bimodal and the cutoff is not delicate.
RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05}
BOND_SCALE = 1.25

# Metal-ligand contacts are not covalent-radius territory. In these 30 complexes the six
# donor atoms sit at 2.00-2.15 A and the next atom in is beyond 2.9 A, so any cutoff in
# that gap gives the same answer. The gap itself is checked rather than assumed.
METALS = {"Ir", "Ru"}
METAL_CUTOFF = 2.60
METAL_GAP = 2.80
N_DONORS = 6

# Hand-authored, one entry per photocatalyst. Ligand fragments repeat verbatim across
# entries on purpose: they are written out in full rather than assembled from pieces, so
# that what is validated is exactly what is read.
#
# Reading one: [Ir+3]%10%11%12(<-L1)(<-L2)<-L3 is the metal opening three ring-closure
# bonds, one per chelate. Each ligand starts at its first donor atom and closes its
# second donor back onto the metal with `->%1n`.
SMILES = {
    # -- heteroleptic [Ir(C^N)2(N^N)]+ and the two Ru(II) polypyridyls -----------------
    # PC1  [Ir(dF(CF3)ppy)2(dtbbpy)]+
    1: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C(F)(F)F)cn->%101)"
       "(<-[c-]1cc(F)cc(F)c1-c1ccc(C(F)(F)F)cn->%111)"
       "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC2  [Ir(ppy)2(dtbbpy)]+
    2: "[Ir+3]%10%11%12(<-[c-]1ccccc1-c1ccccn->%101)"
       "(<-[c-]1ccccc1-c1ccccn->%111)"
       "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC3  fac-Ir(ppy)3
    3: "[Ir+3]%10%11%12(<-[c-]1ccccc1-c1ccccn->%101)"
       "(<-[c-]1ccccc1-c1ccccn->%111)"
       "<-[c-]1ccccc1-c1ccccn->%121",
    # PC4  fac-Ir(dFppy)3
    4: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccccn->%101)"
       "(<-[c-]1cc(F)cc(F)c1-c1ccccn->%111)"
       "<-[c-]1cc(F)cc(F)c1-c1ccccn->%121",
    # PC5  [Ru(dtbbpy)3]2+
    5: "[Ru+2]%10%11%12(<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%101)"
       "(<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%111)"
       "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC6  [Ru(bpz)3]2+, bpz = 2,2'-bipyrazine
    6: "[Ru+2]%10%11%12(<-n1ccncc1-c1cnccn->%101)"
       "(<-n1ccncc1-c1cnccn->%111)"
       "<-n1ccncc1-c1cnccn->%121",
    # PC7  iPr-TXT, 2-isopropyl-9H-thioxanthen-9-one. The one purely organic sensitizer.
    7: "CC(C)c1ccc2sc3ccccc3c(=O)c2c1",
    # PC8  fac-Ir(CF3-pmb)3
    8: "[Ir+3]%10%11%12(<-[c-]1cc(C(F)(F)F)ccc1-n1c2ccccc2[n+](C)[c-]1->%10)"
       "(<-[c-]1cc(C(F)(F)F)ccc1-n1c2ccccc2[n+](C)[c-]1->%11)"
       "<-[c-]1cc(C(F)(F)F)ccc1-n1c2ccccc2[n+](C)[c-]1->%12",
    # PC9  fac-Ir(dF(Me)ppy)3
    9: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%101)"
       "(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%111)"
       "<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%121",
    # PC10 [Ir(dF(CF3)ppy)2(d(5,5)CF3bpy)]+
    10: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C(F)(F)F)cn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccc(C(F)(F)F)cn->%111)"
        "<-n1cc(C(F)(F)F)ccc1-c1ccc(C(F)(F)F)cn->%121",
    # PC11 [Ir(dFppy)2(dtbbpy)]+
    11: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccccn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccccn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC12 [Ir(dF(Me)ppy)2(dtbbpy)]+ -- 5-methylpyridine; isomeric with PC25
    12: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC13 [Ir(dF(CF3)ppy)2(bpy)]+
    13: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C(F)(F)F)cn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccc(C(F)(F)F)cn->%111)"
        "<-n1ccccc1-c1ccccn->%121",
    # PC14 [Ir(dF(Me)ppy)2(d(5,5)CH3bpy)]+
    14: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%111)"
        "<-n1cc(C)ccc1-c1ccc(C)cn->%121",
    # PC15 [Ir(tCF3ppy)2(d(5,5)CH3bpy)]+
    #      tCF3ppy = 2-[2,4-bis(CF3)phenyl]-5-(CF3)pyridine, folder Ir_dCF3_CF3_ppy2_*
    15: "[Ir+3]%10%11%12(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccc(C(F)(F)F)cn->%101)"
        "(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccc(C(F)(F)F)cn->%111)"
        "<-n1cc(C)ccc1-c1ccc(C)cn->%121",
    # PC16 [Ir(dF(Me)ppy)2(d(OMe)bpy)]+ -- 4,4'-dimethoxy-2,2'-bipyridine
    16: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccc(C)cn->%111)"
        "<-n1ccc(OC)cc1-c1cc(OC)ccn->%121",
    # PC17 [Ir(ppy)2(d(OMe)bpy)]+
    17: "[Ir+3]%10%11%12(<-[c-]1ccccc1-c1ccccn->%101)"
        "(<-[c-]1ccccc1-c1ccccn->%111)"
        "<-n1ccc(OC)cc1-c1cc(OC)ccn->%121",
    # PC18 [Ir(tCF3ppy)2(d(5,5)OMebpy)]+
    18: "[Ir+3]%10%11%12(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccc(C(F)(F)F)cn->%101)"
        "(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccc(C(F)(F)F)cn->%111)"
        "<-n1cc(OC)ccc1-c1ccc(OC)cn->%121",
    # PC19 [Ir(dCF3ppy)2(dtbbpy)]+ -- 2-[2,4-bis(CF3)phenyl]pyridine
    19: "[Ir+3]%10%11%12(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccccn->%101)"
        "(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccccn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC20 [Ir(dF(6-OMe)ppy)2(dtbbpy)]+ -- 6-methoxypyridine; isomeric with PC21
    20: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1cccc(OC)n->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1cccc(OC)n->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC21 [Ir(dF(OMe)ppy)2(dtbbpy)]+ -- 5-methoxypyridine
    21: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1ccc(OC)cn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccc(OC)cn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC22 [Ir(tCF3ppy)2(dtbbpy)]+
    22: "[Ir+3]%10%11%12(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccc(C(F)(F)F)cn->%101)"
        "(<-[c-]1cc(C(F)(F)F)cc(C(F)(F)F)c1-c1ccc(C(F)(F)F)cn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC23 [Ir(tetraFppy)2(dtbbpy)]+ -- 2,3,4,5-tetrafluorophenyl
    23: "[Ir+3]%10%11%12(<-[c-]1c(F)c(F)c(F)c(F)c1-c1ccccn->%101)"
        "(<-[c-]1c(F)c(F)c(F)c(F)c1-c1ccccn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC24 [Ir(d(OMe)ppy)2(dtbbpy)]+ -- 2,4-dimethoxyphenyl
    24: "[Ir+3]%10%11%12(<-[c-]1cc(OC)cc(OC)c1-c1ccccn->%101)"
        "(<-[c-]1cc(OC)cc(OC)c1-c1ccccn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC25 [Ir(dF(4-Me)ppy)2(dtbbpy)]+ -- 4-methylpyridine; isomeric with PC12
    25: "[Ir+3]%10%11%12(<-[c-]1cc(F)cc(F)c1-c1cc(C)ccn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1cc(C)ccn->%111)"
        "<-n1ccc(C(C)(C)C)cc1-c1cc(C(C)(C)C)ccn->%121",
    # PC26 [Ir(dFppy)2(MeCN)2]+ -- two nitriles in place of the diimine
    26: "[Ir+3]%10%11(<-[c-]1cc(F)cc(F)c1-c1ccccn->%101)"
        "(<-[c-]1cc(F)cc(F)c1-c1ccccn->%111)(<-N#CC)<-N#CC",
    # PC27 [Ir(ppy)2(phen)]+
    27: "[Ir+3]%10%11%12(<-[c-]1ccccc1-c1ccccn->%101)"
        "(<-[c-]1ccccc1-c1ccccn->%111)"
        "<-n1cccc2ccc3cccn->%12c3c12",
    # -- homoleptic tris-carbene Ir(C^C:)3 --------------------------------------------
    # PC28 fac-Ir(CH3-pmb)3
    28: "[Ir+3]%10%11%12(<-[c-]1cc(C)ccc1-n1c2ccccc2[n+](C)[c-]1->%10)"
        "(<-[c-]1cc(C)ccc1-n1c2ccccc2[n+](C)[c-]1->%11)"
        "<-[c-]1cc(C)ccc1-n1c2ccccc2[n+](C)[c-]1->%12",
    # PC29 fac-Ir(pmb)3, pmb = 1-phenyl-3-methylbenzimidazol-2-ylidene
    29: "[Ir+3]%10%11%12(<-[c-]1ccccc1-n1c2ccccc2[n+](C)[c-]1->%10)"
        "(<-[c-]1ccccc1-n1c2ccccc2[n+](C)[c-]1->%11)"
        "<-[c-]1ccccc1-n1c2ccccc2[n+](C)[c-]1->%12",
    # PC30 fac-Ir(p-F-pmb)3 -- computed and in the photocatalyst sheet, never screened
    30: "[Ir+3]%10%11%12(<-[c-]1cc(F)ccc1-n1c2ccccc2[n+](C)[c-]1->%10)"
        "(<-[c-]1cc(F)ccc1-n1c2ccccc2[n+](C)[c-]1->%11)"
        "<-[c-]1cc(F)ccc1-n1c2ccccc2[n+](C)[c-]1->%12",
    # PC31 fac-Ir(pmi)3 -- imidazolylidene, not benzimidazolylidene; never screened
    31: "[Ir+3]%10%11%12(<-[c-]1ccccc1-n1cc[n+](C)[c-]1->%10)"
        "(<-[c-]1ccccc1-n1cc[n+](C)[c-]1->%11)"
        "<-[c-]1ccccc1-n1cc[n+](C)[c-]1->%12",
}

ELEMENT_ORDER = ["C", "H", "N", "O", "F", "S", "Ir", "Ru"]


def formula(counts):
    """Element counts as a string, ordered by ELEMENT_ORDER so the two sides compare."""
    unknown = set(counts) - set(ELEMENT_ORDER)
    if unknown:
        sys.exit(f"element not in ELEMENT_ORDER: {sorted(unknown)}")
    return "".join(f"{e}{counts[e]}" for e in ELEMENT_ORDER if counts.get(e))


def parse_formula(text):
    """'C42H34F10IrN4+' -> Counter. The trailing charge is dropped, not counted."""
    body = re.sub(r"[+-]\d*$", "", text)
    return collections.Counter(
        {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", body)}
    )


def read_geometry(pc_id):
    """-> (elements, coordinates) from the optimized S0 geometry."""
    path = DFT / f"PC_{pc_id}" / "S0" / "opt_geometry.xyz"
    if not path.exists():
        sys.exit(f"missing DFT geometry: {path}")
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    elements, coordinates = [], []
    for line in lines[2:2 + n]:
        token = line.split()
        elements.append(token[0])
        coordinates.append([float(x) for x in token[1:4]])
    if len(elements) != n:
        sys.exit(f"PC{pc_id}: header says {n} atoms, file has {len(elements)}")
    return elements, np.array(coordinates)


def read_charge(pc_id):
    """The charge on the `*xyzfile <charge> <mult>` line of the same optimization."""
    path = DFT / f"PC_{pc_id}" / "S0" / "opt_input.inp"
    if not path.exists():
        sys.exit(f"missing DFT input: {path}")
    match = re.search(r"^\s*\*\s*xyzfile\s+(-?\d+)\s+(\d+)", path.read_text(), re.M)
    if not match:
        sys.exit(f"PC{pc_id}: no `*xyzfile <charge> <mult>` line in {path.name}")
    return int(match.group(1))


def geometry_graph(pc_id, elements, coordinates):
    """Heavy-atom connectivity implied by the interatomic distances.

    Light-atom bonds come from scaled covalent radii. Metal-ligand contacts get their own
    cutoff, and the assumption behind that cutoff -- a clean gap between the donor shell
    and everything else -- is checked here rather than trusted, because it is the only
    part of this perception that could plausibly go wrong.
    """
    n = len(elements)
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)

    graph = nx.Graph()
    heavy = [i for i in range(n) if elements[i] != "H"]
    for i in heavy:
        graph.add_node(i, el=elements[i])

    for a, i in enumerate(heavy):
        for j in heavy[a + 1:]:
            if elements[i] in METALS or elements[j] in METALS:
                continue
            if distance[i, j] < BOND_SCALE * (RCOV[elements[i]] + RCOV[elements[j]]):
                graph.add_edge(i, j)

    for i in range(n):
        if elements[i] not in METALS:
            continue
        ordered = sorted(range(n), key=lambda j: distance[i, j])[1:]
        donors = [j for j in ordered if distance[i, j] < METAL_CUTOFF]
        beyond = next(j for j in ordered if distance[i, j] >= METAL_CUTOFF)
        if distance[i, beyond] < METAL_GAP:
            sys.exit(f"PC{pc_id}: no clean coordination shell around {elements[i]} -- "
                     f"{len(donors)} contacts below {METAL_CUTOFF} A and the next at "
                     f"{distance[i, beyond]:.2f} A, so the cutoff decides the answer")
        if len(donors) != N_DONORS:
            sys.exit(f"PC{pc_id}: {elements[i]} has {len(donors)} donors, "
                     f"expected {N_DONORS}")
        for j in donors:
            graph.add_edge(i, j)

    return graph


def smiles_graph(mol):
    """The same graph from the SMILES: heavy atoms, elements, bonds, no bond orders."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), el=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return graph


def validate(pc_id, smiles):
    """-> (canonical_smiles, [problems]). Every problem is a reason to stop."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ["SMILES does not parse"]

    elements, coordinates = read_geometry(pc_id)
    reference = collections.Counter(elements)
    problems = []

    mine = parse_formula(CalcMolFormula(mol))
    if formula(mine) != formula(reference):
        problems.append(f"formula {formula(mine)} != DFT {formula(reference)}")

    heavy_dft = sum(v for e, v in reference.items() if e != "H")
    if mol.GetNumAtoms() != heavy_dft:
        problems.append(f"{mol.GetNumAtoms()} heavy atoms != DFT {heavy_dft}")

    charge_dft = read_charge(pc_id)
    if Chem.GetFormalCharge(mol) != charge_dft:
        problems.append(f"charge {Chem.GetFormalCharge(mol):+d} != DFT {charge_dft:+d}")

    # Only worth attempting once the cheap checks agree; a formula mismatch guarantees
    # a graph mismatch and the message would say nothing new.
    if not problems:
        matched = nx.is_isomorphic(
            smiles_graph(mol),
            geometry_graph(pc_id, elements, coordinates),
            node_match=lambda a, b: a["el"] == b["el"],
        )
        if not matched:
            problems.append("connectivity does not match the DFT geometry")

    return Chem.MolToSmiles(mol), problems


def main():
    if not RECORDS.exists():
        sys.exit(f"missing input: {RECORDS}")

    print(f"validating {len(SMILES)} hand-authored structures against "
          f"computational_data_raw/photocatalysts/PC_<n>/S0/\n")

    canonical, failed = {}, []
    for pc_id in sorted(SMILES):
        smiles, problems = validate(pc_id, SMILES[pc_id])
        elements, _ = read_geometry(pc_id)
        counts = collections.Counter(elements)
        print(f"  PC{pc_id:<3} {'ok  ' if not problems else 'FAIL'}  "
              f"{formula(counts):<22} q={read_charge(pc_id):+d}")
        for problem in problems:
            print(f"          {problem}")
        if problems:
            failed.append(pc_id)
        else:
            canonical[pc_id] = smiles

    if failed:
        sys.exit(f"\n{len(failed)} structure(s) do not match their DFT geometry: "
                 f"{failed}\nnothing written -- a wrong photocatalyst structure does "
                 f"not announce itself downstream, it just moves the predicted rate.")

    with RECORDS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{RECORDS.name} has no rows")
    columns = list(rows[0].keys())
    if "smiles" not in columns:
        columns.append("smiles")

    unmatched = []
    for row in rows:
        pc_id = int(row["pc_id"])
        row["smiles"] = canonical.get(pc_id, "")
        if pc_id not in canonical:
            unmatched.append(pc_id)

    with RECORDS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nall {len(canonical)} validated: formula, heavy atoms, charge, connectivity")
    print(f"{len(rows)} rows -> {RECORDS.name}, "
          f"{sum(1 for r in rows if r['smiles'])} with a SMILES")
    if unmatched:
        print(f"  NOTE: no structure for {unmatched}")

    spare = sorted(set(canonical) - {int(r["pc_id"]) for r in rows})
    if spare:
        print(f"\n{len(spare)} validated structure(s) have no row in {RECORDS.name}. "
              f"They were computed but never screened, so\nproperties_photocatalysts."
              f"xlsx has no row for them and none is invented here:")
        for pc_id in spare:
            print(f"  PC{pc_id}  {canonical[pc_id]}")


if __name__ == "__main__":
    main()
