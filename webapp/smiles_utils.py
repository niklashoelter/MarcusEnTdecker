from __future__ import annotations

from typing import Optional
from rdkit import Chem  # type: ignore

from ml.utils.graph_construction import ATOM_FEATURES, UnknownAtomError, smiles_to_graph


def is_valid_smiles(smiles: str) -> bool:
    """
    Return True if the SMILES is valid (RDKit-parseable).
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Return a canonical SMILES string if possible. Returns None if canonicalization fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def is_single_fragment_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    fragments = Chem.GetMolFrags(mol)
    return len(fragments) == 1


def vocabulary_error(smiles: str) -> Optional[str]:
    """
    Error message if the model cannot encode the molecule (UnknownAtomError), else None.
    """
    try:
        smiles_to_graph(smiles, use_explicit_hydrogens=False, use_stereochemistry=True)
    except UnknownAtomError:
        mol = Chem.MolFromSmiles(smiles)
        unknown = sorted({atom.GetSymbol() for atom in mol.GetAtoms()
                          if atom.GetAtomicNum() not in ATOM_FEATURES['atomic_num']})
        return "contains an element outside the model's vocabulary: " + ", ".join(unknown)
    return None