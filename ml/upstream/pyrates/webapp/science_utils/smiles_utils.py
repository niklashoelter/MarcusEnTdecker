from __future__ import annotations

from typing import Optional
from rdkit import Chem  # type: ignore


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

def kekulize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, kekuleSmiles=True) if mol else None

def is_single_fragment_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    fragments = Chem.GetMolFrags(mol)
    return len(fragments) == 1