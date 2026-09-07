from __future__ import annotations
import numpy as np
from typing import Iterable, List

# Conversion factors mapped to kcal/mol per unit
# Values based on CODATA/commonly used constants
# - 1 Hartree = 627.5094740631 kcal/mol
# - 1 eV = 23.060548868 kcal/mol
# - 1 kJ/mol = 0.2390057361376673 kcal/mol (since 1 kcal = 4.184 kJ)
FACTOR_TO_KCAL_PER_MOL = {
    "kcal/mol": 1.0,
    "kJ/mol": 0.2390057361376673,
    "eV": 23.060548868,
    "Hartree": 627.5094740631,
}

SUPPORTED_UNITS: List[str] = ["kcal/mol", "kJ/mol", "eV", "Hartree"]


def normalize_unit(unit: str) -> str:
    """Normalize a unit string to a supported canonical name or raise ValueError."""
    u = unit.strip()
    # Accept some aliases
    aliases = {
        "kcal": "kcal/mol",
        "kcal/mol": "kcal/mol",
        "kJ": "kJ/mol",
        "kJ/mol": "kJ/mol",
        "ev": "eV",
        "eV": "eV",
        "hartree": "Hartree",
        "Hartree": "Hartree",
    }
    if u in aliases:
        return aliases[u]
    if u in FACTOR_TO_KCAL_PER_MOL:
        return u
    raise ValueError(f"Unsupported unit: {unit}")


def convert_energy(value, from_unit: str, to_unit: str):
    """Convert an energy value between supported units by routing through kcal/mol.

    Supported units: Hartree, kcal/mol, kJ/mol, eV.
    """
    fu = normalize_unit(from_unit)
    tu = normalize_unit(to_unit)
    if fu == tu:
        return value
    val_in_kcal = np.asarray(value) * FACTOR_TO_KCAL_PER_MOL[fu]
    return val_in_kcal / FACTOR_TO_KCAL_PER_MOL[tu]


def convert_values(values: Iterable[float], from_unit: str, to_unit: str):
    """Vectorized conversion for iterables of values."""
    fu = normalize_unit(from_unit)
    tu = normalize_unit(to_unit)
    if fu == tu:
        return [float(v) for v in values]
    fac = FACTOR_TO_KCAL_PER_MOL[fu] / FACTOR_TO_KCAL_PER_MOL[tu]
    return [float(v) * fac for v in values]

