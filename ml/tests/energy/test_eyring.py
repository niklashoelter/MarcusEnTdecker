import pytest
import math
from ml.energy.eyring import eyring_equation

def test_eyring_equation_kj_mol():
    dG = 100.0 # kJ/mol
    T = 298.15
    k = eyring_equation(dG, T, units="kJ/mol")
    assert isinstance(k, float)
    assert k > 0

def test_eyring_equation_kcal_mol():
    dG = 23.9 # kcal/mol ~ 100 kJ/mol
    T = 298.15
    k = eyring_equation(dG, T, units="kcal/mol")
    assert isinstance(k, float)
    assert k > 0

def test_eyring_equation_ev():
    dG = 1.036 # eV ~ 100 kJ/mol
    T = 298.15
    k = eyring_equation(dG, T, units="eV")
    assert isinstance(k, float)
    assert k > 0

def test_eyring_equation_j_mol():
    dG = 100000.0 # J/mol
    T = 298.15
    k = eyring_equation(dG, T, units="J/mol")
    assert isinstance(k, float)
    assert k > 0

def test_eyring_equation_invalid_units():
    with pytest.raises(ValueError):
        eyring_equation(100.0, units="invalid")

def test_eyring_equation_values():
    T = 300.0
    k_expected = (1.380649e-23 * T) / 6.62607015e-34
    k_calc = eyring_equation(0.0, T, units="J/mol")
    assert math.isclose(k_calc, k_expected, rel_tol=1e-5)

