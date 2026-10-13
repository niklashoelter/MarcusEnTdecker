import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "computational_data_scripts"))

import molecular_extent  # noqa: E402
from molecular_size import BONDI  # noqa: E402


def test_extent_of_one_atom_is_its_van_der_waals_radius():
    assert molecular_extent.extent_radius(["C"], np.zeros((1, 3))) == pytest.approx(BONDI["C"])


def test_extent_is_half_the_largest_distance_plus_both_radii():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    expected = (4.0 + BONDI["C"] + BONDI["H"]) / 2.0
    assert molecular_extent.extent_radius(["C", "O", "H"], xyz) == pytest.approx(expected)
