import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import molecular_size  # noqa: E402


def test_volume_of_one_atom_is_its_van_der_waals_sphere():
    volume = molecular_size.vdw_volume(["C"], np.zeros((1, 3)))
    assert volume == pytest.approx(4.0 / 3.0 * np.pi * molecular_size.BONDI["C"] ** 3, rel=0.01)
    assert molecular_size.sphere_radius(volume) == pytest.approx(molecular_size.BONDI["C"],
                                                                 rel=0.01)


def test_volume_counts_overlapping_spheres_once():
    single = molecular_size.vdw_volume(["H"], np.zeros((1, 3)))
    apart = molecular_size.vdw_volume(["H", "H"], np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    overlapping = molecular_size.vdw_volume(["H", "H"],
                                            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert apart == pytest.approx(2.0 * single, rel=0.01)
    assert single < overlapping < 2.0 * single


def test_radius_reads_an_xyz_file_and_is_none_without_one(tmp_path):
    xyz = tmp_path / "opt_geometry.xyz"
    xyz.write_text("1\ncomment\nC 0.0 0.0 0.0\n")
    assert molecular_size.radius(xyz) == pytest.approx(molecular_size.BONDI["C"], rel=0.01)
    assert molecular_size.radius(tmp_path / "absent.xyz") is None
