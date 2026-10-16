import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from ml.training.build_dataset import DataLoader

@pytest.fixture
def sample_csv():
    # Create a sample DataFrame
    # 0: OK energy (0.1 -> ~62.7), OK spin (0.1, 0.9)
    # 1: High energy (0.2 -> ~125.5), OK spin (0.5, 0.5)
    # 2: Low energy (0.05 -> ~31.4), BAD spin (-0.1, 1.1)

    df = pd.DataFrame({
        'idx': [1, 2, 3],
        'smiles': ['C', 'CC', 'CCC'],
        'Gibbs_S': [0.0, 0.0, 0.0], # Hartree
        'Gibbs_T': [0.1, 0.2, 0.05],
        'E_S_geometry_S_energy': [0.0, 0.0, 0.0],
        'E_T_geometry_T_energy': [0.0, 0.0, 0.0],
        'E_T_geometry_S_energy': [0.15, 0.25, 0.1],
        'E_S_geometry_T_energy': [-0.05, -0.05, -0.05], # Artificial values
        'spin_population': ["[0.1, 0.9]", "[0.5, 0.5]", "[-0.1, 1.1]"],
        'extra_col': [1, 2, 3]
    })

    # Save to temp file
    try:
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(tmp.name, index=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        yield tmp_path
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_dataloader_init(sample_csv):
    loader = DataLoader(sample_csv)
    assert isinstance(loader.input_data, pd.DataFrame)
    assert len(loader.input_data) == 3

def test_compute_targets_adiabatic(sample_csv):
    loader = DataLoader(sample_csv)
    # adiabatic_energy = (Gibbs_T + E_T_geometry_T) - (Gibbs_S + E_S_geometry_S)
    # For row 0: (0.1 + 0) - (0 + 0) = 0.1 Hartree
    # 0.1 * 627.50962 = 62.75...

    df = loader.compute_targets(target_columns=['adiabatic_energy'], return_compact=False)

    assert 'adiabatic_energy' in df.columns
    val = df['adiabatic_energy'].iloc[0]
    assert np.isclose(val, 0.1 * 627.50962)

def test_compute_targets_spin_population(sample_csv):
    loader = DataLoader(sample_csv)
    df = loader.compute_targets(target_columns=['spin_population'], return_compact=False)

    # Check parsing
    spin_pop = df['spin_population'].iloc[0]
    assert isinstance(spin_pop, list)
    assert spin_pop == [0.1, 0.9]

def test_filter_limits_spin_population_direct(sample_csv):
    """
    Test filtering on spin_population when it is explicitly requested (so parsed).
    """
    loader = DataLoader(sample_csv)

    # Filter: spin_population >= 0. No negative values.
    # Row 2 (CCC) has [-0.1, 1.1], so it should be filtered out.

    df = loader.compute_targets(
        target_columns=['spin_population'],
        filter_limits={'spin_population': (0.0, None)},
        return_compact=False
    )

    # Must remove index 2 (idx 3)
    assert len(df) == 2
    assert 3 not in df['idx'].values
    assert 1 in df['idx'].values
    assert 2 in df['idx'].values

def test_filter_limits_adiabatic(sample_csv):
    loader = DataLoader(sample_csv)
    # adiabatic energies approx: 62.75, 125.5, 31.37

    # Keep [40, 100]. Should keep ONLY row 0 (62.75).

    df = loader.compute_targets(
        target_columns=['adiabatic_energy'],
        filter_limits={'adiabatic_energy': (40, 100)},
        return_compact=False
    )

    assert len(df) == 1
    assert df['idx'].iloc[0] == 1

def test_filter_limits_default(sample_csv):
    # Default filters: adiabatic (15, 120).
    # adiabatic: 62.75 (ok), 125.5 (too high), 31.37 (ok).
    # So should keep 0 and 2.

    loader = DataLoader(sample_csv)
    df = loader.compute_targets(
        target_columns=['adiabatic_energy'],
        filter_limits=None, # Use defaults
        return_compact=False
    )

    assert len(df) == 2
    assert 1 in df['idx'].values
    assert 3 in df['idx'].values # 31.37 is ok
    assert 2 not in df['idx'].values # 125.5 > 120

def test_spin_population_filter_needs_spin_population_among_the_targets(sample_csv):
    """
    The spin populations are parsed from their strings only when `spin_population` is
    one of the target columns. Filtering on them without that compares a string with a
    float, which raises.
    """
    loader = DataLoader(sample_csv)

    with pytest.raises(TypeError):
        loader.compute_targets(
            target_columns=['adiabatic_energy'],
            filter_limits={'spin_population': (0.0, None)},
            return_compact=False
        )


def test_compute_targets_lambda_acceptor_singlet(sample_csv):
    """
    Test computation of lambda_acceptor_singlet.
    Formula: (E_T_geometry_S - E_S_geometry_S) * Eh_to_kcal
    """
    loader = DataLoader(sample_csv)
    # Row 0: E_T_geometry_S=0.15, E_S_geometry_S=0.0 -> diff=0.15
    # 0.15 * 627.50962 = 94.126443

    df = loader.compute_targets(
        target_columns=['lambda_acceptor_singlet'],
        filter_limits={}, # Disable filtering to keep all rows
        return_compact=False
    )

    assert 'lambda_acceptor_singlet' in df.columns
    val = df['lambda_acceptor_singlet'].iloc[0]
    expected = 0.15 * 627.50962
    assert np.isclose(val, expected)


def test_compute_targets_lambda_donor_triplet(sample_csv):
    """
    Test computation of lambda_donor_triplet.
    Formula: (E_S_geometry_T - E_T_geometry_T) * Eh_to_kcal
    """
    loader = DataLoader(sample_csv)
    # Row 0: E_S_geometry_T=-0.05, E_T_geometry_T=0.0 -> diff=-0.05
    # -0.05 * 627.50962 = -31.375481

    df = loader.compute_targets(
        target_columns=['lambda_donor_triplet'],
        filter_limits={}, # Disable filtering
        return_compact=False
    )

    assert 'lambda_donor_triplet' in df.columns
    val = df['lambda_donor_triplet'].iloc[0]
    expected = -0.05 * 627.50962
    assert np.isclose(val, expected)


def test_filter_limits_lambda_acceptor_singlet(sample_csv):
    """
    Test filtering on lambda_acceptor_singlet.
    Row 0: 0.15 -> ~94.1 kcal/mol
    Row 1: 0.25 -> ~156.9 kcal/mol
    Row 2: 0.10 -> ~62.8 kcal/mol
    """
    loader = DataLoader(sample_csv)

    # Filter: Keep [70, 100]. Should keep ONLY Row 0 (94.1).
    # Row 1 is 156.9 (>100), Row 2 is 62.8 (<70).

    df = loader.compute_targets(
        target_columns=['lambda_acceptor_singlet'],
        filter_limits={'lambda_acceptor_singlet': (70, 100)},
        return_compact=False
    )

    assert len(df) == 1
    assert df['idx'].iloc[0] == 1 # idx 1 is Row 0


def test_filter_limits_lambda_donor_triplet(sample_csv):
    """
    Test filtering on lambda_donor_triplet.
    Row 0: -0.05 -> -31.4 kcal/mol
    Row 1: -0.05 -> -31.4 kcal/mol
    Row 2: -0.05 -> -31.4 kcal/mol
    """
    loader = DataLoader(sample_csv)

    # Explicit filter: Keep [-40, -20]. Should keep all.
    df = loader.compute_targets(
        target_columns=['lambda_donor_triplet'],
        filter_limits={'lambda_donor_triplet': (-40, -20)},
        return_compact=False
    )
    assert len(df) == 3

    # Explicit filter: Keep > 0. Should be empty.
    df2 = loader.compute_targets(
        target_columns=['lambda_donor_triplet'],
        filter_limits={'lambda_donor_triplet': (0, None)},
        return_compact=False
    )
    assert len(df2) == 0

