import numpy as np
from ml.energy.marcus_theory import marcus_symmetric, marcus_asymmetric

def test_marcus_symmetric_basic():
    dG = 0.0
    lambda_acc = 1.0
    lambda_don = 1.0
    # Expected: (0 + 1 + 1)^2 / (4 * (1+1)) = 4 / 8 = 0.5
    expected = 0.5
    # marcus_symmetric takes the total reorganization energy, not the two halves.
    result = marcus_symmetric(dG, lambda_acc + lambda_don)
    assert np.isclose(result, expected)

def test_marcus_symmetric_vectorized():
    dG = np.array([0.0, 0.0])
    lambda_acc = np.array([1.0, 2.0])
    lambda_don = np.array([1.0, 2.0])
    # Case 1: 0.5
    # Case 2: (0 + 2 + 2)^2 / (4 * (2+2)) = 16 / 16 = 1.0
    expected = np.array([0.5, 1.0])
    # marcus_symmetric takes the total reorganization energy, not the two halves.
    result = marcus_symmetric(dG, lambda_acc + lambda_don)
    assert np.allclose(result, expected)

def test_marcus_asymmetric_basic():

    dG = -0.5
    l_sub_s = 1.0
    l_sub_t = 0.5
    l_pc_s = 0.5
    l_pc_t = 1.0

    # l_r = 1.0 + 1.0 = 2.0
    # l_p = 0.5 + 0.5 = 1.0
    # term under sqrt: l_r*l_p + (l_r-l_p)*dG = 2*1 + (1)*(-0.5) = 2 - 0.5 = 1.5
    # numerator = -1.0 + sqrt(1.5)
    # denominator = 1.0
    # result = 2.0 * ( (-1 + sqrt(1.5))/1 )^2

    expected = 2.0 * (-1.0 + np.sqrt(1.5))**2

    result = marcus_asymmetric(
        delta_g=dG,
        lambda_substrate_singlet=l_sub_s,
        lambda_substrate_triplet=l_sub_t,
        lambda_pc_singlet=l_pc_s,
        lambda_pc_triplet=l_pc_t
    )

    assert np.isclose(result, expected)

