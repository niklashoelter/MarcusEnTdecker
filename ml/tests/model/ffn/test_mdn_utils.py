import torch
import math
from ml.model.ffn.mdn_utils import compute_mixture_mean, compute_max_mode

def test_compute_mixture_mean():
    # Batch size 1, output dim 1, 2 components
    mu = torch.tensor([[[0.0, 10.0]]])
    sigma = torch.tensor([[[1.0, 1.0]]])
    pi = torch.tensor([[[0.5, 0.5]]])

    # Mean: 0.5 * 0 + 0.5 * 10 = 5.0
    # Var_within: 0.5 * 1^2 + 0.5 * 1^2 = 1.0
    # Var_between: (0.5 * 0^2 + 0.5 * 10^2) - 5.0^2 = 50 - 25 = 25
    # Total var: 1 + 25 = 26
    # Uncertainty: sqrt(26)

    mean, uncertainty = compute_mixture_mean(mu, sigma, pi)

    assert torch.isclose(mean, torch.tensor([[5.0]])).all()
    assert torch.isclose(uncertainty, torch.tensor([[math.sqrt(26.0)]])).all()

def test_compute_mixture_mean_single_component():
    mu = torch.tensor([[[5.0]]])
    sigma = torch.tensor([[[2.0]]])
    pi = torch.tensor([[[1.0]]])

    mean, uncertainty = compute_mixture_mean(mu, sigma, pi)

    assert torch.isclose(mean, torch.tensor([[5.0]])).all()
    assert torch.isclose(uncertainty, torch.tensor([[2.0]])).all() # Just sigma if 1 component

def test_compute_max_mode():
    mu = torch.tensor([[[0.0, 10.0]]])
    sigma = torch.tensor([[[1.0, 2.0]]])
    pi = torch.tensor([[[0.3, 0.7]]])

    max_mu, max_sigma = compute_max_mode(mu, sigma, pi)

    # Should pick index 1 (0.7 > 0.3)
    assert torch.isclose(max_mu, torch.tensor([[10.0]])).all()
    assert torch.isclose(max_sigma, torch.tensor([[2.0]])).all()

def test_compute_max_mode_batch():
    mu = torch.tensor([
        [[0.0, 10.0]],
        [[5.0, -5.0]]
    ])
    sigma = torch.tensor([
        [[1.0, 2.0]],
        [[0.5, 0.5]]
    ])
    pi = torch.tensor([
        [[0.3, 0.7]], # Pick index 1
        [[0.6, 0.4]]  # Pick index 0
    ])

    max_mu, max_sigma = compute_max_mode(mu, sigma, pi)

    expected_mu = torch.tensor([[10.0], [5.0]])
    expected_sigma = torch.tensor([[2.0], [0.5]])

    assert torch.isclose(max_mu, expected_mu).all()
    assert torch.isclose(max_sigma, expected_sigma).all()

