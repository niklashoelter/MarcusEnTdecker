import torch
import torch.nn as nn
import pytest
from pyrates.model.ffn.mixture_density_network import MDN

def test_mdn_init():
    model = MDN(
        input_dim=10,
        output_dim=2,
        n_components=3,
        hidden_dims=[32, 16]
    )
    assert isinstance(model, nn.Module)

def test_mdn_forward_shapes():
    batch_size = 5
    input_dim = 10
    output_dim = 2 # e.g. 2 targets
    n_components = 3

    model = MDN(
        input_dim=input_dim,
        output_dim=output_dim,
        n_components=n_components,
        hidden_dims=[32]
    )

    x = torch.randn(batch_size, input_dim)
    mu, sigma, pi = model(x)

    # Check shapes: (batch_size, output_dim, n_components)
    assert mu.shape == (batch_size, output_dim, n_components)
    assert sigma.shape == (batch_size, output_dim, n_components)
    assert pi.shape == (batch_size, output_dim, n_components)

    # Check constraints
    assert torch.all(sigma > 0)
    # Check probability sum
    pi_sum = pi.sum(dim=-1)
    assert torch.allclose(pi_sum, torch.ones(batch_size, output_dim))

def test_mdn_no_hidden_layers():
    model = MDN(
        input_dim=10,
        output_dim=1,
        n_components=1,
        hidden_dims=[]
    )
    x = torch.randn(2, 10)
    mu, sigma, pi = model(x)
    # Just linear projection
    assert mu.shape == (2, 1, 1)

def test_mdn_activation_elu():
    model = MDN(
        input_dim=10,
        output_dim=1,
        activation='elu'
    )
    assert isinstance(model.activation, nn.ELU)

def test_mdn_invalid_activation():
    with pytest.raises(ValueError):
        MDN(input_dim=10, output_dim=1, activation='invalid')

