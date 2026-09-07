import torch
import pytest
from unittest.mock import MagicMock
from pyrates.model.training import compute_loss

class MockBatch:
    def __init__(self, batch_size=2, num_nodes=4):
        self.batch_size = batch_size
        self.num_graphs = batch_size
        self.batch = torch.tensor([0, 0, 1, 1], dtype=torch.long) # 2 nodes per graph
        self.y = torch.randn(batch_size, 1) # Graph targets
        self.node_y = torch.randn(num_nodes, 1) # Node targets

    def to(self, device):
        return self

def test_compute_loss_graph_only():
    # Mock model
    model = MagicMock()
    model.num_graph_tasks = 1
    model.num_node_tasks = 0
    model.lambda_cal = 0.0
    model.lambda_sharp = 0.0

    # Batch
    batch = MockBatch()

    # Output: (mu, sigma, pi)
    # Batch size 2.
    mu = torch.randn(2, 1, 3)
    sigma = torch.abs(torch.randn(2, 1, 3))
    pi = torch.softmax(torch.randn(2, 1, 3), dim=-1)
    outputs = (mu, sigma, pi)

    loss_dict = compute_loss(model, outputs, batch)

    assert "loss" in loss_dict
    assert loss_dict["loss"].ndim == 0 # scalar

def test_compute_loss_node_only():
    model = MagicMock()
    model.num_graph_tasks = 0
    model.num_node_tasks = 1
    model.lambda_cal = 0.0
    model.lambda_sharp = 0.0

    batch = MockBatch()

    # Output: node level. 4 nodes.
    mu = torch.randn(4, 1, 3)
    sigma = torch.abs(torch.randn(4, 1, 3))
    pi = torch.softmax(torch.randn(4, 1, 3), dim=-1)
    outputs = (mu, sigma, pi)

    loss_dict = compute_loss(model, outputs, batch)

    assert "loss" in loss_dict

def test_compute_loss_mixed():
    model = MagicMock()
    model.num_graph_tasks = 1
    model.num_node_tasks = 1
    model.weight_graph_loss = 0.5
    model.lambda_cal = 0.0
    model.lambda_sharp = 0.0

    batch = MockBatch()

    # Graph output
    g_mu = torch.randn(2, 1, 3)
    g_sigma = torch.abs(torch.randn(2, 1, 3))
    g_pi = torch.softmax(torch.randn(2, 1, 3), dim=-1)

    # Node output
    n_mu = torch.randn(4, 1, 3)
    n_sigma = torch.abs(torch.randn(4, 1, 3))
    n_pi = torch.softmax(torch.randn(4, 1, 3), dim=-1)

    outputs = ((g_mu, g_sigma, g_pi), (n_mu, n_sigma, n_pi))

    loss_dict = compute_loss(model, outputs, batch)

    assert "loss" in loss_dict
    assert "loss_g" in loss_dict
    assert "loss_n" in loss_dict

    # Validate weighting
    expected_loss = 0.5 * loss_dict["loss_g"] + 0.5 * loss_dict["loss_n"]
    assert torch.isclose(loss_dict["loss"], expected_loss)

