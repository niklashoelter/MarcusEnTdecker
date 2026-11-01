import torch
import pytest
import numpy as np
from unittest.mock import MagicMock
from ml.model.inference import predict

# Mock data batch
class MockBatch:
    def __init__(self, batch_size=2, num_nodes=5):
        self.batch_size = batch_size
        self.num_graphs = batch_size
        self.x = torch.randn(num_nodes, 10)
        self.edge_index = torch.zeros(2, 0)
        self.y = torch.randn(batch_size, 1) # Graph targets
        self.node_y = torch.randn(num_nodes, 1) # Node targets

    def to(self, device):
        return self

@pytest.fixture
def mock_loader():
    # Return a list of 1 batch
    return [MockBatch(batch_size=2, num_nodes=4)]

@pytest.fixture
def mock_model_dual():
    model = MagicMock()
    model.eval.return_value = None

    def forward(batch):
        # Return (graph_out, node_out)
        # graph_out: (mu, sigma, pi)
        # shapes: (batch_size, tasks, components)

        # Graph: batch_size=2, tasks=1, comp=3
        g_mu = torch.randn(2, 1, 3)
        g_sigma = torch.abs(torch.randn(2, 1, 3))
        g_pi = torch.softmax(torch.randn(2, 1, 3), dim=-1)

        # Node: num_nodes=4, tasks=1, comp=3
        n_mu = torch.randn(4, 1, 3)
        n_sigma = torch.abs(torch.randn(4, 1, 3))
        n_pi = torch.softmax(torch.randn(4, 1, 3), dim=-1)

        return ((g_mu, g_sigma, g_pi), (n_mu, n_sigma, n_pi))

    model.side_effect = forward
    return model

@pytest.fixture
def mock_model_graph_only():
    model = MagicMock()
    model.eval.return_value = None

    def forward(batch):
        # Return graph_out only
        g_mu = torch.randn(2, 1, 3)
        g_sigma = torch.abs(torch.randn(2, 1, 3))
        g_pi = torch.softmax(torch.randn(2, 1, 3), dim=-1)
        return (g_mu, g_sigma, g_pi)

    model.side_effect = forward
    return model

class MockScaler:
    def __init__(self):
        self.scale_ = np.array([1.0])
        self.mean_ = np.array([0.0])

    def inverse_transform(self, x):
        return x

class MockDualScaler:
    def __init__(self):
        self.graph_scaler = MockScaler()
        self.node_scaler = MockScaler()

def test_predict_dual_head(mock_loader, mock_model_dual):
    scaler = MockDualScaler()
    device = 'cpu'

    result = predict(mock_model_dual, mock_loader, device, scaler, mode='mixture_mean')

    assert 'graph' in result
    assert 'node' in result

    # Check graph results
    g_res = result['graph']
    assert g_res['predictions'] is not None
    assert g_res['predictions'].shape[0] == 2 # 2 graphs

    # Check node results
    n_res = result['node']
    assert n_res['predictions'] is not None
    assert n_res['predictions'].shape[0] == 4 # 4 nodes

def test_predict_graph_only(mock_loader, mock_model_graph_only):
    scaler = MockDualScaler() # Has both but model only returns graph
    device = 'cpu'

    # single head; output size == num_graphs -> graph-level

    result = predict(mock_model_graph_only, mock_loader, device, scaler, mode='mixture_mean')

    assert result['graph']['predictions'] is not None
    assert result['node']['predictions'] is None
    assert result['graph']['predictions'].shape[0] == 2

def test_predict_no_targets(mock_model_graph_only):
    # Test with batch having no targets
    batch = MockBatch(batch_size=2)
    batch.y = None
    batch.node_y = None
    loader = [batch]

    scaler = MockDualScaler()
    result = predict(mock_model_graph_only, loader, 'cpu', scaler)

    assert result['graph']['targets'] is None


@pytest.mark.parametrize("head", ["graph", "node"])
@pytest.mark.parametrize("mode", ["mixture_mean", "max_mode"])
def test_predictions_and_mixtures_are_returned_in_physical_units(head, mode):
    from sklearn.preprocessing import StandardScaler
    from ml.utils.molecule_dataset import DualScaler

    batch = MockBatch(batch_size=2, num_nodes=4)
    batch.y = torch.zeros(2, 1)
    batch.node_y = torch.zeros(4, 1)
    size = 2 if head == "graph" else 4
    mu = torch.tensor([[[0.0, 2.0]]]).repeat(size, 1, 1)
    sigma = torch.ones_like(mu)
    pi = torch.tensor([[[0.75, 0.25]]]).repeat(size, 1, 1)
    model = MagicMock(return_value=(mu, sigma, pi))
    graph_scaler = StandardScaler().fit([[8.0], [12.0]])
    node_scaler = StandardScaler().fit([[-8.0], [-2.0]])
    scaler = graph_scaler if head == "graph" else node_scaler

    result = predict(model, [batch], "cpu", DualScaler(graph_scaler, node_scaler), mode)[head]
    standard_mean = 0.5 if mode == "mixture_mean" else 0.0
    standard_sd = np.sqrt(1.75) if mode == "mixture_mean" else 1.0
    np.testing.assert_allclose(result["predictions"],
                               np.full((size, 1), standard_mean * scaler.scale_[0] + scaler.mean_[0]))
    np.testing.assert_allclose(result["uncertainties"],
                               np.full((size, 1), standard_sd * scaler.scale_[0]), rtol=1e-6)
    np.testing.assert_allclose(result["targets"], np.full((size, 1), scaler.mean_[0]))
    np.testing.assert_allclose(result["mdn_params"]["mu"], mu.numpy() * scaler.scale_[0] + scaler.mean_[0])
    np.testing.assert_allclose(result["mdn_params"]["sigma"], sigma.numpy() * scaler.scale_[0])


