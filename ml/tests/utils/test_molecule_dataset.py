import torch
import pytest
import numpy as np
from ml.utils.molecule_dataset import MoleculeDataset
from unittest.mock import patch

class MockGraph:
    def __init__(self, num_nodes=2):
        self.num_nodes = num_nodes
        self.x = torch.zeros(num_nodes, 10)
        self.edge_index = torch.zeros(2, 0)
        self.rwse = None
        self.rrwp_edge = None

    def clone(self):
        return self

@pytest.fixture
def mock_smiles_to_graph_func():
    with patch('ml.utils.molecule_dataset.smiles_to_graph') as mock:
        # Default behavior: return a MockGraph with 2 nodes
        mock.return_value = MockGraph(num_nodes=2)
        yield mock

def test_moleculedataset_graph_only(mock_smiles_to_graph_func):
    smiles_list = np.array(["CC", "CO"])
    # 2 samples, 1 task
    graph_targets = np.array([[1.0], [2.0]])

    dataset = MoleculeDataset(smiles_list, graph_targets=graph_targets)

    assert len(dataset) == 2
    data = dataset[0]
    assert hasattr(data, 'y')
    # (1,) target -> y of shape (1, 1)
    assert data.y.shape == (1, 1)
    assert data.y.item() == 1.0

def test_moleculedataset_mixed_targets(mock_smiles_to_graph_func):
    smiles_list = np.array(["CC", "CO"])
    # 2 samples. Graph (scalar), Node (list of 2 values)
    graph_targets = np.array([1.0, 2.0])
    node_targets = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=object) # Arrays of lists or arrays

    # Check if this crashes
    try:
        dataset = MoleculeDataset(smiles_list, node_targets=node_targets, graph_targets=graph_targets)
    except Exception as e:
        pytest.fail(f"MoleculeDataset failed with mixed targets: {e}")

    assert len(dataset) == 2
    data = dataset[0]

    # Check Graph target
    assert hasattr(data, 'y')
    # graph target is scalar 1.0 from array [1.0, 2.0]
    assert data.y.item() == 1.0

    # Check Node target
    assert hasattr(data, 'node_y')
    # node_targets[0] -> [0.1, 0.2].
    # In code: tensor([0.1, 0.2]) -> shape (2,) -> unsqueeze(1) -> (2, 1)
    assert data.node_y.shape == (2, 1)
    # Use torch.allclose
    assert torch.allclose(data.node_y, torch.tensor([[0.1], [0.2]]))

def test_moleculedataset_node_count_mismatch(mock_smiles_to_graph_func):
    # If node target has length 3 but graph has 2 nodes -> Discard
    smiles_list = np.array(["CC"])
    node_targets = np.array([[0.1, 0.2, 0.3]], dtype=object)

    # mock returns graph with 2 nodes. Target has 3.
    # Should print warning and discard.

    dataset = MoleculeDataset(smiles_list, node_targets=node_targets)

    assert len(dataset) == 0

def test_moleculedataset_node_only(mock_smiles_to_graph_func):
    smiles_list = np.array(["CC"])
    node_targets = np.array([[0.1, 0.2]], dtype=object) # 1 sample, 2 nodes

    dataset = MoleculeDataset(smiles_list, node_targets=node_targets)

    assert len(dataset) == 1
    data = dataset[0]
    # node-only: data.y = data.node_y
    assert torch.allclose(data.y, torch.tensor([0.1, 0.2]).unsqueeze(1))

    assert hasattr(data, 'y')
    if hasattr(data, 'node_y'):
        assert torch.allclose(data.y, data.node_y)

    assert data.y.shape == (2, 1)
    assert torch.allclose(data.y, torch.tensor([[0.1], [0.2]]))

