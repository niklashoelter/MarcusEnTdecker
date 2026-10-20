import torch
import pytest
import numpy as np
from ml.utils.molecule_dataset import MoleculeDataset
from unittest.mock import patch
from torch_geometric.data import Data

@pytest.fixture
def mock_smiles_to_graph_func():
    with patch('ml.utils.molecule_dataset.smiles_to_graph') as mock:
        mock.side_effect = lambda *args, **kwargs: Data(
            x=torch.zeros(2, 10), edge_index=torch.empty(2, 0, dtype=torch.long))
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

    dataset = MoleculeDataset(smiles_list, node_targets=node_targets, graph_targets=graph_targets)

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
    assert data.y.shape == (2, 1)
    assert torch.allclose(data.y, data.node_y)
    assert torch.allclose(data.y, torch.tensor([[0.1], [0.2]]))


def test_molecule_graphs_and_items_are_independent(mock_smiles_to_graph_func):
    dataset = MoleculeDataset(np.array(["CC", "CO"]),
                              graph_targets=np.array([[1.0], [2.0]]))
    first, second = dataset[0], dataset[1]
    first.x.fill_(9)
    assert first.y.item() == 1.0 and second.y.item() == 2.0
    assert not dataset.graphs[0].x.any() and not second.x.any()
