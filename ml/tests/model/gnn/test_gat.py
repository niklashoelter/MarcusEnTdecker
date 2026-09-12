import torch
import pytest
from torch_geometric.data import Data, Batch
from ml.model.gnn.gat import GAT

def create_dummy_batch(num_graphs=2, num_nodes=5, num_edges=10, feature_dim=10):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, feature_dim)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)
    return Batch.from_data_list(data_list)

def test_gat_init():
    model = GAT(node_features=10, edge_features=10, hidden_size=32)
    assert isinstance(model, GAT)

def test_gat_forward():
    model = GAT(node_features=10, edge_features=10, hidden_size=32)
    batch = create_dummy_batch(feature_dim=10)
    out = model(batch)
    # Output should be (batch_size, hidden_size)
    assert out.shape == (2, 32)

def test_gat_node_embeddings():
    model = GAT(node_features=10, edge_features=10, hidden_size=32)
    batch = create_dummy_batch(num_graphs=1, num_nodes=5, feature_dim=10)
    out = model(batch, return_node_embeddings=True)
    # Output should be (num_nodes, hidden_size)
    assert out.shape == (5, 32)

