import torch
import pytest
from torch_geometric.data import Data, Batch
from pyrates.model.gnn.dmpnn import DMPNN

def create_dummy_batch(num_graphs=2, num_nodes=5, num_edges=10, feature_dim=10):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, feature_dim)
        # Assuming DMPNN would create rev_edge_index during processing or expect it.
        # But wait, looking at DMPNN.py logic, it expects `data.rev_edge_index`.
        # So I must add `rev_edge_index`.
        # Simple undirected graph: edges (u,v) and (v,u).
        # We need to construct rev_edge_index which maps edge (u,v) to (v,u).
        # For simplicity, let's create a dataset where edges are explicitly symmetric and tracked.

        # Or, I can check if DMPNN automatically handles missing rev_edge_index (it doesn't seem to based on my read).
        # The standard DMPNN in PyG or Chemprop usually requires pre-processing.
        # Let's mock rev_edge_index.

        # If I have 10 edges, I need an index array of size 10 mapping each edge to its reverse.
        # Let's just use random valid indices for the mock test.
        rev_edge_index = torch.arange(num_edges).flip(0) # Simple reverse

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, rev_edge_index=rev_edge_index)
        data_list.append(data)
    return Batch.from_data_list(data_list)

def test_dmpnn_init():
    model = DMPNN(node_features=10, edge_features=10, hidden_size=32)
    assert isinstance(model, DMPNN)

def test_dmpnn_forward():
    model = DMPNN(node_features=10, edge_features=10, hidden_size=32)
    batch = create_dummy_batch(feature_dim=10)
    out = model(batch)
    # Output should be (batch_size, hidden_size)
    assert out.shape == (2, 32)

def test_dmpnn_node_embeddings():
    model = DMPNN(node_features=10, edge_features=10, hidden_size=32)
    batch = create_dummy_batch(num_graphs=1, num_nodes=5, feature_dim=10)
    out = model(batch, return_node_embeddings=True)
    # Output should be (num_nodes, hidden_size)
    assert out.shape == (5, 32)

