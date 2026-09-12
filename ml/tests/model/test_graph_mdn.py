import torch
import pytest
from torch_geometric.data import Data, Batch
from ml.model.graph_mdn import GraphMDN

def create_dummy_batch(num_graphs=2, num_nodes=5, num_edges=10, feature_dim=10):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, feature_dim) # Assuming edge features = node features for simplicity
        rev_edge_index = torch.zeros(num_edges, dtype=torch.long)
        for i in range(num_edges):
            rev_edge_index[i] = i + 1 if i % 2 == 0 else i - 1
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, rev_edge_index=rev_edge_index)
        data_list.append(data)

    return Batch.from_data_list(data_list)

def create_dummy_batch_gps(num_graphs=2, num_nodes=5, num_edges=10, feature_dim=10, walk_length=10):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, feature_dim)
        rev_edge_index = torch.zeros(num_edges, dtype=torch.long)
        for i in range(num_edges):
            rev_edge_index[i] = i + 1 if i % 2 == 0 else i - 1

        rwse = torch.randn(num_nodes, walk_length)
        rrwp_edge = torch.randn(num_edges, walk_length)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, rev_edge_index=rev_edge_index,
                    rwse=rwse, rrwp_edge=rrwp_edge)
        data_list.append(data)

    return Batch.from_data_list(data_list)

def test_graph_mdn_initialization_mpnn():
    model = GraphMDN(
        encoder_type='mpnn',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
        edge_features_dim=10,
        node_features_dim=10
    )
    assert isinstance(model, GraphMDN)

def test_graph_mdn_forward_mpnn():
    feature_dim = 10
    model = GraphMDN(
        encoder_type='mpnn',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
        edge_features_dim=feature_dim,
        node_features_dim=feature_dim
    )

    batch = create_dummy_batch(feature_dim=feature_dim)

    # Forward pass
    output = model(batch)

    # Output is the graph output tuple (mu, sigma, pi)
    # Since num_node_tasks=0 (default), only graph output is returned
    assert isinstance(output, tuple)
    graph_out = output

    # Inspect graph output
    mu, sigma, pi = graph_out

    # Check shapes
    assert mu.shape == (2, 1, 3) # Batch size 2, 1 task, 3 components
    assert sigma.shape == (2, 1, 3)
    assert pi.shape == (2, 1, 3)

    # Check probabilities sum to 1
    pi_sum = pi.sum(dim=-1)
    assert torch.allclose(pi_sum, torch.ones_like(pi_sum))

def test_graph_mdn_forward_gat():
    feature_dim = 10
    # GAT typically doesn't use edge features unless specified, but let's check basic run
    model = GraphMDN(
        encoder_type='gat',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
        heads=2 # GAT specific
    )

    batch = create_dummy_batch(feature_dim=feature_dim)
    output = model(batch)

    graph_out = output
    mu, sigma, pi = graph_out
    assert mu.shape == (2, 1, 3) # Batch size 2, 1 task, 3 components

def test_graph_mdn_forward_dmpnn():
    feature_dim = 10
    model = GraphMDN(
        encoder_type='dmpnn',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
    )

    batch = create_dummy_batch(feature_dim=feature_dim)
    output = model(batch)

    graph_out = output
    mu, sigma, pi = graph_out
    assert mu.shape == (2, 1, 3) # Batch size 2, 1 task, 3 components

def test_graph_mdn_forward_gin():
    feature_dim = 10
    model = GraphMDN(
        encoder_type='gin',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
    )

    batch = create_dummy_batch(feature_dim=feature_dim)
    output = model(batch)
    graph_out = output
    mu, sigma, pi = graph_out
    assert mu.shape == (2, 1, 3)

def test_graph_mdn_forward_transformer():
    feature_dim = 10
    model = GraphMDN(
        encoder_type='transformer',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
    )

    batch = create_dummy_batch(feature_dim=feature_dim)
    output = model(batch)
    graph_out = output
    mu, sigma, pi = graph_out
    assert mu.shape == (2, 1, 3)

def test_graph_mdn_forward_graph_gps():
    feature_dim = 10
    walk_length = 10
    model = GraphMDN(
        encoder_type='gps',
        node_hidden_dim=32,
        encoder_layers=2,
        num_graph_tasks=1,
        n_components=3,
        mdn_hidden_dims=[32, 16],
        walk_length=walk_length
    )

    batch = create_dummy_batch_gps(feature_dim=feature_dim, walk_length=walk_length)
    output = model(batch)
    graph_out = output
    mu, sigma, pi = graph_out
    assert mu.shape == (2, 1, 3)

