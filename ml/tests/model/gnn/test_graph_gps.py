import torch
from torch_geometric.data import Data, Batch
from ml.model.gnn.graph_gps import GraphGPS

def create_dummy_batch(num_graphs=2, num_nodes=5, num_edges=10, feature_dim=10, walk_length=10):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, feature_dim)

        # Positional Encodings
        rwse = torch.randn(num_nodes, walk_length)
        rrwp_edge = torch.randn(num_edges, walk_length)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, rwse=rwse, rrwp_edge=rrwp_edge)
        data_list.append(data)
    return Batch.from_data_list(data_list)

def test_gps_init():
    model = GraphGPS(node_features=10, edge_features=10, hidden_size=32, walk_length=10)
    assert isinstance(model, GraphGPS)

def test_gps_forward():
    walk_length = 5
    model = GraphGPS(node_features=10, edge_features=10, hidden_size=32, walk_length=walk_length)
    batch = create_dummy_batch(feature_dim=10, walk_length=walk_length)
    out = model(batch)
    # Output should be (batch_size, hidden_size)
    assert out.shape == (2, 32)

def test_gps_node_embeddings():
    walk_length = 5
    model = GraphGPS(node_features=10, edge_features=10, hidden_size=32, walk_length=walk_length)
    batch = create_dummy_batch(num_graphs=1, num_nodes=5, feature_dim=10, walk_length=walk_length)
    out = model(batch, return_node_embeddings=True)
    # Output should be (num_nodes, hidden_size)
    assert out.shape == (5, 32)


