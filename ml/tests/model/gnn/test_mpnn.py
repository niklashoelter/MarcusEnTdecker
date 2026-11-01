from ml.model.gnn.mpnn import MPNN

def test_mpnn_init():
    model = MPNN(node_features=10, edge_features=10, hidden_size=32)
    assert isinstance(model, MPNN)

def test_mpnn_forward(make_graph_batch):
    model = MPNN(node_features=10, edge_features=10, hidden_size=32)
    batch = make_graph_batch(feature_dim=10)
    out = model(batch)
    # Output should be (batch_size, hidden_size)
    assert out.shape == (2, 32)

def test_mpnn_node_embeddings(make_graph_batch):
    model = MPNN(node_features=10, edge_features=10, hidden_size=32)
    batch = make_graph_batch(num_graphs=1, num_nodes=5, feature_dim=10)
    out = model(batch, return_node_embeddings=True)
    # Output should be (num_nodes, hidden_size)
    assert out.shape == (5, 32)

