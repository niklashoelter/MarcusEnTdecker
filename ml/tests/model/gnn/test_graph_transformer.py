from ml.model.gnn.graph_transformer import BaseGraphTransformer, GraphTransformer

def test_gt_init():
    model = BaseGraphTransformer(node_features=10, edge_features=10, hidden_size=32)
    assert isinstance(model, BaseGraphTransformer)

def test_gt_forward(make_graph_batch):
    model = BaseGraphTransformer(node_features=10, edge_features=10, hidden_size=32)
    batch = make_graph_batch(feature_dim=10)
    out = model(batch)
    # Output should be (batch_size, hidden_size)
    assert out.shape == (2, 32)

def test_gt_node_embeddings(make_graph_batch):
    model = BaseGraphTransformer(node_features=10, edge_features=10, hidden_size=32)
    batch = make_graph_batch(num_graphs=1, num_nodes=5, feature_dim=10)
    out = model(batch, return_node_embeddings=True)
    # Output should be (num_nodes, hidden_size)
    assert out.shape == (5, 32)

def test_gtv2_init():
    model = GraphTransformer(node_features=10, edge_features=10, hidden_size=32)
    assert isinstance(model, GraphTransformer)

def test_gtv2_forward(make_graph_batch):
    model = GraphTransformer(node_features=10, edge_features=10, hidden_size=32)
    batch = make_graph_batch(feature_dim=10)
    out = model(batch)
    # Output should be (batch_size, hidden_size)
    assert out.shape == (2, 32)

def test_gtv2_node_embeddings(make_graph_batch):
    model = GraphTransformer(node_features=10, edge_features=10, hidden_size=32)
    batch = make_graph_batch(num_graphs=1, num_nodes=5, feature_dim=10)
    out = model(batch, return_node_embeddings=True)
    # Output should be (num_nodes, hidden_size)
    assert out.shape == (5, 32)

