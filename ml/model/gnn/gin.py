import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) with Edge Features (GINEConv).
    Known for high expressiveness in graph regression tasks.
    """
    def __init__(self, node_features, edge_features, hidden_size, depth=3, dropout=0.1, **kwargs):
        super(GIN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        # Initial embedding
        self.node_embedding = nn.Linear(node_features, hidden_size)

        # Edge embedding needed because GINEConv expects edge_attr to match node dim
        self.edge_embedding = nn.Linear(edge_features, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            # GIN requires an MLP for the aggregation step
            mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            # GINEConv expects edge features
            self.layers.append(GINEConv(mlp, train_eps=True))

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_embedding(x))
        edge_attr = F.relu(self.edge_embedding(edge_attr)) # Project edges to hidden_dim

        for layer in self.layers:
            # GINEConv needs edge_attr to have same dim as x, hence the embedding above
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out

