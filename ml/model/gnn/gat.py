import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


class GAT(nn.Module):
    """Graph Attention Network (GAT) with Edge Feature Support"""
    def __init__(self, node_features, edge_features, hidden_size, num_heads=8, depth=3, dropout=0.1, **kwargs):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout

        self.initial_embedding = nn.Linear(node_features, hidden_size)
        self.gat_layers = nn.ModuleList()

        # Input Layer
        # edge_dim tells GATConv to expect edge features of this size
        self.gat_layers.append(GATConv(
            hidden_size,
            hidden_size,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_features
        ))

        # Hidden Layers
        for _ in range(depth - 2):
            self.gat_layers.append(GATConv(
                hidden_size * num_heads,
                hidden_size,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_features
            ))

        # Output Layer (heads=1, concat=False)
        self.gat_layers.append(GATConv(
            hidden_size * num_heads,
            hidden_size,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_features
        ))

        self.readout = nn.Linear(hidden_size, hidden_size)

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.initial_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.gat_layers):
            # Pass edge_attr to the GAT layer
            x = layer(x, edge_index, edge_attr=edge_attr)

            if i < self.depth - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.readout(x))

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out

