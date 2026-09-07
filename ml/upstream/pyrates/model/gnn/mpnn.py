import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool

class MPNNLayer(MessagePassing):
    """
    Standard Message Passing Layer (Message -> Aggregate -> Update)
    Using GRU for update.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.0):
        super(MPNNLayer, self).__init__(aggr='add')  # Sum aggregation

        # Message function: MLP(cat(x_i, x_j, e_ij))
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update function: GRU
        self.gru = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x: node features [num_nodes, node_dim]
        edge_index: [2, num_edges]
        edge_attr: edge features [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target, x_j: source
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        # x is the previous node state
        new_x = self.gru(aggr_out, x)
        return new_x


class MPNN(nn.Module):
    """Standard MPNN Encoder (undirected message passing)"""
    def __init__(self, node_features, edge_features, hidden_size, depth=3, dropout=0.1, **kwargs):
        super(MPNN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # Initial embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_size),
            nn.ReLU()
        )

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(
                node_dim=hidden_size,
                edge_dim=edge_features,
                hidden_dim=hidden_size,
                dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_embedding(x)

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out

