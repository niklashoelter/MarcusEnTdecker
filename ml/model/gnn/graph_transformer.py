import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool


class BaseGraphTransformer(nn.Module):
    """
    Graph Transformer Network.
    """
    def __init__(self, node_features, edge_features, hidden_size, num_heads=4, depth=3, dropout=0.1, **kwargs):
        super(BaseGraphTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        self.node_embedding = nn.Linear(node_features, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            # TransformerConv supports edge_dim directly
            self.layers.append(TransformerConv(
                in_channels=hidden_size,
                out_channels=hidden_size // num_heads, # PyG concatenates heads by default
                heads=num_heads,
                edge_dim=edge_features,
                dropout=dropout
            ))

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_embedding(x))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            # TransformerConv applies dropout internally, but we can add layer dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out


class GraphTransformer(nn.Module):
    """
    Graph Transformer Network V2.
    Incorporates Residual Connections, LayerNorm, and Feed-Forward Networks
    """
    def __init__(self, node_features, edge_features, hidden_size, num_heads=4, depth=3, dropout=0.1, **kwargs):
        super(GraphTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        self.node_embedding = nn.Linear(node_features, hidden_size)

        # Linear projection for edge features to match SimG style preprocessing
        self.edge_embedding = nn.Linear(edge_features, edge_features)

        self.layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(depth):
            # 1. Attention Part
            self.layers.append(TransformerConv(
                in_channels=hidden_size,
                out_channels=hidden_size // num_heads,
                heads=num_heads,
                edge_dim=edge_features,
                dropout=dropout,
                beta=True # Learnable beta (bias) in TransformerConv
            ))
            self.norms1.append(nn.LayerNorm(hidden_size))

            # 2. Feed-Forward Part (FFN)
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(dropout)
            ))
            self.norms2.append(nn.LayerNorm(hidden_size))

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_embedding(x))
        # Optional: Project edges if needed, though TransformerConv handles edge_dim.
        # But we can nonlinearize them first if we wanted.
        # For now, we pass them directly as TransformerConv expects.

        for layer, norm1, ffn, norm2 in zip(self.layers, self.norms1, self.ffns, self.norms2):
            # --- Sub-Layer 1: Attention ---
            x_in = x
            x_out = layer(x, edge_index, edge_attr=edge_attr)
            x = norm1(x_in + x_out) # Residual + Norm

            # --- Sub-Layer 2: Feed Forward ---
            x_in = x
            x_out = ffn(x)
            x = norm2(x_in + x_out) # Residual + Norm

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out

