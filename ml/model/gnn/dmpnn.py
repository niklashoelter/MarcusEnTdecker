import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class DMPNN(nn.Module):
    """Directed Message Passing Neural Network (D-MPNN)"""
    def __init__(self, node_features, edge_features, hidden_size, depth=3, dropout=0.1, **kwargs):
        super(DMPNN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        self.W_i = nn.Linear(node_features + edge_features, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(node_features + hidden_size, hidden_size)

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr, rev_edge_index = data.x, data.edge_index, data.edge_attr, data.rev_edge_index
        num_nodes = x.size(0)
        row, col = edge_index

        input_ = torch.cat([x[row], edge_attr], dim=1)
        message = torch.relu(self.W_i(input_))
        message = self.dropout(message)

        for _ in range(self.depth - 1):
            rev_messages = message[rev_edge_index]
            aggr_message = torch.zeros(num_nodes, self.hidden_size, device=x.device)
            aggr_message.index_add_(0, col, message)
            message = torch.relu(self.W_h(aggr_message[row] - rev_messages))
            message = self.dropout(message)

        aggr_message = torch.zeros(num_nodes, self.hidden_size, device=x.device)
        aggr_message.index_add_(0, col, message)

        node_h = torch.cat([x, aggr_message], dim=1)
        node_h = torch.relu(self.W_o(node_h))
        node_h = self.dropout(node_h)

        if return_node_embeddings:
            return node_h

        graph_out = global_add_pool(node_h, data.batch)
        return graph_out

