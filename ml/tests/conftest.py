"""Shared graph inputs for encoder tests."""

import pytest
import torch
from torch_geometric.data import Batch, Data


@pytest.fixture
def make_graph_batch():
    def make(num_graphs=2, num_nodes=5, num_edges=10, feature_dim=10):
        graphs = []
        for _ in range(num_graphs):
            starts = torch.arange(num_edges // 2) % num_nodes
            bonds = torch.stack((starts, (starts + 1) % num_nodes))
            edges = torch.stack((bonds, bonds.flip(0)), dim=-1).reshape(2, -1)
            attributes = torch.randn(num_edges // 2, feature_dim).repeat_interleave(2, dim=0)
            graphs.append(Data(x=torch.randn(num_nodes, feature_dim),
                               edge_index=edges, edge_attr=attributes))
        batch = Batch.from_data_list(graphs)
        batch.rev_edge_index = torch.arange(batch.num_edges) ^ 1
        return batch

    return make
