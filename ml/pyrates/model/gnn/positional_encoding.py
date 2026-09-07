import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import AddRandomWalkPE


def add_random_walk_positional_encoding(graph, walk_length=10):
    """
    Stores RWSE as graph.rwse — (num_nodes, walk_length)
    """
    transform = AddRandomWalkPE(walk_length=walk_length, attr_name='rwse')
    return transform(graph)


def add_relative_random_walk_positional_encoding(graph, walk_length=10):
    """
    Stores RRWP as graph.rrwp_edge — (num_edges, walk_length)
    """
    N = graph.num_nodes
    adj = to_dense_adj(graph.edge_index, max_num_nodes=N)[0]
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
    T = adj / deg

    rw_powers = []
    Tk = T.clone()
    for _ in range(walk_length):
        rw_powers.append(Tk)
        Tk = Tk @ T

    rw_stack = torch.stack(rw_powers, dim=-1)  # (N, N, walk_length)
    src, dst = graph.edge_index
    graph.rrwp_edge = rw_stack[src, dst]       # (num_edges, walk_length)
    return graph