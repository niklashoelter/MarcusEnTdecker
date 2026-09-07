import torch
import torch.nn as torch_nn
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
from typing import Any, Dict, Optional

class RedrawProjection:
    def __init__(self, model: torch_nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GraphGPS(torch_nn.Module):
    """
    Graph GPS Encoder.
    Combines local MPNN (GINEConv) with global attention (Transformer/Performer).
    Consumes graph.rwse as positional encoding — added by dataset, ignored by other encoders.
    """
    def __init__(self, node_features: int, edge_features: int, hidden_size: int,
                 walk_length: int = 10,
                 pe_dim: int = 28,
                 num_heads: int = 4,
                 depth: int = 6,
                 dropout: float = 0.1,
                 attn_type: str = 'multihead',
                 attn_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()

        self.pe_norm = torch_nn.BatchNorm1d(walk_length)  # for rwse (node PE)
        self.pe_lin = torch_nn.Linear(walk_length, pe_dim)  # rwse → pe_dim
        self.edge_pe_norm = torch_nn.BatchNorm1d(walk_length)  # for rrwp_edge (edge PE)
        self.edge_pe_lin = torch_nn.Linear(walk_length, pe_dim)  # rrwp_edge → pe_dim

        self.node_emb = torch_nn.Linear(node_features, hidden_size - pe_dim)
        self.edge_emb = torch_nn.Linear(edge_features, hidden_size - pe_dim)

        self.convs = torch_nn.ModuleList()
        for _ in range(depth):
            nn = torch_nn.Sequential(
                torch_nn.Linear(hidden_size, hidden_size * 2),
                torch_nn.BatchNorm1d(hidden_size * 2),
                torch_nn.ReLU(),
                torch_nn.Linear(hidden_size * 2, hidden_size),
            )
            conv = GPSConv(
                channels=hidden_size,
                conv=GINEConv(nn, edge_dim=hidden_size),
                heads=num_heads,
                norm='layer_norm',
                dropout=dropout,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs or {}
            )
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None
        )

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # ── Inject Positional Encoding ──
        pe = self.pe_lin(self.pe_norm(data.rwse))
        x = torch.cat([self.node_emb(x), pe], dim=-1)
        edge_pe = self.edge_pe_lin(self.edge_pe_norm(data.rrwp_edge))
        edge_attr = torch.cat([self.edge_emb(edge_attr), edge_pe], dim=-1)

        for conv in self.convs:
            self.redraw_projection.redraw_projections()
            x = conv(x, edge_index, batch=data.batch, edge_attr=edge_attr)

        if return_node_embeddings:
            return x  # (N, hidden_size)

        return global_add_pool(x, data.batch)  # (B, hidden_size)