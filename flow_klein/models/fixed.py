"""
Klein Model Components for Graph Encoding and Decoding.

This module provides the V2 encoder/decoder stack used by Klein GraphTask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv

from flow_klein.geometry.klein import Klein
from flow_klein.models.layers import node_mlp


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling that ignores padded nodes."""
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max pooling that ignores padded nodes."""
    masked_x = x.masked_fill(mask.unsqueeze(-1) < 0.5, -1e9)
    values = masked_x.max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


class KleinHNNLayer(nn.Module):
    """Euclidean-to-Klein mapping layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.manifold = Klein()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h_klein = self.manifold.expmap0(h, c=1.0)
        return self.manifold.proj(h_klein, c=1.0)


class ResidualGraphBlock(nn.Module):
    """Residual graph block with global-context feedback."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.graph_conv = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        batch_size: list
    ) -> torch.Tensor:
        residual = x
        x = self.graph_conv(graph, x)
        x = self.norm1(residual + self.dropout(F.gelu(x)))

        batch_graphs, max_nodes = batch_size
        x_batched = x.reshape(batch_graphs, max_nodes, -1)
        x_batched = x_batched * node_mask.unsqueeze(-1)

        global_context = masked_mean(x_batched, node_mask)
        global_delta = self.global_mlp(global_context).unsqueeze(1)
        x_batched = x_batched + node_mask.unsqueeze(-1) * global_delta

        x_batched = self.norm2(x_batched + self.dropout(self.ffn(x_batched)))
        x_batched = x_batched * node_mask.unsqueeze(-1)
        return x_batched.reshape(batch_graphs * max_nodes, -1)


class AttentionReadout(nn.Module):
    """Attention pooling for graph-level summaries."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        attn = torch.softmax(logits, dim=1)
        attn = attn * mask
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (x * attn.unsqueeze(-1)).sum(dim=1)


class KleinEncoder(nn.Module):
    """
    Graph encoder with residual graph blocks, masked pooling, and conditional outputs.
    """

    def __init__(
        self,
        in_feature_dim: int,
        hidden_layers: list = None,
        graph_latent_dim: int = 1024,
        dropout: float = 0.1,
        cond_dim: int = 64,
        encoder_blocks: int = 4,
        input_proj_dim: int = 64,
        aux_stat_dim: int = 6,
    ):
        super().__init__()
        self.manifold = Klein()
        self.graph_latent_dim = graph_latent_dim
        self.cond_dim = cond_dim
        self.aux_stat_dim = aux_stat_dim

        hidden_layers = hidden_layers or [128, 128, 128, 128]
        hidden_dim = hidden_layers[0]

        self.input_proj = nn.Sequential(
            nn.Linear(in_feature_dim, input_proj_dim),
            nn.GELU(),
            nn.LayerNorm(input_proj_dim),
            nn.Linear(input_proj_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualGraphBlock(hidden_dim, dropout=dropout) for _ in range(encoder_blocks)]
        )
        self.readout_attn = AttentionReadout(hidden_dim)
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.cond_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.mean_layer = node_mlp(hidden_dim, [graph_latent_dim])
        self.log_std_layer = node_mlp(hidden_dim, [graph_latent_dim])
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, aux_stat_dim),
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: torch.Tensor,
        batch_size: list,
        node_mask: torch.Tensor = None,
        return_euclidean: bool = False,
    ):
        batch_graphs, max_nodes = batch_size
        if node_mask is None:
            node_mask = torch.ones(batch_graphs, max_nodes, device=features.device, dtype=features.dtype)

        h = self.input_proj(features)
        h = h.reshape(batch_graphs, max_nodes, -1) * node_mask.unsqueeze(-1)
        h = h.reshape(batch_graphs * max_nodes, -1)

        for block in self.blocks:
            h = block(graph, h, node_mask, batch_size)

        h = h.reshape(batch_graphs, max_nodes, -1)
        attn_pool = self.readout_attn(h, node_mask)
        mean_pool = masked_mean(h, node_mask)
        max_pool = masked_max(h, node_mask)
        graph_repr = self.readout_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1))

        cond_code = self.cond_head(graph_repr)
        mean = self.mean_layer(graph_repr, activation=lambda x: x)
        log_std = self.log_std_layer(graph_repr, activation=lambda x: x)
        aux_stats = self.aux_head(graph_repr)

        h_klein = self.manifold.expmap0(mean, c=1.0)
        h_klein = self.manifold.proj(h_klein, c=1.0)

        if return_euclidean:
            return h_klein, mean, log_std, cond_code, aux_stats, graph_repr
        return h_klein, mean, log_std, cond_code, aux_stats

    def reparameterize(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z_tangent = mean + eps * std
        z_klein = self.manifold.expmap0(z_tangent, c=1.0)
        return self.manifold.proj(z_klein, c=1.0)


class MaskedGraphDecoder(nn.Module):
    """Graph decoder that predicts node presence, edges, and graph-level stats."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        max_nodes: int,
        directed: bool = False,
        node_dim: int = 128,
        edge_dim: int = 16,
        aux_stat_dim: int = 6,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.directed = directed
        self.node_dim = node_dim

        self.slot_embed = nn.Parameter(torch.randn(max_nodes, node_dim) * 0.02)
        self.global_proj = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.film = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim * 2),
        )
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.node_head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, 1),
        )

        self.edge_src = nn.Linear(node_dim, edge_dim)
        self.edge_prod = nn.Linear(node_dim, edge_dim)
        self.edge_diff = nn.Linear(node_dim, edge_dim)
        self.edge_global = nn.Linear(latent_dim + cond_dim, edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Linear(edge_dim * 2, 1),
        )
        self.stats_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, aux_stat_dim),
        )

    def forward(
        self,
        z_tangent: torch.Tensor,
        cond_code: torch.Tensor,
        node_mask: torch.Tensor = None,
    ):
        batch_size = z_tangent.shape[0]
        latent_cond = torch.cat([z_tangent, cond_code], dim=-1)
        global_context = self.global_proj(latent_cond).unsqueeze(1)
        shift, scale = self.film(latent_cond).chunk(2, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)

        slots = self.slot_embed.unsqueeze(0).expand(batch_size, -1, -1)
        slots = slots * (1.0 + scale) + shift + global_context
        slots = slots + self.slot_mlp(slots)

        node_logits = self.node_head(slots).squeeze(-1)

        src = self.edge_src(slots)
        prod = self.edge_prod(slots)
        diff = self.edge_diff(slots)
        global_edge = self.edge_global(latent_cond).unsqueeze(1).unsqueeze(1)

        src_pair = src.unsqueeze(2) + src.unsqueeze(1)
        prod_pair = prod.unsqueeze(2) * prod.unsqueeze(1)
        diff_pair = torch.abs(diff.unsqueeze(2) - diff.unsqueeze(1))
        edge_hidden = src_pair + prod_pair + diff_pair + global_edge
        edge_logits = self.edge_mlp(edge_hidden).squeeze(-1)

        if not self.directed:
            edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))

        diag_idx = torch.arange(self.max_nodes, device=edge_logits.device)
        edge_logits[:, diag_idx, diag_idx] = -20.0

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            edge_logits = edge_logits.masked_fill(pair_mask < 0.5, -20.0)

        stats_pred = self.stats_head(latent_cond)
        return edge_logits, node_logits, stats_pred


class KleinGraphVAE(nn.Module):
    """Klein Graph VAE with conditional decoder outputs."""

    def __init__(
        self,
        encoder: KleinEncoder,
        decoder: nn.Module,
        kernel_model: nn.Module = None,
        auto_encoder: bool = False,
    ):
        super().__init__()
        self.manifold = Klein()
        self.encoder = encoder
        self.decoder = decoder
        self.kernel_model = kernel_model
        self.auto_encoder = auto_encoder
        self.embeding_dim = encoder.graph_latent_dim

    def encode(self, graph, features, batch_size, node_mask=None):
        return self.encoder(graph, features, batch_size, node_mask=node_mask)

    def decode(self, z_klein: torch.Tensor, cond_code: torch.Tensor, node_mask: torch.Tensor = None):
        z_tangent = self.manifold.logmap0(z_klein, c=1.0)
        return self.decoder(z_tangent, cond_code, node_mask=node_mask)

    def reparameterize(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        if self.auto_encoder:
            z_tangent = mean
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            z_tangent = mean + eps * std
        z_klein = self.manifold.expmap0(z_tangent, c=1.0)
        return self.manifold.proj(z_klein, c=1.0)

    def forward(self, graph, features, batch_size, node_mask=None):
        h_klein, mean, log_std, cond_code, encoder_aux = self.encode(
            graph, features, batch_size, node_mask=node_mask
        )
        samples = self.reparameterize(mean, log_std)
        edge_logits, node_logits, stats_pred = self.decode(
            samples, cond_code, node_mask=node_mask
        )
        reconstructed_adj = torch.sigmoid(edge_logits)

        aux_preds = {
            "node_logits": node_logits,
            "stats_pred": stats_pred,
            "encoder_aux": encoder_aux,
        }

        return reconstructed_adj, samples, mean, log_std, cond_code, aux_preds, edge_logits
