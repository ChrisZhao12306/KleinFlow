"""
DiT (Diffusion Transformer) Architecture for Klein Flow Matching.

This implements the standard DiT architecture with:
- Sinusoidal timestep embeddings
- AdaLN-Zero modulation
- Multi-head self-attention
- Feed-forward network
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (batch_size,) with values in [0, 1]
        Returns:
            Positional embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, dim)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP to process timestep embeddings."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPosEmbed(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    DiT Block with AdaLN-Zero modulation.
    
    Structure:
    1. LayerNorm -> AdaLN modulation -> Multi-head Self-Attention -> Scale gate
    2. LayerNorm -> AdaLN modulation -> Feed-forward -> Scale gate
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = FeedForward(hidden_dim, mlp_ratio, dropout)
        
        # AdaLN-Zero modulation: outputs 6 modulation parameters
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
        # Initialize gate parameters to zero for zero-initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            t_emb: Timestep embedding of shape (batch_size, hidden_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Get modulation parameters
        modulation_params = self.adaLN_modulation(t_emb)  # (batch_size, 6 * hidden_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation_params.chunk(6, dim=-1)
        
        # Expand modulation params for sequence dimension
        # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)
        
        # Attention block with AdaLN-Zero
        x_norm = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate1 * attn_out
        
        # FFN block with AdaLN-Zero
        x_norm = modulate(self.norm2(x), shift2, scale2)
        ffn_out = self.ffn(x_norm)
        x = x + gate2 * ffn_out
        
        return x






class ConditionEmbedding(nn.Module):
    """MLP encoder for graph-level conditioning vectors."""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class ConditionalDiT(nn.Module):
    """
    Conditional latent DiT that operates on learned tokens.

    The input vector is projected into a short token sequence so attention remains
    meaningful even though the latent is originally a single vector.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cond_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        seq_len: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, seq_len * hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.time_embed = TimestepEmbedding(hidden_dim, hidden_dim)
        self.cond_embed = ConditionEmbedding(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.output_proj = nn.Linear(seq_len * hidden_dim, output_dim)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = self.input_proj(x).view(batch_size, self.seq_len, self.hidden_dim)
        tokens = tokens + self.pos_embed

        mod_emb = self.time_embed(t) + self.cond_embed(cond)
        for block in self.blocks:
            tokens = block(tokens, mod_emb)

        shift, scale = self.final_adaLN(mod_emb).chunk(2, dim=-1)
        tokens = modulate(self.final_norm(tokens), shift.unsqueeze(1), scale.unsqueeze(1))
        tokens = tokens.reshape(batch_size, self.seq_len * self.hidden_dim)
        return self.output_proj(tokens)
