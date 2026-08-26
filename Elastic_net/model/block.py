"""
Transformer Block — LLaMA 3.2 style.

Architecture per block:
    Input → RMSNorm → GQA → Residual → RMSNorm → SwiGLU FFN → Residual → Output

Supports optional stochastic depth (LayerDrop):
    If stochastic_depth_prob > 0 and training, the block may be randomly skipped.
    If stochastic_depth_prob == 0, block always runs (zero overhead).
"""

import torch
import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    """Single transformer decoder block (LLaMA 3.2 style).

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Dimension per attention head.
        max_seq_len: Maximum sequence length for RoPE.
        rope_base: Base frequency for RoPE.
        rms_norm_eps: Epsilon for RMSNorm.
        stochastic_depth_prob: Probability of dropping this block during training.
            0.0 means always run (disabled), >0 enables stochastic depth.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        rope_base: float = 500_000.0,
        rms_norm_eps: float = 1e-5,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        # Pre-attention norm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Grouped Query Attention
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        # Pre-FFN norm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # SwiGLU FFN
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size)

        # Stochastic depth
        self.stochastic_depth_prob = stochastic_depth_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional stochastic depth.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of same shape.
        """
        # Stochastic depth: randomly skip this block during training
        if self.stochastic_depth_prob > 0.0 and self.training:
            if torch.rand(1, device=x.device).item() < self.stochastic_depth_prob:
                # Skip this block entirely — return input unchanged
                return x

        # Pre-norm → Attention → Residual
        h = x + self.self_attn(self.input_layernorm(x))
        # Pre-norm → FFN → Residual
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out
