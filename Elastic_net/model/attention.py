"""
Grouped Query Attention (GQA) — LLaMA 3.2 style.

Uses fewer KV heads than query heads (4:1 ratio) for efficiency.
- 16 query heads, 4 KV heads
- KV heads are repeated to match query head count
- No bias in any projections
- Causal masking via PyTorch scaled_dot_product_attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads.

    Args:
        x: Tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        n_rep: Number of times to repeat each KV head.

    Returns:
        Tensor of shape (batch, n_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    # (batch, n_kv_heads, 1, seq_len, head_dim) -> (batch, n_kv_heads, n_rep, seq_len, head_dim)
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE (LLaMA 3.2 style).

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads (GQA).
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length for RoPE cache.
        rope_base: Base frequency for RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 512,
        rope_base: float = 500_000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads  # KV repetition factor

        # Q, K, V, O projections — no bias (LLaMA 3.2 style)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim, max_seq_len=max_seq_len, base=rope_base
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Repeat KV heads to match query heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention with causal mask
        # Uses Flash Attention when available (PyTorch 2.1+)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        # Reshape back: (batch, heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.num_heads * self.head_dim)

        # Output projection
        return self.o_proj(attn_output)
