"""
Rotary Positional Embeddings (RoPE) — LLaMA 3.2 style.

RoPE encodes position information by rotating query and key vectors
in 2D subspaces using sinusoidal frequencies. This allows the model
to learn relative positional relationships.

Base frequency: 500,000 (LLaMA 3.2 value for extended context).
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base frequency for sinusoidal encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 500_000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: θ_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for positions [0, seq_len)."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        # Outer product: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate for full dimension: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for the given sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin), each of shape (seq_len, dim).
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension.

    Splits x into two halves along last dim, then returns [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        cos: Cosine values of shape (seq_len, head_dim).
        sin: Sine values of shape (seq_len, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
