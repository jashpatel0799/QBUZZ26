"""
SwiGLU Feed-Forward Network — LLaMA 3.2 style.

SwiGLU is a gated FFN variant that uses SiLU (Swish) activation:
    FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

No bias in any projection (LLaMA 3.2 convention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension (typically 4× hidden_size).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Gate projection: determines which features to activate
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Up projection: projects to intermediate dimension
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Down projection: projects back to hidden dimension
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Computes: down(SiLU(gate(x)) * up(x))

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of same shape.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
